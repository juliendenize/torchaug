from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.utils._pytree import tree_flatten

import torchaug.transforms as transforms
import torchaug.transforms.functional as F
from tests.utils._make_tensors import (
    make_nested_bounding_boxes,
    make_nested_images,
    make_nested_segmentation_masks,
    make_nested_videos,
)
from torchaug import ta_tensors
from torchaug.ta_tensors import set_return_type
from torchaug.ta_tensors._batch_concatenated_ta_tensor import _BatchConcatenatedTATensor
from torchaug.transforms import RandomApplyTransform
from torchaug.transforms.functional._utils._kernel import _get_kernel

from ..utils import (
    assert_equal,
    assert_not_equal,
    freeze_rng_state,
    make_batch_images,
    make_batch_videos,
    make_image,
    make_video,
)
from ..utils._transform_utils import _make_transform_batch_sample, _make_transform_sample


class TestRandomApplyTransform:
    def test_p_value(self):
        with pytest.raises(ValueError):
            RandomApplyTransform(p=-0.5)
        with pytest.raises(ValueError):
            RandomApplyTransform(p=1.5)

    @pytest.mark.parametrize("make_input", [make_image, make_video, make_batch_images, make_batch_videos])
    @pytest.mark.parametrize("p", [0, 0.5, 1.0])
    def test_forward_random_single(self, make_input, p):
        batch = make_input in (make_batch_images, make_batch_videos)
        transform = transforms.RandomColorJitter(brightness=0.5, p=p)
        transform._receive_flatten_inputs = True

        input_sample = _make_transform_sample(transform, image_or_video=make_input(), adapter=None, batch=batch)
        input_flat, spec = tree_flatten(input_sample)

        output_tensor = transform(*input_flat)
        needs_transform_list = transform._needs_transform_list(input_flat)
        for inpt, opt, need_transform in zip(input_flat, output_tensor, needs_transform_list):
            assert type(inpt) is type(opt)
            if not need_transform:
                assert opt is inpt
                return
            try:
                _get_kernel(F.adjust_brightness, type(inpt), allow_passthrough=False)
                has_kernel = True
            except TypeError:
                has_kernel = False
            if has_kernel and p == 1.0:
                assert_not_equal(opt, inpt)
            else:
                if not has_kernel or p == 0:
                    assert_equal(opt, inpt)

    @pytest.mark.parametrize("make_input", [make_image, make_video, make_batch_images, make_batch_videos])
    def test_forward_single_reshape(self, make_input):
        batch = make_input in (make_batch_images, make_batch_videos)
        transform = transforms.RandomResizedCrop(size=[224, 224])
        transform._receive_flatten_inputs = True

        input_sample = _make_transform_sample(transform, image_or_video=make_input(), adapter=None, batch=batch)
        input_flat, _ = tree_flatten(input_sample)

        output_tensor = transform(*input_flat)
        needs_transform_list = transform._needs_transform_list(input_flat)
        for inpt, opt, need_transform in zip(input_flat, output_tensor, needs_transform_list):
            assert type(inpt) is type(opt)
            if not need_transform:
                assert opt is inpt
                return
            try:
                _get_kernel(F.resized_crop, type(inpt), allow_passthrough=False)
                has_kernel = True
            except TypeError:
                has_kernel = False
            if has_kernel:
                assert_not_equal(opt, inpt)
            else:
                assert_equal(opt, inpt)

    @pytest.mark.parametrize("make_input", [make_batch_images, make_batch_videos])
    @pytest.mark.parametrize("p", [0, 0.5, 1.0])
    @pytest.mark.parametrize("num_chunks", [-1, 1, 2, 4])
    @pytest.mark.parametrize("permute_chunks", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("batch_inplace", [False, True])
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_forward_batch(self, make_input, p, num_chunks, permute_chunks, batch_size, batch_inplace, seed):
        transform = transforms.RandomColorJitter(
            brightness=0.5,
            p=p,
            num_chunks=num_chunks,
            permute_chunks=permute_chunks,
            batch_inplace=batch_inplace,
            batch_transform=True,
        )
        transform._receive_flatten_inputs = True

        if num_chunks == -1:
            num_chunks = batch_size

        input_sample = _make_transform_batch_sample(
            transform, image_or_video=make_input(batch_dims=(batch_size,)), adapter=None, batch_size=batch_size
        )

        input_flat, _ = tree_flatten(input_sample)
        cloned_input = deepcopy(input_sample)
        cloned_input_flat, _ = tree_flatten(cloned_input)

        with freeze_rng_state():
            torch.manual_seed(seed)
            output_flat = transform(*input_flat)

        if p == 0:
            for opt, inpt in zip(output_flat, input_flat):
                assert type(opt) is type(inpt)
                assert opt is inpt
            return

        with freeze_rng_state():
            torch.manual_seed(seed)
            needs_transform_list = transform._needs_transform_list(cloned_input_flat)
            if p == 1:
                indices_transform = torch.tensor(list(range(batch_size)), dtype=torch.long)
            else:
                indices_transform = transform._get_indices_transform(
                    batch_size,
                    "cpu",
                )
            num_chunks = min(num_chunks, len(indices_transform))

            if num_chunks == 0:
                for opt, inpt in zip(output_flat, input_flat):
                    assert type(opt) is type(inpt)
                    assert opt is inpt
                return

            chunks_indices = transform._get_chunks_indices(len(indices_transform), num_chunks, "cpu")

            params = transform._get_params(
                cloned_input_flat,
                len(chunks_indices),
                chunks_indices,
            )

            for inpt, cloned_inpt, opt, need_transform in zip(
                input_flat,
                cloned_input_flat,
                output_flat,
                needs_transform_list,
            ):
                if not need_transform:
                    if isinstance(opt, torch.Tensor):
                        if inpt.isnan().any():
                            continue
                        assert_equal(opt, inpt)
                    elif isinstance(opt, np.ndarray):
                        assert_equal(opt, inpt)
                    else:
                        assert opt == inpt
                    continue
                else:
                    assert type(opt) is type(cloned_inpt)

                is_ta_inpt = isinstance(inpt, ta_tensors.TATensor)
                is_contatenated_batch_ta_tensors = isinstance(inpt, _BatchConcatenatedTATensor)
                indices_non_transform = torch.tensor(
                    [i for i in range(batch_size) if i not in indices_transform], dtype=torch.long
                )

                if is_contatenated_batch_ta_tensors:
                    transform_cloned_inpt = cloned_inpt.get_chunk(indices_transform)
                    transform_opt = opt.get_chunk(indices_transform)
                    not_transform_inpt = inpt.get_chunk(indices_non_transform)
                    not_transform_opt = opt.get_chunk(indices_non_transform)
                else:
                    with set_return_type("TATensor" if is_ta_inpt else "Tensor"):
                        transform_cloned_inpt = cloned_inpt[indices_transform]
                        transform_opt = opt[indices_transform]
                        not_transform_inpt = inpt[indices_non_transform]
                        not_transform_opt = opt[indices_non_transform]

                assert_equal(not_transform_inpt, not_transform_opt)

                for i, chunk_indices in enumerate(chunks_indices):
                    if is_contatenated_batch_ta_tensors:
                        chunk_cloned_inpt = transform_cloned_inpt.get_chunk(chunk_indices)
                        chunk_opt = transform_opt.get_chunk(chunk_indices)
                    else:
                        with set_return_type("TATensor" if is_ta_inpt else "Tensor"):
                            chunk_cloned_inpt = transform_cloned_inpt[chunk_indices]
                            chunk_opt = transform_opt[chunk_indices]

                    try:
                        _get_kernel(F.adjust_brightness, type(inpt), allow_passthrough=False)
                        has_kernel = True
                    except TypeError:
                        has_kernel = False

                    if has_kernel:
                        chunk_expected_opt = F.adjust_brightness_batch(
                            chunk_cloned_inpt, brightness_factor=params[i]["brightness_factor"]
                        )
                        assert_equal(chunk_opt, chunk_expected_opt)
                    else:
                        assert_equal(chunk_cloned_inpt, chunk_opt)

    @pytest.mark.parametrize("make_input", [make_batch_images, make_batch_videos])
    @pytest.mark.parametrize("num_chunks", [-1, 1, 2, 4])
    @pytest.mark.parametrize("permute_chunks", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_forward_batch_reshape(self, make_input, num_chunks, permute_chunks, batch_size, seed):
        transform = transforms.RandomResizedCrop(
            size=[3, 3],
            num_chunks=num_chunks,
            permute_chunks=permute_chunks,
            batch_transform=True,
        )
        transform._receive_flatten_inputs = True

        if num_chunks == -1:
            num_chunks = batch_size

        input_sample = _make_transform_batch_sample(
            transform, image_or_video=make_input(batch_dims=(batch_size,)), adapter=None, batch_size=batch_size
        )

        input_flat, _ = tree_flatten(input_sample)
        cloned_input = deepcopy(input_sample)
        cloned_input_flat, _ = tree_flatten(cloned_input)

        with freeze_rng_state():
            torch.manual_seed(seed)
            output_flat = transform(*input_flat)

        with freeze_rng_state():
            torch.manual_seed(seed)
            needs_transform_list = transform._needs_transform_list(cloned_input_flat)
            indices_transform = torch.tensor(list(range(batch_size)), dtype=torch.long)
            num_chunks = min(num_chunks, len(indices_transform))
            chunks_indices = transform._get_chunks_indices(len(indices_transform), num_chunks, "cpu")

            params = transform._get_params(
                [
                    inpt_flat
                    for inpt_flat, need_transform in zip(cloned_input_flat, needs_transform_list)
                    if need_transform
                ],
                len(chunks_indices),
                chunks_indices,
            )

            for inpt, cloned_inpt, opt, need_transform in zip(
                input_flat,
                cloned_input_flat,
                output_flat,
                needs_transform_list,
            ):
                if not need_transform:
                    if isinstance(opt, torch.Tensor):
                        if inpt.isnan().any():
                            continue
                        assert_equal(opt, inpt)
                    elif isinstance(opt, np.ndarray):
                        assert_equal(opt, inpt)
                    else:
                        assert opt == inpt
                    continue
                else:
                    assert type(opt) is type(cloned_inpt)

                is_ta_inpt = isinstance(inpt, ta_tensors.TATensor)
                is_contatenated_batch_ta_tensors = isinstance(inpt, _BatchConcatenatedTATensor)

                if is_contatenated_batch_ta_tensors:
                    transform_cloned_inpt = cloned_inpt.get_chunk(indices_transform)
                    transform_opt = opt.get_chunk(indices_transform)
                else:
                    with set_return_type("TATensor" if is_ta_inpt else "Tensor"):
                        transform_cloned_inpt = cloned_inpt[indices_transform]
                        transform_opt = opt[indices_transform]

                for i, chunk_indices in enumerate(chunks_indices):
                    if is_contatenated_batch_ta_tensors:
                        chunk_cloned_inpt = transform_cloned_inpt.get_chunk(chunk_indices)
                        chunk_opt = transform_opt.get_chunk(chunk_indices)
                    else:
                        with set_return_type("TATensor" if is_ta_inpt else "Tensor"):
                            chunk_cloned_inpt = transform_cloned_inpt[chunk_indices]
                            chunk_opt = transform_opt[chunk_indices]

                    try:
                        _get_kernel(F.resized_crop, type(cloned_inpt), allow_passthrough=False)
                        has_kernel = True
                    except TypeError:
                        has_kernel = False

                    if has_kernel:
                        chunk_expected_opt = F.resized_crop(chunk_cloned_inpt, size=(3, 3), **params[i])
                        assert_equal(chunk_opt, chunk_expected_opt)
                    else:
                        assert_equal(chunk_cloned_inpt, chunk_opt)

    @pytest.mark.parametrize(
        "make_input",
        [make_nested_images, make_nested_videos, make_nested_bounding_boxes, make_nested_segmentation_masks],
    )
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seed", list(range(5)))
    @pytest.mark.parametrize(
        "transform",
        [
            transforms.RandomColorJitter(
                brightness=0.5,
                p=0.5,
            ),
            transforms.RandomResizedCrop(
                size=[3, 3],
            ),
        ],
    )
    def test_forward_nested(self, make_input, batch_size, seed, transform):
        nested_input = make_input(batch_dims=batch_size)
        batch_input = list(nested_input.tensors)
        with freeze_rng_state():
            torch.manual_seed(seed)
            nested_output = transform(nested_input)

        for nest_in, nest_out in zip(nested_input, nested_output):
            assert isinstance(nest_out, type(nest_in))

        with freeze_rng_state():
            torch.manual_seed(seed)
            batch_output = [transform(batch_input) for batch_input in batch_input]

        for nested, batch in zip(nested_output, batch_output):
            torch.testing.assert_close(nested, batch)
