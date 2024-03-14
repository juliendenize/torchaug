import inspect
import pickle
import re
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from torch.testing import assert_close
from torch.utils._pytree import tree_flatten
from torchvision.transforms.v2._utils import check_type

import torchaug.transforms as transforms
from torchaug import ta_tensors
from torchaug.ta_tensors import _BatchConcatenatedTATensor
from torchaug.transforms import functional as F
from torchaug.transforms.functional._utils._tensor import (
    _max_value as get_max_value,
)
from torchaug.transforms.functional._utils._tensor import (
    is_pure_tensor,
)

from ._common import (
    assert_equal,
    cache,
    freeze_rng_state,
    ignore_jit_no_profile_information_warning,
    set_rng_seed,
)
from ._make_tensors import (
    make_batch_bounding_boxes,
    make_batch_detection_masks,
    make_batch_images,
    make_batch_segmentation_masks,
    make_batch_videos,
    make_bounding_boxes,
    make_detection_masks,
    make_image,
    make_segmentation_mask,
    make_video,
)


EXHAUSTIVE_TYPE_FILLS = [
    None,
    1,
    0.5,
    [1],
    [0.2],
    (0,),
    (0.7,),
    [1, 0, 1],
    [0.1, 0.2, 0.3],
    (0, 1, 0),
    (0.9, 0.234, 0.314),
]
CORRECTNESS_FILLS = [
    v for v in EXHAUSTIVE_TYPE_FILLS if v is None or isinstance(v, float) or (isinstance(v, list) and len(v) > 1)
]


@cache
def _script(obj):
    try:
        return torch.jit.script(obj)
    except Exception as error:
        name = getattr(obj, "__name__", obj.__class__.__name__)
        raise AssertionError(f"Trying to `torch.jit.script` '{name}' raised the error above.") from error


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


@pytest.fixture(autouse=True)
def fix_rng_seed():
    set_rng_seed(0)
    yield


def _to_tolerances(maybe_tolerance_dict):
    if not isinstance(maybe_tolerance_dict, dict):
        return dict(rtol=None, atol=None)

    tolerances = dict(rtol=0, atol=0)
    tolerances.update(maybe_tolerance_dict)
    return tolerances


def check_kernel_cuda_vs_cpu(kernel, input, *args, rtol, atol, **kwargs):
    """Checks if the kernel produces closes results for inputs on GPU and CPU."""
    if input.device.type != "cuda":
        return

    input_cuda = input.as_subclass(torch.Tensor)
    input_cpu = input_cuda.to("cpu")

    with freeze_rng_state():
        actual = kernel(input_cuda, *args, **kwargs)
    with freeze_rng_state():
        expected = kernel(input_cpu, *args, **kwargs)

    assert_close(actual, expected, check_device=False, rtol=rtol, atol=atol)


@cache
def _script(obj):
    try:
        return torch.jit.script(obj)
    except Exception as error:
        name = getattr(obj, "__name__", obj.__class__.__name__)
        raise AssertionError(f"Trying to `torch.jit.script` '{name}' raised the error above.") from error


def _check_kernel_scripted_vs_eager(kernel, input, *args, rtol, atol, **kwargs):
    """Checks if the kernel is scriptable and if the scripted output is close to the eager one."""
    if input.device.type != "cpu":
        return

    kernel_scripted = _script(kernel)

    input = input.as_subclass(torch.Tensor)
    with ignore_jit_no_profile_information_warning():
        actual = kernel_scripted(input, *args, **kwargs)
    expected = kernel(input, *args, **kwargs)

    assert_close(actual, expected, rtol=rtol, atol=atol)


def _check_kernel_batched_vs_unbatched(kernel, input, *args, rtol, atol, **kwargs):
    """Checks if the kernel produces close results for batched and unbatched inputs."""
    unbatched_input = input.as_subclass(torch.Tensor)

    for batch_dims in [(2,), (2, 1)]:
        repeats = [*batch_dims, *[1] * input.ndim]

        actual = kernel(unbatched_input.repeat(repeats), *args, **kwargs)

        expected = kernel(unbatched_input, *args, **kwargs)
        # We can't directly call `.repeat()` on the output, since some kernel also return some additional metadata
        if isinstance(expected, torch.Tensor):
            expected = expected.repeat(repeats)
        else:
            tensor, *metadata = expected
            expected = (tensor.repeat(repeats), *metadata)

        assert_close(actual, expected, rtol=rtol, atol=atol)

    for degenerate_batch_dims in [(0,), (5, 0), (0, 5)]:
        degenerate_batched_input = torch.empty(
            degenerate_batch_dims + input.shape, dtype=input.dtype, device=input.device
        )

        output = kernel(degenerate_batched_input, *args, **kwargs)
        # Most kernels just return a tensor, but some also return some additional metadata
        if not isinstance(output, torch.Tensor):
            output, *_ = output

        assert output.shape[: -input.ndim] == degenerate_batch_dims


def _check_batch_kernel_leading_dims(kernel, input, *args, rtol, atol, **kwargs):
    """Checks if the kernel produces close results for batched and unbatched inputs."""
    unbatched_input = input.as_subclass(torch.Tensor)

    for leading_dims in [(1,), (1, 1)]:
        repeats = [1, *leading_dims, *[1] * (input.ndim - 1)]
        view = [-1, *leading_dims, *unbatched_input.shape[1:]]

        actual = kernel(unbatched_input.view(view).repeat(repeats), *args, **kwargs)

        expected = kernel(unbatched_input, *args, **kwargs)
        # We can't directly call `.repeat()` on the output, since some kernel also return some additional metadata
        repeats = [1, *leading_dims, *[1] * (input.ndim - 1)]
        if isinstance(expected, torch.Tensor):
            view = [-1, *leading_dims, *expected.shape[1:]]
            expected = expected.view(view).repeat(repeats)
        else:
            tensor, *metadata = expected
            view = [-1, *leading_dims, *tensor.shape[1:]]
            expected = (tensor.view(view).repeat(repeats), *metadata)

        assert_close(actual, expected, rtol=rtol, atol=atol)

    for degenerate_leading_dims in [(0,), (5, 0), (0, 5)]:
        degenerate_batched_input = torch.empty(
            [input.shape[0]] + list(degenerate_leading_dims) + list(input.shape[1:]),
            dtype=input.dtype,
            device=input.device,
        )

        output = kernel(degenerate_batched_input, *args, **kwargs)
        # Most kernels just return a tensor, but some also return some additional metadata
        if not isinstance(output, torch.Tensor):
            output, *_ = output
        assert output.shape[1 : -(input.ndim - 1)] == degenerate_leading_dims


def check_kernel(
    kernel,
    input,
    *args,
    check_cuda_vs_cpu=True,
    check_scripted_vs_eager=True,
    check_batched_vs_unbatched=True,
    check_batch_kernel_leading_dims=True,
    **kwargs,
):
    initial_input_version = input._version

    output = kernel(input.as_subclass(torch.Tensor), *args, **kwargs)
    # Most kernels just return a tensor, but some also return some additional metadata
    if not isinstance(output, torch.Tensor):
        output, *_ = output

    # check that no inplace operation happened
    assert input._version == initial_input_version

    if kernel not in {F.to_dtype_image, F.to_dtype_video, F.to_dtype}:
        assert output.dtype == input.dtype
    assert output.device == input.device

    if check_cuda_vs_cpu:
        check_kernel_cuda_vs_cpu(kernel, input, *args, **kwargs, **_to_tolerances(check_cuda_vs_cpu))

    if check_scripted_vs_eager:
        _check_kernel_scripted_vs_eager(kernel, input, *args, **kwargs, **_to_tolerances(check_scripted_vs_eager))

    is_batch_input = type(input) in (
        ta_tensors.BatchImages,
        ta_tensors.BatchVideos,
        ta_tensors.BatchBoundingBoxes,
        ta_tensors.BatchMasks,
    )

    if check_batched_vs_unbatched and not is_batch_input:
        _check_kernel_batched_vs_unbatched(
            kernel, input, *args, **kwargs, **_to_tolerances(check_batched_vs_unbatched)
        )

    if check_batch_kernel_leading_dims and is_batch_input:
        _check_batch_kernel_leading_dims(kernel, input, *args, **kwargs, **_to_tolerances(check_batched_vs_unbatched))


def _check_functional_scripted_smoke(functional, input, *args, **kwargs):
    """Checks if the functional can be scripted and the scripted version can be called without error."""
    if not isinstance(input, ta_tensors.Image):
        return

    functional_scripted = _script(functional)
    with ignore_jit_no_profile_information_warning():
        functional_scripted(input.as_subclass(torch.Tensor), *args, **kwargs)


def check_functional(functional, input, *args, check_scripted_smoke=True, **kwargs):
    unknown_input = object()
    with pytest.raises(TypeError, match=re.escape(str(type(unknown_input)))):
        functional(unknown_input, *args, **kwargs)

    with mock.patch("torch._C._log_api_usage_once", wraps=torch._C._log_api_usage_once) as spy:
        output = functional(input, *args, **kwargs)

        spy.assert_any_call(f"{functional.__module__}.{functional.__name__}")

    assert isinstance(output, type(input))

    if (
        isinstance(input, (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes))
        and functional is not F.convert_bounding_box_format
    ):
        assert output.format == input.format

    if check_scripted_smoke:
        _check_functional_scripted_smoke(functional, input, *args, **kwargs)


def check_functional_kernel_signature_match(functional, *, kernel, input_type):
    """Checks if the signature of the functional matches the kernel signature."""
    functional_params = list(inspect.signature(functional).parameters.values())[1:]
    kernel_params = list(inspect.signature(kernel).parameters.values())[1:]

    if issubclass(input_type, ta_tensors.TATensor):
        # We filter out metadata that is implicitly passed to the functional through the input ta_tensor, but has to be
        # explicitly passed to the kernel.
        explicit_metadata = {
            ta_tensors.BoundingBoxes: {"format", "canvas_size"},
            ta_tensors.BatchBoundingBoxes: {"format", "canvas_size", "idx_sample"},
        }
        kernel_params = [
            param for param in kernel_params if param.name not in explicit_metadata.get(input_type, set())
        ]

    functional_params = iter(functional_params)
    for functional_param, kernel_param in zip(functional_params, kernel_params):
        try:
            # In general, the functional parameters are a superset of the kernel parameters. Thus, we filter out
            # functional parameters that have no kernel equivalent while keeping the order intact.
            while functional_param.name != kernel_param.name:
                functional_param = next(functional_params)
        except StopIteration:
            raise AssertionError(
                f"Parameter `{kernel_param.name}` of kernel `{kernel.__name__}` "
                f"has no corresponding parameter on the functional `{functional.__name__}`."
            ) from None

        assert functional_param == kernel_param


def _make_transform_sample(transform, *, image_or_video, adapter, batch=False):
    device = image_or_video.device if isinstance(image_or_video, torch.Tensor) else "cpu"
    size = F.get_size(image_or_video)
    input = (
        dict(
            image_or_video=image_or_video,
            image_ta_tensor=make_image(size, device=device),
            video_ta_tensor=make_video(size, device=device),
            bounding_boxes_xyxy=make_bounding_boxes(size, format=ta_tensors.BoundingBoxFormat.XYXY, device=device),
            bounding_boxes_xywh=make_bounding_boxes(size, format=ta_tensors.BoundingBoxFormat.XYWH, device=device),
            bounding_boxes_cxcywh=make_bounding_boxes(size, format=ta_tensors.BoundingBoxFormat.CXCYWH, device=device),
            bounding_boxes_degenerate_xyxy=ta_tensors.BoundingBoxes(
                [
                    [0, 0, 0, 0],  # no height or width
                    [0, 0, 0, 1],  # no height
                    [0, 0, 1, 0],  # no width
                    [2, 0, 1, 1],  # x1 > x2, y1 < y2
                    [0, 2, 1, 1],  # x1 < x2, y1 > y2
                    [2, 2, 1, 1],  # x1 > x2, y1 > y2
                ],
                format=ta_tensors.BoundingBoxFormat.XYXY,
                canvas_size=size,
                device=device,
            ),
            bounding_boxes_degenerate_xywh=ta_tensors.BoundingBoxes(
                [
                    [0, 0, 0, 0],  # no height or width
                    [0, 0, 0, 1],  # no height
                    [0, 0, 1, 0],  # no width
                    [0, 0, 1, -1],  # negative height
                    [0, 0, -1, 1],  # negative width
                    [0, 0, -1, -1],  # negative height and width
                ],
                format=ta_tensors.BoundingBoxFormat.XYWH,
                canvas_size=size,
                device=device,
            ),
            bounding_boxes_degenerate_cxcywh=ta_tensors.BoundingBoxes(
                [
                    [0, 0, 0, 0],  # no height or width
                    [0, 0, 0, 1],  # no height
                    [0, 0, 1, 0],  # no width
                    [0, 0, 1, -1],  # negative height
                    [0, 0, -1, 1],  # negative width
                    [0, 0, -1, -1],  # negative height and width
                ],
                format=ta_tensors.BoundingBoxFormat.CXCYWH,
                canvas_size=size,
                device=device,
            ),
            detection_mask=make_detection_masks(size, device=device),
            segmentation_mask=make_segmentation_mask(size, device=device),
            int=0,
            float=0.0,
            bool=True,
            none=None,
            str="str",
            path=Path.cwd(),
            object=object(),
            tensor=torch.empty(5),
            array=np.empty(5),
        )
        if not batch
        else dict(
            image_or_video=image_or_video,
            batch_images_ta_tensor=make_batch_images(size, device=device),
            batch_videos_ta_tensor=make_batch_videos(size, device=device),
            batch_bounding_boxes_xyxy=make_batch_bounding_boxes(
                size, format=ta_tensors.BoundingBoxFormat.XYXY, device=device
            ),
            batch_bounding_boxes_xywh=make_batch_bounding_boxes(
                size, format=ta_tensors.BoundingBoxFormat.XYWH, device=device
            ),
            batch_bounding_boxes_cxcywh=make_batch_bounding_boxes(
                size, format=ta_tensors.BoundingBoxFormat.CXCYWH, device=device
            ),
            batch_bounding_boxes_degenerate_xyxy=ta_tensors.BatchBoundingBoxes(
                [
                    [0, 0, 0, 0],  # no height or width
                    [0, 0, 0, 1],  # no height
                    [0, 0, 1, 0],  # no width
                    [2, 0, 1, 1],  # x1 > x2, y1 < y2
                    [0, 2, 1, 1],  # x1 < x2, y1 > y2
                    [2, 2, 1, 1],  # x1 > x2, y1 > y2
                ],
                format=ta_tensors.BoundingBoxFormat.XYXY,
                canvas_size=size,
                idx_sample=[0, 3, 6],
                device=device,
            ),
            batch_bounding_boxes_degenerate_xywh=ta_tensors.BatchBoundingBoxes(
                [
                    [0, 0, 0, 0],  # no height or width
                    [0, 0, 0, 1],  # no height
                    [0, 0, 1, 0],  # no width
                    [0, 0, 1, -1],  # negative height
                    [0, 0, -1, 1],  # negative width
                    [0, 0, -1, -1],  # negative height and width
                ],
                format=ta_tensors.BoundingBoxFormat.XYWH,
                canvas_size=size,
                idx_sample=[0, 3, 6],
                device=device,
            ),
            batch_bounding_boxes_degenerate_cxcywh=ta_tensors.BatchBoundingBoxes(
                [
                    [0, 0, 0, 0],  # no height or width
                    [0, 0, 0, 1],  # no height
                    [0, 0, 1, 0],  # no width
                    [0, 0, 1, -1],  # negative height
                    [0, 0, -1, 1],  # negative width
                    [0, 0, -1, -1],  # negative height and width
                ],
                format=ta_tensors.BoundingBoxFormat.CXCYWH,
                canvas_size=size,
                idx_sample=[0, 3, 6],
                device=device,
            ),
            batch_detection_masks=make_batch_detection_masks(size, device=device),
            batch_segmentation_masks=make_batch_segmentation_masks(size, device=device),
            int=0,
            float=0.0,
            bool=True,
            none=None,
            str="str",
            path=Path.cwd(),
            object=object(),
            tensor=torch.empty(5),
            array=np.empty(5),
        )
    )
    if adapter is not None:
        input = adapter(transform, input, device)
    return input


def _make_transform_batch_sample(transform, *, image_or_video, adapter, batch_size):
    device = image_or_video.device if isinstance(image_or_video, torch.Tensor) else "cpu"
    size = F.get_size(image_or_video)
    input = dict(
        image_or_video=image_or_video,
        batch_images_ta_tensor=make_batch_images(size, device=device, batch_dims=(batch_size,)),
        batch_videos_ta_tensor=make_batch_videos(size, device=device, batch_dims=(batch_size,)),
        batch_bounding_boxes_xyxy=make_batch_bounding_boxes(
            size,
            format=ta_tensors.BoundingBoxFormat.XYXY,
            device=device,
            batch_dims=(batch_size,),
        ),
        batch_bounding_boxes_xywh=make_batch_bounding_boxes(
            size,
            format=ta_tensors.BoundingBoxFormat.XYWH,
            device=device,
            batch_dims=(batch_size,),
        ),
        batch_bounding_boxes_cxcywh=make_batch_bounding_boxes(
            size,
            format=ta_tensors.BoundingBoxFormat.CXCYWH,
            device=device,
            batch_dims=(batch_size,),
        ),
        batch_bounding_boxes_degenerate_xyxy=ta_tensors.BatchBoundingBoxes(
            [
                [0, 0, 0, 0],  # no height or width
                [0, 0, 0, 1],  # no height
                [0, 0, 1, 0],  # no width
                [2, 0, 1, 1],  # x1 > x2, y1 < y2
                [0, 2, 1, 1],  # x1 < x2, y1 > y2
                [2, 2, 1, 1],  # x1 > x2, y1 > y2
            ]
            * batch_size,
            format=ta_tensors.BoundingBoxFormat.XYXY,
            canvas_size=size,
            idx_sample=torch.tensor([0] + [6] * batch_size, dtype=torch.long).cumsum(0).tolist(),
            device=device,
        ),
        batch_bounding_boxes_degenerate_xywh=ta_tensors.BatchBoundingBoxes(
            [
                [0, 0, 0, 0],  # no height or width
                [0, 0, 0, 1],  # no height
                [0, 0, 1, 0],  # no width
                [0, 0, 1, -1],  # negative height
                [0, 0, -1, 1],  # negative width
                [0, 0, -1, -1],  # negative height and width
            ]
            * batch_size,
            format=ta_tensors.BoundingBoxFormat.XYWH,
            canvas_size=size,
            idx_sample=torch.tensor([0] + [6] * batch_size, dtype=torch.long).cumsum(0).tolist(),
            device=device,
        ),
        batch_bounding_boxes_degenerate_cxcywh=ta_tensors.BatchBoundingBoxes(
            [
                [0, 0, 0, 0],  # no height or width
                [0, 0, 0, 1],  # no height
                [0, 0, 1, 0],  # no width
                [0, 0, 1, -1],  # negative height
                [0, 0, -1, 1],  # negative width
                [0, 0, -1, -1],  # negative height and width
            ]
            * batch_size,
            format=ta_tensors.BoundingBoxFormat.CXCYWH,
            canvas_size=size,
            idx_sample=torch.tensor([0] + [6] * batch_size, dtype=torch.long).cumsum(0).tolist(),
            device=device,
        ),
        batch_detection_masks=make_batch_detection_masks(size=size, device=device, batch_dims=(batch_size,)),
        batch_segmentation_masks=make_batch_segmentation_masks(size=size, device=device, batch_dims=(batch_size,)),
        int=0,
        float=0.0,
        bool=True,
        none=None,
        str="str",
        path=Path.cwd(),
        object=object(),
        tensor=torch.empty(5),
        array=np.empty(5),
    )
    if adapter is not None:
        input = adapter(transform, input, device)
    return input


def _check_transform_sample_input_smoke(transform, input, *, adapter, batch=False):
    # This is a bunch of input / output convention checks, using a big sample with different parts as input.

    if not batch:
        if not check_type(
            input,
            (
                is_pure_tensor,
                ta_tensors.Image,
                ta_tensors.Video,
                ta_tensors.BatchImages,
                ta_tensors.BatchVideos,
            ),
        ):
            return
    else:
        if not check_type(input, (is_pure_tensor, ta_tensors.BatchImages, ta_tensors.BatchVideos)):
            return

    sample = _make_transform_sample(
        # adapter might change transform inplace
        transform=transform if adapter is None else deepcopy(transform),
        image_or_video=input,
        adapter=adapter,
        batch=batch,
    )

    for container_type in [dict, list, tuple]:
        if container_type is dict:
            input = sample
        else:
            input = container_type(sample.values())

        input_flat, input_spec = tree_flatten(input)

        with freeze_rng_state():
            torch.manual_seed(0)
            output = transform(input)
        output_flat, output_spec = tree_flatten(output)

        assert output_spec == input_spec

        for output_item, input_item, should_be_transformed in zip(
            output_flat,
            input_flat,
            transforms.RandomApplyTransform()._needs_transform_list(input_flat),
        ):
            if should_be_transformed:
                assert type(output_item) is type(input_item)
            else:
                assert output_item is input_item

    # Enforce that the transform does not turn a degenerate bounding box, e.g. marked by RandomIoUCrop (or any other
    # future transform that does this), back into a valid one.
    for degenerate_bounding_boxes in (
        bounding_box
        for name, bounding_box in sample.items()
        if "degenerate" in name and isinstance(bounding_box, (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes))
    ):
        sample = dict(
            boxes=degenerate_bounding_boxes,
            labels=torch.randint(
                10,
                (degenerate_bounding_boxes.shape[0],),
                device=degenerate_bounding_boxes.device,
            ),
        )
        assert transforms.SanitizeBoundingBoxes()(sample)["boxes"].shape == (0, 4)


def check_transform(transform, input, check_sample_input=True, batch=False):
    pickle.loads(pickle.dumps(transform))

    output = transform(input)
    assert isinstance(output, type(input))

    if isinstance(input, (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes)) and not isinstance(
        transform, transforms.ConvertBoundingBoxFormat
    ):
        assert output.format == input.format

    if check_sample_input:
        _check_transform_sample_input_smoke(
            transform,
            input,
            adapter=check_sample_input if callable(check_sample_input) else None,
            batch=batch,
        )

    return output


def _check_transform_batch_sample_input_smoke(transform, input, *, adapter, batch_size):
    # This is a bunch of input / output convention checks, using a big sample with different parts as input.
    if not check_type(input, (is_pure_tensor, ta_tensors.BatchImages, ta_tensors.BatchVideos)):
        return

    sample = _make_transform_batch_sample(
        # adapter might change transform inplace
        transform=transform if adapter is None else deepcopy(transform),
        image_or_video=input,
        adapter=adapter,
        batch_size=batch_size,
    )

    p = transform.p

    for container_type in [dict, list, tuple]:
        if container_type is dict:
            input = sample
        else:
            input = container_type(sample.values())

        input_flat, input_spec = tree_flatten(input)
        cloned_input = deepcopy(input)
        cloned_input_flat, cloned_input_spec = tree_flatten(cloned_input)

        with freeze_rng_state():
            torch.manual_seed(0)
            output = transform(cloned_input)
        output_flat, output_spec = tree_flatten(output)

        assert output_spec == input_spec

        if p == 0:
            output == cloned_input
        else:
            with freeze_rng_state():
                torch.manual_seed(0)
                needs_transform_list = transform._needs_transform_list(input_flat)
                batch_size = transform._get_input_batch_size(input_flat[0])
                device = input_flat[0].device
                if p == 1:
                    indices_transform = torch.tensor(list(range(batch_size)), dtype=torch.long, device=device)
                else:
                    indices_transform = transform._get_indices_transform(
                        batch_size,
                        device,
                    )
                if indices_transform.shape[0] == 0:
                    assert_equal(output, input)
                for inpt, cloned_inpt, opt, need_transform, name in zip(
                    input_flat,
                    cloned_input_flat,
                    output_flat,
                    needs_transform_list,
                    sample.keys(),
                ):
                    if isinstance(inpt, (_BatchConcatenatedTATensor)):
                        continue
                    if not need_transform:
                        if isinstance(opt, torch.Tensor):
                            if inpt.isnan().any():
                                continue
                            assert_equal(opt, inpt)
                        elif isinstance(opt, np.ndarray):
                            assert_equal(opt, inpt)
                        else:
                            assert opt == cloned_inpt
                    else:
                        assert type(opt) == type(cloned_inpt)

    # Enforce that the transform does not turn a degenerate bounding box, e.g. marked by RandomIoUCrop (or any other
    # future transform that does this), back into a valid one.
    for degenerate_bounding_boxes in (
        bounding_box
        for name, bounding_box in sample.items()
        if "degenerate" in name and isinstance(bounding_box, (ta_tensors.BatchBoundingBoxes))
    ):
        sample = dict(
            boxes=degenerate_bounding_boxes,
            labels=torch.randint(
                10,
                (degenerate_bounding_boxes.shape[0],),
                device=degenerate_bounding_boxes.device,
            ),
        )
        assert transforms.SanitizeBoundingBoxes()(sample)["boxes"].shape == (0, 4)


def check_batch_transform(transform, input, batch_size, check_sample_input=True):
    pickle.loads(pickle.dumps(transform))

    cloned_input = deepcopy(input)

    output = transform(cloned_input)
    assert isinstance(output, type(cloned_input))

    if isinstance(input, (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes)) and not isinstance(
        transform, transforms.ConvertBoundingBoxFormat
    ):
        assert output.format == input.format

    if check_sample_input:
        _check_transform_batch_sample_input_smoke(
            transform,
            cloned_input,
            adapter=check_sample_input if callable(check_sample_input) else None,
            batch_size=batch_size,
        )

    return output


def transform_cls_to_functional(transform_cls, **transform_specific_kwargs):
    def wrapper(input, *args, **kwargs):
        transform = transform_cls(*args, **transform_specific_kwargs, **kwargs)
        return transform(input)

    wrapper.__name__ = transform_cls.__name__

    return wrapper


def param_value_parametrization(**kwargs):
    """Helper function to turn

    @pytest.mark.parametrize(
        ("param", "value"),
        ("a", 1),
        ("a", 2),
        ("a", 3),
        ("b", -1.0)
        ("b", 1.0)
    )

    into

    @param_value_parametrization(a=[1, 2, 3], b=[-1.0, 1.0])
    """
    return pytest.mark.parametrize(
        ("param", "value"),
        [(param, value) for param, values in kwargs.items() for value in values],
    )


def adapt_fill(value, *, dtype):
    """Adapt fill values in the range [0.0, 1.0] to the value range of the dtype"""
    if value is None:
        return value

    max_value = get_max_value(dtype)
    value_type = float if dtype.is_floating_point else int

    if isinstance(value, (int, float)):
        return value_type(value * max_value)
    elif isinstance(value, (list, tuple)):
        return type(value)(value_type(v * max_value) for v in value)
    else:
        raise ValueError(f"fill should be an int or float, or a list or tuple of the former, but got '{value}'.")
