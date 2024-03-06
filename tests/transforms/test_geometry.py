import functools
import math

import numpy as np
import pytest
import torch
import torchaug.transforms as transforms
import torchaug.transforms.functional as F
import torchvision.transforms.v2 as tv_transforms
import torchvision.transforms.v2.functional as TVF
from torch import nn
from torchaug import ta_tensors
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import _get_perspective_coeffs

from ..utils import (
    adapt_fill,
    ALL_IMAGES_MAKERS,
    assert_equal,
    BATCH_IMAGES_TENSOR_AND_MAKERS,
    BOUNDING_BOXES_MAKERS,
    check_batch_transform,
    check_functional,
    check_functional_kernel_signature_match,
    check_kernel,
    check_transform,
    CORRECTNESS_FILLS,
    cpu_and_cuda,
    EXHAUSTIVE_TYPE_FILLS,
    freeze_rng_state,
    IMAGE_MAKERS,
    IMAGE_TENSOR_AND_MAKERS,
    make_batch_bounding_boxes,
    make_batch_detection_masks,
    make_batch_images,
    make_batch_images_tensor,
    make_batch_segmentation_masks,
    make_batch_videos,
    make_bounding_boxes,
    make_detection_masks,
    make_image,
    make_image_tensor,
    make_segmentation_mask,
    make_video,
    MASKS_MAKERS,
    param_value_parametrization,
    transform_cls_to_functional,
    VIDEO_MAKERS,
)


# We cannot use `list(transforms.InterpolationMode)` here, since it includes some PIL-only ones as well
INTERPOLATION_MODES = [
    transforms.InterpolationMode.NEAREST,
    transforms.InterpolationMode.NEAREST_EXACT,
    transforms.InterpolationMode.BILINEAR,
    transforms.InterpolationMode.BICUBIC,
]


def reference_affine_bounding_boxes_helper(
    bounding_boxes, *, affine_matrix, new_canvas_size=None, clamp=True, is_batch=False
):
    format = bounding_boxes.format
    canvas_size = new_canvas_size or bounding_boxes.canvas_size
    if is_batch:
        idx_sample = bounding_boxes.idx_sample

    def affine_bounding_boxes(bounding_boxes):
        dtype = bounding_boxes.dtype
        device = bounding_boxes.device

        # Go to float before converting to prevent precision loss in case of CXCYWH -> XYXY and W or H is 1
        input_xyxy = F.convert_bounding_box_format(
            bounding_boxes.to(dtype=torch.float64, device="cpu", copy=True),
            old_format=format,
            new_format=ta_tensors.BoundingBoxFormat.XYXY,
            inplace=True,
        )
        x1, y1, x2, y2 = input_xyxy.squeeze(0).tolist()

        points = np.array(
            [
                [x1, y1, 1.0],
                [x2, y1, 1.0],
                [x1, y2, 1.0],
                [x2, y2, 1.0],
            ]
        )
        transformed_points = np.matmul(points, affine_matrix.astype(points.dtype).T)

        output_xyxy = torch.Tensor(
            [
                float(np.min(transformed_points[:, 0])),
                float(np.min(transformed_points[:, 1])),
                float(np.max(transformed_points[:, 0])),
                float(np.max(transformed_points[:, 1])),
            ]
        )

        output = F.convert_bounding_box_format(
            output_xyxy, old_format=ta_tensors.BoundingBoxFormat.XYXY, new_format=format
        )

        if clamp:
            # It is important to clamp before casting, especially for CXCYWH format, dtype=int64
            output = F.clamp_bounding_boxes(
                output,
                format=format,
                canvas_size=canvas_size,
            )
        else:
            # We leave the bounding box as float64 so the caller gets the full precision to perform any additional
            # operation
            dtype = output.dtype

        return output.to(dtype=dtype, device=device)

    if is_batch:
        return ta_tensors.BatchBoundingBoxes(
            torch.cat(
                [
                    affine_bounding_boxes(b)
                    for b in bounding_boxes.reshape(-1, 4).unbind()
                ],
                dim=0,
            ).reshape(bounding_boxes.shape),
            format=format,
            canvas_size=canvas_size,
            idx_sample=idx_sample,
        )
    else:
        return ta_tensors.BoundingBoxes(
            torch.cat(
                [
                    affine_bounding_boxes(b)
                    for b in bounding_boxes.reshape(-1, 4).unbind()
                ],
                dim=0,
            ).reshape(bounding_boxes.shape),
            format=format,
            canvas_size=canvas_size,
        )


class TestResize:
    INPUT_SIZE = (17, 11)
    OUTPUT_SIZES = [17, [17], (17,), [12, 13], (12, 13)]

    def _make_max_size_kwarg(self, *, use_max_size, size):
        if use_max_size:
            if not (isinstance(size, int) or len(size) == 1):
                # This would result in an `ValueError`
                return None

            max_size = (size if isinstance(size, int) else size[0]) + 1
        else:
            max_size = None

        return dict(max_size=max_size)

    def _compute_output_size(self, *, input_size, size, max_size):
        if not (isinstance(size, int) or len(size) == 1):
            return tuple(size)

        if not isinstance(size, int):
            size = size[0]

        old_height, old_width = input_size
        ratio = old_width / old_height
        if ratio > 1:
            new_height = size
            new_width = int(ratio * new_height)
        else:
            new_width = size
            new_height = int(new_width / ratio)

        if max_size is not None and max(new_height, new_width) > max_size:
            # Need to recompute the aspect ratio, since it might have changed due to rounding
            ratio = new_width / new_height
            if ratio > 1:
                new_width = max_size
                new_height = int(new_width / ratio)
            else:
                new_height = max_size
                new_width = int(new_height * ratio)

        return new_height, new_width

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("interpolation", INTERPOLATION_MODES)
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("antialias", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(
        self, size, interpolation, use_max_size, antialias, dtype, device
    ):
        if not (
            max_size_kwarg := self._make_max_size_kwarg(
                use_max_size=use_max_size, size=size
            )
        ):
            return

        # In contrast to CPU, there is no native `InterpolationMode.BICUBIC` implementation for uint8 images on CUDA.
        # Internally, it uses the float path. Thus, we need to test with an enormous tolerance here to account for that.
        atol = (
            30
            if (
                interpolation is transforms.InterpolationMode.BICUBIC
                and dtype is torch.uint8
            )
            else 1
        )
        check_cuda_vs_cpu_tolerances = dict(
            rtol=0, atol=atol / 255 if dtype.is_floating_point else atol
        )

        check_kernel(
            F.resize_image,
            make_image(self.INPUT_SIZE, dtype=dtype, device=device),
            size=size,
            interpolation=interpolation,
            **max_size_kwarg,
            antialias=antialias,
            check_cuda_vs_cpu=check_cuda_vs_cpu_tolerances,
            check_scripted_vs_eager=not isinstance(size, int),
        )

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("interpolation", INTERPOLATION_MODES)
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("antialias", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_batch_images(
        self, size, interpolation, use_max_size, antialias, dtype, device
    ):
        if not (
            max_size_kwarg := self._make_max_size_kwarg(
                use_max_size=use_max_size, size=size
            )
        ):
            return

        # In contrast to CPU, there is no native `InterpolationMode.BICUBIC` implementation for uint8 images on CUDA.
        # Internally, it uses the float path. Thus, we need to test with an enormous tolerance here to account for that.
        atol = (
            30
            if (
                interpolation is transforms.InterpolationMode.BICUBIC
                and dtype is torch.uint8
            )
            else 1
        )
        check_cuda_vs_cpu_tolerances = dict(
            rtol=0, atol=atol / 255 if dtype.is_floating_point else atol
        )

        check_kernel(
            F.resize_image,
            make_batch_images(self.INPUT_SIZE, dtype=dtype, device=device),
            size=size,
            interpolation=interpolation,
            **max_size_kwarg,
            antialias=antialias,
            check_cuda_vs_cpu=check_cuda_vs_cpu_tolerances,
            check_scripted_vs_eager=not isinstance(size, int),
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_kernel_bounding_boxes(
        self, format, size, use_max_size, dtype, device, make_input
    ):
        if not (
            max_size_kwarg := self._make_max_size_kwarg(
                use_max_size=use_max_size, size=size
            )
        ):
            return

        bounding_boxes = make_input(
            format=format,
            canvas_size=self.INPUT_SIZE,
            dtype=dtype,
            device=device,
        )
        check_kernel(
            F.resize_bounding_boxes,
            bounding_boxes,
            canvas_size=bounding_boxes.canvas_size,
            size=size,
            **max_size_kwarg,
            check_scripted_vs_eager=not isinstance(size, int),
        )

    @pytest.mark.parametrize("make_mask", MASKS_MAKERS)
    def test_kernel_mask(self, make_mask):
        check_kernel(
            F.resize_mask, make_mask(self.INPUT_SIZE), size=self.OUTPUT_SIZES[-1]
        )

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(
            F.resize_video,
            make_video(self.INPUT_SIZE),
            size=self.OUTPUT_SIZES[-1],
            antialias=True,
        )

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    def test_functional(self, size, make_input):
        check_functional(
            F.resize,
            make_input(self.INPUT_SIZE),
            size=size,
            antialias=True,
            check_scripted_smoke=not isinstance(size, int),
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.resize_image, torch.Tensor),
            (F.resize_image, ta_tensors.Image),
            (F.resize_image, ta_tensors.BatchImages),
            (F.resize_bounding_boxes, ta_tensors.BoundingBoxes),
            (F.resize_bounding_boxes, ta_tensors.BatchBoundingBoxes),
            (F.resize_mask, ta_tensors.Mask),
            (F.resize_mask, ta_tensors.BatchMasks),
            (F.resize_video, ta_tensors.BatchVideos),
            (F.resize_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.resize, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize(
        "make_input",
        IMAGE_TENSOR_AND_MAKERS,
    )
    def test_transform(self, size, device, make_input):
        check_transform(
            transforms.Resize(size=size, antialias=True),
            make_input(self.INPUT_SIZE, device=device),
            batch=False,
        )

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", BATCH_IMAGES_TENSOR_AND_MAKERS)
    def test_transform_batch(self, size, device, make_input):
        check_transform(
            transforms.Resize(size=size, antialias=True),
            make_input(self.INPUT_SIZE, device=device),
            batch=True,
        )

    def _check_output_size(self, input, output, *, size, max_size):
        assert tuple(F.get_size(output)) == self._compute_output_size(
            input_size=F.get_size(input), size=size, max_size=max_size
        )

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    # `InterpolationMode.NEAREST` is modeled after the buggy `INTER_NEAREST` interpolation of CV2.
    # The PIL equivalent of `InterpolationMode.NEAREST` is `InterpolationMode.NEAREST_EXACT`
    @pytest.mark.parametrize(
        "interpolation",
        set(INTERPOLATION_MODES) - {transforms.InterpolationMode.NEAREST},
    )
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize(
        "fn", [F.resize, transform_cls_to_functional(transforms.Resize)]
    )
    def test_image_correctness(self, size, interpolation, use_max_size, fn):
        if not (
            max_size_kwarg := self._make_max_size_kwarg(
                use_max_size=use_max_size, size=size
            )
        ):
            return

        image = make_image(self.INPUT_SIZE, dtype=torch.uint8)

        actual = fn(
            image,
            size=size,
            interpolation=interpolation,
            **max_size_kwarg,
            antialias=True,
        )
        expected = TVF.resize(
            torch.as_tensor(image),
            size=size,
            interpolation=interpolation,
            **max_size_kwarg,
            antialias=True,
        )

        self._check_output_size(image, actual, size=size, **max_size_kwarg)
        torch.testing.assert_close(actual, expected, atol=1, rtol=0)

    def _reference_resize_bounding_boxes(
        self, bounding_boxes, *, size, max_size=None, is_batch=False
    ):
        old_height, old_width = bounding_boxes.canvas_size
        new_height, new_width = self._compute_output_size(
            input_size=bounding_boxes.canvas_size, size=size, max_size=max_size
        )

        if (old_height, old_width) == (new_height, new_width):
            return bounding_boxes

        affine_matrix = np.array(
            [
                [new_width / old_width, 0, 0],
                [0, new_height / old_height, 0],
            ],
        )

        return reference_affine_bounding_boxes_helper(
            bounding_boxes,
            affine_matrix=affine_matrix,
            new_canvas_size=(new_height, new_width),
            is_batch=is_batch,
        )

    def _reference_resize_batch_bounding_boxes(
        self, bounding_boxes, *, size, max_size=None, is_batch=False
    ):
        old_height, old_width = bounding_boxes.canvas_size
        new_height, new_width = self._compute_output_size(
            input_size=bounding_boxes.canvas_size, size=size, max_size=max_size
        )

        if (old_height, old_width) == (new_height, new_width):
            return bounding_boxes

        affine_matrix = np.array(
            [
                [new_width / old_width, 0, 0],
                [0, new_height / old_height, 0],
            ],
        )

        return reference_affine_bounding_boxes_helper(
            bounding_boxes,
            affine_matrix=affine_matrix,
            new_canvas_size=(new_height, new_width),
            is_batch=is_batch,
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize("use_max_size", [True, False])
    @pytest.mark.parametrize(
        "fn", [F.resize, transform_cls_to_functional(transforms.Resize)]
    )
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_bounding_boxes_correctness(
        self, format, size, use_max_size, fn, make_input
    ):
        if not (
            max_size_kwarg := self._make_max_size_kwarg(
                use_max_size=use_max_size, size=size
            )
        ):
            return

        is_batch = make_input == make_batch_bounding_boxes
        bounding_boxes = make_input(format=format, canvas_size=self.INPUT_SIZE)

        actual = fn(bounding_boxes, size=size, **max_size_kwarg)
        expected = self._reference_resize_bounding_boxes(
            bounding_boxes, size=size, **max_size_kwarg, is_batch=is_batch
        )

        self._check_output_size(bounding_boxes, actual, size=size, **max_size_kwarg)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("size", OUTPUT_SIZES)
    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    def test_max_size_error(self, size, make_input):
        if isinstance(size, int) or len(size) == 1:
            max_size = (size if isinstance(size, int) else size[0]) - 1
            match = "must be strictly greater than the requested size"
        else:
            # value can be anything other than None
            max_size = -1
            match = "size should be an int or a sequence of length 1"

        with pytest.raises(ValueError, match=match):
            F.resize(
                make_input(self.INPUT_SIZE),
                size=size,
                max_size=max_size,
                antialias=True,
            )

    def test_transform_unknown_size_error(self):
        with pytest.raises(
            ValueError,
            match="size can either be an integer or a sequence of one or two integers",
        ):
            transforms.Resize(size=object())

    @pytest.mark.parametrize(
        "size",
        [
            min(INPUT_SIZE),
            [min(INPUT_SIZE)],
            (min(INPUT_SIZE),),
            list(INPUT_SIZE),
            tuple(INPUT_SIZE),
        ],
    )
    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    def test_noop(self, size, make_input):
        input = make_input(self.INPUT_SIZE)

        output = F.resize(input, size=F.get_size(input), antialias=True)

        # This identity check is not a requirement. It is here to avoid breaking the behavior by accident. If there
        # is a good reason to break this, feel free to downgrade to an equality check.
        if isinstance(input, ta_tensors.TATensor):
            # We can't test identity directly, since that checks for the identity of the Python object. Since all
            # ta_tensors unwrap before a kernel and wrap again afterwards, the Python object changes. Thus, we check
            # that the underlying storage is the same
            assert output.data_ptr() == input.data_ptr()
        else:
            assert output is input

    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    def test_no_regression_5405(self, make_input):
        # Checks that `max_size` is not ignored if `size == small_edge_size`
        # See https://github.com/pytorch/vision/issues/5405

        input = make_input(self.INPUT_SIZE)

        size = min(F.get_size(input))
        max_size = size + 1
        output = F.resize(input, size=size, max_size=max_size, antialias=True)

        assert max(F.get_size(output)) == max_size

    def _make_image(
        self, *args, batch_dims=(), memory_format=torch.contiguous_format, **kwargs
    ):
        # torch.channels_last memory_format is only available for 4D tensors, i.e. (B, C, H, W). However, images coming
        # from PIL or our own I/O functions do not have a batch dimensions and are thus 3D, i.e. (C, H, W). Still, the
        # layout of the data in memory is channels last. To emulate this when a 3D input is requested here, we create
        # the image as 4D and create a view with the right shape afterwards. With this the layout in memory is channels
        # last although PyTorch doesn't recognizes it as such.
        emulate_channels_last = (
            memory_format is torch.channels_last and len(batch_dims) != 1
        )

        image = make_image(
            *args,
            batch_dims=(math.prod(batch_dims),)
            if emulate_channels_last
            else batch_dims,
            memory_format=memory_format,
            **kwargs,
        )

        if emulate_channels_last:
            image = ta_tensors.wrap(
                image.view(*batch_dims, *image.shape[-3:]), like=image
            )

        return image

    def _check_stride(self, image, *, memory_format):
        C, H, W = F.get_dimensions(image)
        if memory_format is torch.contiguous_format:
            expected_stride = (H * W, W, 1)
        elif memory_format is torch.channels_last:
            expected_stride = (1, W * C, C)
        else:
            raise ValueError(f"Unknown memory_format: {memory_format}")

        assert image.stride() == expected_stride

    # TODO: We can remove this test and related torchvision workaround
    #  once we fixed related pytorch issue: https://github.com/pytorch/pytorch/issues/68430
    @pytest.mark.parametrize("interpolation", INTERPOLATION_MODES)
    @pytest.mark.parametrize("antialias", [True, False])
    @pytest.mark.parametrize(
        "memory_format", [torch.contiguous_format, torch.channels_last]
    )
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image_memory_format_consistency(
        self, interpolation, antialias, memory_format, dtype, device
    ):
        size = self.OUTPUT_SIZES[0]

        input = self._make_image(
            self.INPUT_SIZE, dtype=dtype, device=device, memory_format=memory_format
        )

        # Smoke test to make sure we aren't starting with wrong assumptions
        self._check_stride(input, memory_format=memory_format)

        output = F.resize_image(
            input, size=size, interpolation=interpolation, antialias=antialias
        )

        self._check_stride(output, memory_format=memory_format)

    def test_float16_no_rounding(self):
        # Make sure Resize() doesn't round float16 images
        # Non-regression test for https://github.com/pytorch/vision/issues/7667

        input = make_image_tensor(self.INPUT_SIZE, dtype=torch.float16)
        output = F.resize_image(input, size=self.OUTPUT_SIZES[0], antialias=True)

        assert output.dtype is torch.float16
        assert (output.round() - output).abs().sum() > 0


class TestHorizontalFlip:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_input):
        check_kernel(
            F.horizontal_flip_image,
            make_input(dtype=dtype, device=device),
            check_batch_kernel_leading_dims=False,
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize(
        "make_input",
        BOUNDING_BOXES_MAKERS,
    )
    def test_kernel_bounding_boxes(self, format, dtype, device, make_input):
        bounding_boxes = make_input(format=format, dtype=dtype, device=device)
        check_kernel(
            F.horizontal_flip_bounding_boxes,
            bounding_boxes,
            format=format,
            canvas_size=bounding_boxes.canvas_size,
        )

    @pytest.mark.parametrize(
        "make_mask",
        MASKS_MAKERS,
    )
    def test_kernel_mask(self, make_mask):
        check_kernel(F.horizontal_flip_mask, make_mask())

    @pytest.mark.parametrize("make_input", VIDEO_MAKERS)
    def test_kernel_video(self, make_input):
        check_kernel(F.horizontal_flip_video, make_input())

    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    def test_functional(self, make_input):
        check_functional(F.horizontal_flip, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.horizontal_flip_image, torch.Tensor),
            (F.horizontal_flip_image, ta_tensors.Image),
            (F.horizontal_flip_bounding_boxes, ta_tensors.BoundingBoxes),
            (F.horizontal_flip_mask, ta_tensors.Mask),
            (F.horizontal_flip_video, ta_tensors.Video),
            (F.horizontal_flip_image, ta_tensors.BatchImages),
            (F.horizontal_flip_video, ta_tensors.BatchVideos),
            (F.horizontal_flip_bounding_boxes, ta_tensors.BatchBoundingBoxes),
            (F.horizontal_flip_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.horizontal_flip, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        check_transform(transforms.RandomHorizontalFlip(p=1), make_input(device=device))

    @pytest.mark.parametrize(
        "fn",
        [
            F.horizontal_flip,
            transform_cls_to_functional(transforms.RandomHorizontalFlip, p=1),
        ],
    )
    def test_image_correctness(self, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = TVF.hflip(torch.as_tensor(image))

        torch.testing.assert_close(actual, expected)

    def _reference_horizontal_flip_bounding_boxes(self, bounding_boxes, is_batch=False):
        affine_matrix = np.array(
            [
                [-1, 0, bounding_boxes.canvas_size[1]],
                [0, 1, 0],
            ],
        )

        return reference_affine_bounding_boxes_helper(
            bounding_boxes, affine_matrix=affine_matrix, is_batch=is_batch
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize(
        "fn",
        [
            F.horizontal_flip,
            transform_cls_to_functional(transforms.RandomHorizontalFlip, p=1),
        ],
    )
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_bounding_boxes_correctness(self, format, fn, make_input):
        is_batch = make_input == make_batch_bounding_boxes
        bounding_boxes = make_input(format=format)

        actual = fn(bounding_boxes)
        expected = self._reference_horizontal_flip_bounding_boxes(
            bounding_boxes, is_batch=is_batch
        )

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform_noop(self, make_input, device):
        input = make_input(device=device)

        transform = transforms.RandomHorizontalFlip(p=0)

        output = transform(input)

        assert_equal(output, input)

    @pytest.mark.parametrize("make_input", BATCH_IMAGES_TENSOR_AND_MAKERS)
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("p", [0, 1, 0.5])
    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("batch_inplace", [True, False])
    def test_batch_transform(self, make_input, device, p, batch_size, batch_inplace):
        check_batch_transform(
            transforms.RandomHorizontalFlip(
                p=p, batch_inplace=batch_inplace, batch_transform=True
            ),
            make_input(batch_dims=(batch_size,), device=device),
            batch_size=batch_size,
        )


class TestAffine:
    _EXHAUSTIVE_TYPE_AFFINE_KWARGS = dict(
        # float, int
        angle=[-10.9, 18],
        # two-list of float, two-list of int, two-tuple of float, two-tuple of int
        translate=[[6.3, -0.6], [1, -3], (16.6, -6.6), (-2, 4)],
        # float
        scale=[0.5],
        # float, int,
        # one-list of float, one-list of int, one-tuple of float, one-tuple of int
        # two-list of float, two-list of int, two-tuple of float, two-tuple of int
        shear=[
            35.6,
            38,
            [-37.7],
            [-23],
            (5.3,),
            (-52,),
            [5.4, 21.8],
            [-47, 51],
            (-11.2, 36.7),
            (8, -53),
        ],
        # None
        # two-list of float, two-list of int, two-tuple of float, two-tuple of int
        center=[None, [1.2, 4.9], [-3, 1], (2.5, -4.7), (3, 2)],
    )
    # The special case for shear makes sure we pick a value that is supported while JIT scripting
    _MINIMAL_AFFINE_KWARGS = {
        k: vs[0] if k != "shear" else next(v for v in vs if isinstance(v, list))
        for k, vs in _EXHAUSTIVE_TYPE_AFFINE_KWARGS.items()
    }
    _CORRECTNESS_AFFINE_KWARGS = {
        k: [
            v
            for v in vs
            if v is None or isinstance(v, float) or (isinstance(v, list) and len(v) > 1)
        ]
        for k, vs in _EXHAUSTIVE_TYPE_AFFINE_KWARGS.items()
    }

    _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES = dict(
        degrees=[30, (-15, 20)],
        translate=[None, (0.5, 0.5)],
        scale=[None, (0.75, 1.25)],
        shear=[None, (12, 30, -17, 5), 10, (-5, 12)],
    )
    _CORRECTNESS_TRANSFORM_AFFINE_RANGES = {
        k: next(v for v in vs if v is not None)
        for k, vs in _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES.items()
    }

    def _check_kernel(self, kernel, input, *args, **kwargs):
        kwargs_ = self._MINIMAL_AFFINE_KWARGS.copy()
        kwargs_.update(kwargs)
        check_kernel(kernel, input, *args, **kwargs_)

    @param_value_parametrization(
        angle=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["angle"],
        translate=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["translate"],
        shear=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["shear"],
        center=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["center"],
        interpolation=[
            transforms.InterpolationMode.NEAREST,
            transforms.InterpolationMode.BILINEAR,
        ],
        fill=EXHAUSTIVE_TYPE_FILLS,
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, param, value, dtype, device, make_input):
        if param == "fill":
            value = adapt_fill(value, dtype=dtype)
        self._check_kernel(
            F.affine_image,
            make_input(dtype=dtype, device=device),
            **{param: value},
            check_scripted_vs_eager=not (
                param in {"shear", "fill"} and isinstance(value, (int, float))
            ),
            check_cuda_vs_cpu=dict(atol=1, rtol=0)
            if dtype is torch.uint8
            and param == "interpolation"
            and value is transforms.InterpolationMode.BILINEAR
            else True,
        )

    @param_value_parametrization(
        angle=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["angle"],
        translate=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["translate"],
        shear=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["shear"],
        center=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["center"],
    )
    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_kernel_bounding_boxes(
        self, param, value, format, dtype, device, make_input
    ):
        bounding_boxes = make_input(format=format, dtype=dtype, device=device)
        self._check_kernel(
            F.affine_bounding_boxes,
            bounding_boxes,
            format=format,
            canvas_size=bounding_boxes.canvas_size,
            **{param: value},
            check_scripted_vs_eager=not (
                param == "shear" and isinstance(value, (int, float))
            ),
        )

    @pytest.mark.parametrize("make_mask", MASKS_MAKERS)
    def test_kernel_mask(self, make_mask):
        self._check_kernel(F.affine_mask, make_mask())

    @pytest.mark.parametrize("make_input", VIDEO_MAKERS)
    def test_kernel_video(self, make_input):
        self._check_kernel(F.affine_video, make_input())

    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    def test_functional(self, make_input):
        check_functional(F.affine, make_input(), **self._MINIMAL_AFFINE_KWARGS)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.affine_image, torch.Tensor),
            (F.affine_image, ta_tensors.Image),
            (F.affine_image, ta_tensors.BatchImages),
            (F.affine_bounding_boxes, ta_tensors.BoundingBoxes),
            (F.affine_bounding_boxes, ta_tensors.BatchBoundingBoxes),
            (F.affine_mask, ta_tensors.Mask),
            (F.affine_mask, ta_tensors.BatchMasks),
            (F.affine_video, ta_tensors.Video),
            (F.affine_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.affine, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        input = make_input(device=device)

        check_transform(
            transforms.RandomAffine(**self._CORRECTNESS_TRANSFORM_AFFINE_RANGES), input
        )

    @pytest.mark.parametrize(
        "make_input",
        BATCH_IMAGES_TENSOR_AND_MAKERS,
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("batch_inplace", [True, False])
    @pytest.mark.parametrize("num_chunks", [-1, 1, 2])
    @pytest.mark.parametrize("permute_chunks", [True, False])
    def test_batch_transform(
        self, make_input, device, batch_size, batch_inplace, num_chunks, permute_chunks
    ):
        input = make_input(device=device, batch_dims=(batch_size,))

        check_batch_transform(
            transforms.RandomAffine(
                **self._CORRECTNESS_TRANSFORM_AFFINE_RANGES,
                batch_transform=True,
                batch_inplace=batch_inplace,
                permute_chunks=permute_chunks,
                num_chunks=num_chunks,
            ),
            input,
            batch_size=batch_size,
        )

    @pytest.mark.parametrize("angle", _CORRECTNESS_AFFINE_KWARGS["angle"])
    @pytest.mark.parametrize("translate", _CORRECTNESS_AFFINE_KWARGS["translate"])
    @pytest.mark.parametrize("scale", _CORRECTNESS_AFFINE_KWARGS["scale"])
    @pytest.mark.parametrize("shear", _CORRECTNESS_AFFINE_KWARGS["shear"])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize(
        "interpolation",
        [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR],
    )
    @pytest.mark.parametrize("fill", CORRECTNESS_FILLS)
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_functional_image_correctness(
        self, angle, translate, scale, shear, center, interpolation, fill, make_input
    ):
        image = make_input(dtype=torch.uint8, device="cpu")

        fill = adapt_fill(fill, dtype=torch.uint8)

        actual = F.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
            interpolation=interpolation,
            fill=fill,
        )
        expected = TVF.affine(
            torch.as_tensor(image),
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
            interpolation=interpolation,
            fill=fill,
        )

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 2 if interpolation is transforms.InterpolationMode.NEAREST else 8

    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize(
        "interpolation",
        [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR],
    )
    @pytest.mark.parametrize("fill", CORRECTNESS_FILLS)
    @pytest.mark.parametrize("seed", list(range(5)))
    @pytest.mark.parametrize(
        "make_input",
        IMAGE_MAKERS,
    )
    def test_transform_image_correctness(
        self, center, interpolation, fill, seed, make_input
    ):
        image = make_input(dtype=torch.uint8, device="cpu")

        fill = adapt_fill(fill, dtype=torch.uint8)

        transform = transforms.RandomAffine(
            **self._CORRECTNESS_TRANSFORM_AFFINE_RANGES,
            center=center,
            interpolation=interpolation,
            fill=fill,
        )

        torchvision_transform = tv_transforms.RandomAffine(
            **self._CORRECTNESS_TRANSFORM_AFFINE_RANGES,
            center=center,
            interpolation=interpolation,
            fill=fill,
        )

        torch.manual_seed(seed)
        actual = transform(image)

        torch.manual_seed(seed)
        expected = torchvision_transform(torch.as_tensor(image))

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 2 if interpolation is transforms.InterpolationMode.NEAREST else 8

    def _compute_affine_matrix(self, *, angle, translate, scale, shear, center):
        rot = math.radians(angle)
        cx, cy = center
        tx, ty = translate
        sx, sy = [
            math.radians(s)
            for s in ([shear, 0.0] if isinstance(shear, (int, float)) else shear)
        ]

        c_matrix = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        t_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        c_matrix_inv = np.linalg.inv(c_matrix)
        rs_matrix = np.array(
            [
                [scale * math.cos(rot), -scale * math.sin(rot), 0],
                [scale * math.sin(rot), scale * math.cos(rot), 0],
                [0, 0, 1],
            ]
        )
        shear_x_matrix = np.array([[1, -math.tan(sx), 0], [0, 1, 0], [0, 0, 1]])
        shear_y_matrix = np.array([[1, 0, 0], [-math.tan(sy), 1, 0], [0, 0, 1]])
        rss_matrix = np.matmul(rs_matrix, np.matmul(shear_y_matrix, shear_x_matrix))
        true_matrix = np.matmul(
            t_matrix, np.matmul(c_matrix, np.matmul(rss_matrix, c_matrix_inv))
        )
        return true_matrix[:2, :]

    def _reference_affine_bounding_boxes(
        self, bounding_boxes, *, angle, translate, scale, shear, center, is_batch
    ):
        if center is None:
            center = [s * 0.5 for s in bounding_boxes.canvas_size[::-1]]

        return reference_affine_bounding_boxes_helper(
            bounding_boxes,
            affine_matrix=self._compute_affine_matrix(
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                center=center,
            ),
            is_batch=is_batch,
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("angle", _CORRECTNESS_AFFINE_KWARGS["angle"])
    @pytest.mark.parametrize("translate", _CORRECTNESS_AFFINE_KWARGS["translate"])
    @pytest.mark.parametrize("scale", _CORRECTNESS_AFFINE_KWARGS["scale"])
    @pytest.mark.parametrize("shear", _CORRECTNESS_AFFINE_KWARGS["shear"])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_functional_bounding_boxes_correctness(
        self, format, angle, translate, scale, shear, center, make_input
    ):
        is_batch = make_input == make_batch_bounding_boxes
        bounding_boxes = make_input(format=format)

        actual = F.affine(
            bounding_boxes,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
        )
        expected = self._reference_affine_bounding_boxes(
            bounding_boxes,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
            is_batch=is_batch,
        )

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize("seed", list(range(5)))
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_transform_bounding_boxes_correctness(
        self, format, center, seed, make_input
    ):
        is_batch = make_input == make_batch_bounding_boxes
        bounding_boxes = make_input(format=format)

        transform = transforms.RandomAffine(
            **self._CORRECTNESS_TRANSFORM_AFFINE_RANGES,
            center=center,
            batch_transform=is_batch,
        )

        torch.manual_seed(seed)
        params = transform._get_params([bounding_boxes], 1, torch.tensor([0]))[0]

        torch.manual_seed(seed)
        actual = transform(bounding_boxes)

        expected = self._reference_affine_bounding_boxes(
            bounding_boxes, **params, center=center, is_batch=is_batch
        )

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "degrees", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["degrees"]
    )
    @pytest.mark.parametrize(
        "translate", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["translate"]
    )
    @pytest.mark.parametrize("scale", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["scale"])
    @pytest.mark.parametrize("shear", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["shear"])
    @pytest.mark.parametrize("seed", list(range(10)))
    @pytest.mark.parametrize("num_chunks", [1, 2])
    def test_transform_get_params_bounds(
        self, degrees, translate, scale, shear, seed, num_chunks
    ):
        image = make_batch_images(batch_dims=(num_chunks,))
        height, width = F.get_size(image)

        transform = transforms.RandomAffine(
            degrees=degrees, translate=translate, scale=scale, shear=shear
        )

        torch.manual_seed(seed)
        chunks_params = transform._get_params(
            [image], num_chunks, torch.arange(image.shape[0])
        )
        for i in range(num_chunks):
            params = chunks_params[i]
            if isinstance(degrees, (int, float)):
                assert -degrees <= params["angle"] <= degrees
            else:
                assert degrees[0] <= params["angle"] <= degrees[1]

            if translate is not None:
                width_max = int(round(translate[0] * width))
                height_max = int(round(translate[1] * height))
                assert -width_max <= params["translate"][0] <= width_max
                assert -height_max <= params["translate"][1] <= height_max
            else:
                assert params["translate"] == (0, 0)

            if scale is not None:
                assert scale[0] <= params["scale"] <= scale[1]
            else:
                assert params["scale"] == 1.0

            if shear is not None:
                if isinstance(shear, (int, float)):
                    assert -shear <= params["shear"][0] <= shear
                    assert params["shear"][1] == 0.0
                elif len(shear) == 2:
                    assert shear[0] <= params["shear"][0] <= shear[1]
                    assert params["shear"][1] == 0.0
                elif len(shear) == 4:
                    assert shear[0] <= params["shear"][0] <= shear[1]
                    assert shear[2] <= params["shear"][1] <= shear[3]
            else:
                assert params["shear"] == (0, 0)

    @pytest.mark.parametrize(
        "param", ["degrees", "translate", "scale", "shear", "center"]
    )
    @pytest.mark.parametrize("value", [0, [0], [0, 0, 0]])
    def test_transform_sequence_len_errors(self, param, value):
        if param in {"degrees", "shear"} and not isinstance(value, list):
            return

        kwargs = {param: value}
        if param != "degrees":
            kwargs["degrees"] = 0

        with pytest.raises(
            ValueError if isinstance(value, list) else TypeError,
            match=f"{param} should be a sequence of length 2",
        ):
            transforms.RandomAffine(**kwargs)

    def test_transform_negative_degrees_error(self):
        with pytest.raises(
            ValueError, match="If degrees is a single number, it must be positive"
        ):
            transforms.RandomAffine(degrees=-1)

    @pytest.mark.parametrize("translate", [[-1, 0], [2, 0], [-1, 2]])
    def test_transform_translate_range_error(self, translate):
        with pytest.raises(
            ValueError, match="translation values should be between 0 and 1"
        ):
            transforms.RandomAffine(degrees=0, translate=translate)

    @pytest.mark.parametrize("scale", [[-1, 0], [0, -1], [-1, -1]])
    def test_transform_scale_range_error(self, scale):
        with pytest.raises(ValueError, match="scale values should be positive"):
            transforms.RandomAffine(degrees=0, scale=scale)

    def test_transform_negative_shear_error(self):
        with pytest.raises(
            ValueError, match="If shear is a single number, it must be positive"
        ):
            transforms.RandomAffine(degrees=0, shear=-1)

    def test_transform_unknown_fill_error(self):
        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.RandomAffine(degrees=0, fill="fill")


class TestVerticalFlip:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_input):
        check_kernel(F.vertical_flip_image, make_input(dtype=dtype, device=device))

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_kernel_bounding_boxes(self, format, dtype, device, make_input):
        bounding_boxes = make_input(format=format, dtype=dtype, device=device)
        check_kernel(
            F.vertical_flip_bounding_boxes,
            bounding_boxes,
            format=format,
            canvas_size=bounding_boxes.canvas_size,
        )

    @pytest.mark.parametrize("make_mask", MASKS_MAKERS)
    def test_kernel_mask(self, make_mask):
        check_kernel(F.vertical_flip_mask, make_mask())

    @pytest.mark.parametrize("make_input", VIDEO_MAKERS)
    def test_kernel_video(self, make_input):
        check_kernel(F.vertical_flip_video, make_input())

    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    def test_functional(self, make_input):
        check_functional(F.vertical_flip, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.vertical_flip_image, torch.Tensor),
            (F.vertical_flip_image, ta_tensors.Image),
            (F.vertical_flip_bounding_boxes, ta_tensors.BoundingBoxes),
            (F.vertical_flip_mask, ta_tensors.Mask),
            (F.vertical_flip_video, ta_tensors.Video),
            (F.vertical_flip_image, ta_tensors.BatchImages),
            (F.vertical_flip_bounding_boxes, ta_tensors.BatchBoundingBoxes),
            (F.vertical_flip_mask, ta_tensors.BatchMasks),
            (F.vertical_flip_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.vertical_flip, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        check_transform(transforms.RandomVerticalFlip(p=1), make_input(device=device))

    @pytest.mark.parametrize(
        "make_input",
        BATCH_IMAGES_TENSOR_AND_MAKERS,
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("p", [0, 1, 0.5])
    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("batch_inplace", [True, False])
    def test_batch_transform(self, make_input, device, p, batch_size, batch_inplace):
        check_batch_transform(
            transforms.RandomVerticalFlip(
                p=p, batch_inplace=batch_inplace, batch_transform=True
            ),
            make_input(device=device, batch_dims=(batch_size,)),
            batch_size=batch_size,
        )

    @pytest.mark.parametrize(
        "fn",
        [
            F.vertical_flip,
            transform_cls_to_functional(transforms.RandomVerticalFlip, p=1),
        ],
    )
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_image_correctness(self, fn, make_input):
        image = make_input(dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = TVF.vertical_flip(torch.as_tensor(image))

        torch.testing.assert_close(actual, expected)

    def _reference_vertical_flip_bounding_boxes(self, bounding_boxes, is_batch):
        affine_matrix = np.array(
            [
                [1, 0, 0],
                [0, -1, bounding_boxes.canvas_size[0]],
            ],
        )

        return reference_affine_bounding_boxes_helper(
            bounding_boxes, affine_matrix=affine_matrix, is_batch=is_batch
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize(
        "fn",
        [
            F.vertical_flip,
            transform_cls_to_functional(transforms.RandomVerticalFlip, p=1),
        ],
    )
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_bounding_boxes_correctness(self, format, fn, make_input):
        is_batch = make_input == make_batch_bounding_boxes
        bounding_boxes = make_input(format=format)

        actual = fn(bounding_boxes)
        expected = self._reference_vertical_flip_bounding_boxes(
            bounding_boxes, is_batch
        )

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform_noop(self, make_input, device):
        input = make_input(device=device)

        transform = transforms.RandomVerticalFlip(p=0)

        output = transform(input)

        assert_equal(output, input)


class TestRotate:
    _EXHAUSTIVE_TYPE_AFFINE_KWARGS = dict(
        # float, int
        angle=[-10.9, 18],
        # None
        # two-list of float, two-list of int, two-tuple of float, two-tuple of int
        center=[None, [1.2, 4.9], [-3, 1], (2.5, -4.7), (3, 2)],
    )
    _MINIMAL_AFFINE_KWARGS = {
        k: vs[0] for k, vs in _EXHAUSTIVE_TYPE_AFFINE_KWARGS.items()
    }
    _CORRECTNESS_AFFINE_KWARGS = {
        k: [v for v in vs if v is None or isinstance(v, float) or isinstance(v, list)]
        for k, vs in _EXHAUSTIVE_TYPE_AFFINE_KWARGS.items()
    }

    _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES = dict(
        degrees=[30, (-15, 20)],
    )
    _CORRECTNESS_TRANSFORM_AFFINE_RANGES = {
        k: vs[0] for k, vs in _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES.items()
    }

    @param_value_parametrization(
        angle=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["angle"],
        interpolation=[
            transforms.InterpolationMode.NEAREST,
            transforms.InterpolationMode.BILINEAR,
        ],
        expand=[False, True],
        center=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["center"],
        fill=EXHAUSTIVE_TYPE_FILLS,
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, param, value, dtype, device, make_input):
        kwargs = {param: value}
        if param != "angle":
            kwargs["angle"] = self._MINIMAL_AFFINE_KWARGS["angle"]
        check_kernel(
            F.rotate_image,
            make_input(dtype=dtype, device=device),
            **kwargs,
            check_scripted_vs_eager=not (
                param == "fill" and isinstance(value, (int, float))
            ),
        )

    @param_value_parametrization(
        angle=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["angle"],
        expand=[False, True],
        center=_EXHAUSTIVE_TYPE_AFFINE_KWARGS["center"],
    )
    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_kernel_bounding_boxes(
        self, param, value, format, dtype, device, make_input
    ):
        kwargs = {param: value}
        if param != "angle":
            kwargs["angle"] = self._MINIMAL_AFFINE_KWARGS["angle"]

        bounding_boxes = make_input(format=format, dtype=dtype, device=device)

        check_kernel(
            F.rotate_bounding_boxes,
            bounding_boxes,
            format=format,
            canvas_size=bounding_boxes.canvas_size,
            **kwargs,
        )

    @pytest.mark.parametrize("make_mask", MASKS_MAKERS)
    def test_kernel_mask(self, make_mask):
        check_kernel(F.rotate_mask, make_mask(), **self._MINIMAL_AFFINE_KWARGS)

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(F.rotate_video, make_video(), **self._MINIMAL_AFFINE_KWARGS)

    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    def test_functional(self, make_input):
        check_functional(F.rotate, make_input(), **self._MINIMAL_AFFINE_KWARGS)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.rotate_image, torch.Tensor),
            (F.rotate_image, ta_tensors.Image),
            (F.rotate_image, ta_tensors.BatchImages),
            (F.rotate_bounding_boxes, ta_tensors.BoundingBoxes),
            (F.rotate_bounding_boxes, ta_tensors.BatchBoundingBoxes),
            (F.rotate_mask, ta_tensors.Mask),
            (F.rotate_mask, ta_tensors.BatchMasks),
            (F.rotate_video, ta_tensors.BatchVideos),
            (F.rotate_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.rotate, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        check_transform(
            transforms.RandomRotation(**self._CORRECTNESS_TRANSFORM_AFFINE_RANGES),
            make_input(device=device),
        )

    @pytest.mark.parametrize(
        "make_input",
        BATCH_IMAGES_TENSOR_AND_MAKERS,
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("batch_inplace", [True, False])
    @pytest.mark.parametrize("num_chunks", [-1, 1, 2])
    @pytest.mark.parametrize("permute_chunks", [True, False])
    def test_batch_transform(
        self, make_input, device, batch_size, num_chunks, permute_chunks, batch_inplace
    ):
        check_batch_transform(
            transforms.RandomRotation(
                **self._CORRECTNESS_TRANSFORM_AFFINE_RANGES,
                batch_transform=True,
                batch_inplace=batch_inplace,
                permute_chunks=permute_chunks,
                num_chunks=num_chunks,
            ),
            make_input(device=device, batch_dims=(batch_size,)),
            batch_size=batch_size,
        )

    @pytest.mark.parametrize("angle", _CORRECTNESS_AFFINE_KWARGS["angle"])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize(
        "interpolation",
        [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR],
    )
    @pytest.mark.parametrize("expand", [False, True])
    @pytest.mark.parametrize("fill", CORRECTNESS_FILLS)
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_functional_image_correctness(
        self, angle, center, interpolation, expand, fill, make_input
    ):
        image = make_input(dtype=torch.uint8, device="cpu")

        fill = adapt_fill(fill, dtype=torch.uint8)

        actual = F.rotate(
            image,
            angle=angle,
            center=center,
            interpolation=interpolation,
            expand=expand,
            fill=fill,
        )
        expected = TVF.rotate(
            torch.as_tensor(image),
            angle=angle,
            center=center,
            interpolation=interpolation,
            expand=expand,
            fill=fill,
        )

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 1 if interpolation is transforms.InterpolationMode.NEAREST else 6

    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize(
        "interpolation",
        [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR],
    )
    @pytest.mark.parametrize("expand", [False, True])
    @pytest.mark.parametrize("fill", CORRECTNESS_FILLS)
    @pytest.mark.parametrize("seed", list(range(5)))
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_transform_image_correctness(
        self, center, interpolation, expand, fill, seed, make_input
    ):
        image = make_input(dtype=torch.uint8, device="cpu")

        fill = adapt_fill(fill, dtype=torch.uint8)

        transform = transforms.RandomRotation(
            **self._CORRECTNESS_TRANSFORM_AFFINE_RANGES,
            center=center,
            interpolation=interpolation,
            expand=expand,
            fill=fill,
        )

        torchvision_transform = tv_transforms.RandomRotation(
            **self._CORRECTNESS_TRANSFORM_AFFINE_RANGES,
            center=center,
            interpolation=interpolation,
            expand=expand,
            fill=fill,
        )

        torch.manual_seed(seed)
        actual = transform(image)

        torch.manual_seed(seed)
        expected = torchvision_transform(torch.as_tensor(image))

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 1 if interpolation is transforms.InterpolationMode.NEAREST else 6

    def _compute_output_canvas_size(self, *, expand, canvas_size, affine_matrix):
        if not expand:
            return canvas_size, (0.0, 0.0)

        input_height, input_width = canvas_size

        input_image_frame = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, input_height, 1.0],
                [input_width, input_height, 1.0],
                [input_width, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        output_image_frame = np.matmul(
            input_image_frame, affine_matrix.astype(input_image_frame.dtype).T
        )

        recenter_x = float(np.min(output_image_frame[:, 0]))
        recenter_y = float(np.min(output_image_frame[:, 1]))

        output_width = int(np.max(output_image_frame[:, 0]) - recenter_x)
        output_height = int(np.max(output_image_frame[:, 1]) - recenter_y)

        return (output_height, output_width), (recenter_x, recenter_y)

    def _recenter_bounding_boxes_after_expand(self, bounding_boxes, *, recenter_xy):
        x, y = recenter_xy
        if bounding_boxes.format is ta_tensors.BoundingBoxFormat.XYXY:
            translate = [x, y, x, y]
        else:
            translate = [x, y, 0.0, 0.0]
        return ta_tensors.wrap(
            (bounding_boxes.to(torch.float64) - torch.tensor(translate)).to(
                bounding_boxes.dtype
            ),
            like=bounding_boxes,
        )

    def _reference_rotate_bounding_boxes(
        self, bounding_boxes, *, angle, expand, center, is_batch
    ):
        if center is None:
            center = [s * 0.5 for s in bounding_boxes.canvas_size[::-1]]
        cx, cy = center

        a = np.cos(angle * np.pi / 180.0)
        b = np.sin(angle * np.pi / 180.0)
        affine_matrix = np.array(
            [
                [a, b, cx - cx * a - b * cy],
                [-b, a, cy + cx * b - a * cy],
            ],
        )

        new_canvas_size, recenter_xy = self._compute_output_canvas_size(
            expand=expand,
            canvas_size=bounding_boxes.canvas_size,
            affine_matrix=affine_matrix,
        )

        output = reference_affine_bounding_boxes_helper(
            bounding_boxes,
            affine_matrix=affine_matrix,
            new_canvas_size=new_canvas_size,
            clamp=False,
            is_batch=is_batch,
        )

        return F.clamp_bounding_boxes(
            self._recenter_bounding_boxes_after_expand(output, recenter_xy=recenter_xy)
        ).to(bounding_boxes)

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("angle", _CORRECTNESS_AFFINE_KWARGS["angle"])
    @pytest.mark.parametrize("expand", [False, True])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_functional_bounding_boxes_correctness(
        self, format, angle, expand, center, make_input
    ):
        is_batch = make_input == make_batch_bounding_boxes
        bounding_boxes = make_input(format=format)

        actual = F.rotate(bounding_boxes, angle=angle, expand=expand, center=center)
        expected = self._reference_rotate_bounding_boxes(
            bounding_boxes, angle=angle, expand=expand, center=center, is_batch=is_batch
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            F.get_size(actual), F.get_size(expected), atol=2 if expand else 0, rtol=0
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("expand", [False, True])
    @pytest.mark.parametrize("center", _CORRECTNESS_AFFINE_KWARGS["center"])
    @pytest.mark.parametrize("seed", list(range(5)))
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_transform_bounding_boxes_correctness(
        self, format, expand, center, seed, make_input
    ):
        is_batch = make_input == make_batch_bounding_boxes

        if is_batch and expand:
            return

        bounding_boxes = make_input(format=format)

        transform = transforms.RandomRotation(
            **self._CORRECTNESS_TRANSFORM_AFFINE_RANGES,
            expand=expand,
            center=center,
            batch_transform=is_batch,
        )

        torch.manual_seed(seed)
        params = transform._get_params([bounding_boxes], 1, torch.tensor([0]))[0]

        torch.manual_seed(seed)
        actual = transform(bounding_boxes)

        expected = self._reference_rotate_bounding_boxes(
            bounding_boxes, **params, expand=expand, center=center, is_batch=is_batch
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            F.get_size(actual), F.get_size(expected), atol=2 if expand else 0, rtol=0
        )

    @pytest.mark.parametrize(
        "degrees", _EXHAUSTIVE_TYPE_TRANSFORM_AFFINE_RANGES["degrees"]
    )
    @pytest.mark.parametrize("seed", list(range(10)))
    def test_transform_get_params_bounds(self, degrees, seed):
        transform = transforms.RandomRotation(degrees=degrees)

        torch.manual_seed(seed)
        params = transform._get_params([], 1, torch.tensor([0]))[0]

        if isinstance(degrees, (int, float)):
            assert -degrees <= params["angle"] <= degrees
        else:
            assert degrees[0] <= params["angle"] <= degrees[1]

    @pytest.mark.parametrize("param", ["degrees", "center"])
    @pytest.mark.parametrize("value", [0, [0], [0, 0, 0]])
    def test_transform_sequence_len_errors(self, param, value):
        if param == "degrees" and not isinstance(value, list):
            return

        kwargs = {param: value}
        if param != "degrees":
            kwargs["degrees"] = 0

        with pytest.raises(
            ValueError if isinstance(value, list) else TypeError,
            match=f"{param} should be a sequence of length 2",
        ):
            transforms.RandomRotation(**kwargs)

    def test_transform_negative_degrees_error(self):
        with pytest.raises(
            ValueError, match="If degrees is a single number, it must be positive"
        ):
            transforms.RandomAffine(degrees=-1)

    def test_transform_unknown_fill_error(self):
        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.RandomAffine(degrees=0, fill="fill")


class TestCenterCrop:
    INPUT_SIZE = (17, 11)
    OUTPUT_SIZES = [
        (3, 5),
        (5, 3),
        (4, 4),
        (21, 9),
        (13, 15),
        (19, 14),
        3,
        (4,),
        [5],
        INPUT_SIZE,
    ]

    @pytest.mark.parametrize("output_size", OUTPUT_SIZES)
    @pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, output_size, dtype, device, make_input):
        check_kernel(
            F.center_crop_image,
            make_input(self.INPUT_SIZE, dtype=dtype, device=device),
            output_size=output_size,
            check_scripted_vs_eager=not isinstance(output_size, int),
        )

    @pytest.mark.parametrize("output_size", OUTPUT_SIZES)
    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_kernel_bounding_boxes(self, output_size, format, make_bounding_boxes):
        bounding_boxes = make_bounding_boxes(self.INPUT_SIZE, format=format)
        check_kernel(
            F.center_crop_bounding_boxes,
            bounding_boxes,
            format=bounding_boxes.format,
            canvas_size=bounding_boxes.canvas_size,
            output_size=output_size,
            check_scripted_vs_eager=not isinstance(output_size, int),
        )

    @pytest.mark.parametrize(
        "make_mask",
        MASKS_MAKERS,
    )
    def test_kernel_mask(self, make_mask):
        check_kernel(F.center_crop_mask, make_mask(), output_size=self.OUTPUT_SIZES[0])

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(
            F.center_crop_video,
            make_video(self.INPUT_SIZE),
            output_size=self.OUTPUT_SIZES[0],
        )

    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    def test_functional(self, make_input):
        check_functional(
            F.center_crop, make_input(self.INPUT_SIZE), output_size=self.OUTPUT_SIZES[0]
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.center_crop_image, torch.Tensor),
            (F.center_crop_image, ta_tensors.Image),
            (F.center_crop_image, ta_tensors.BatchImages),
            (F.center_crop_bounding_boxes, ta_tensors.BoundingBoxes),
            (F.center_crop_bounding_boxes, ta_tensors.BatchBoundingBoxes),
            (F.center_crop_mask, ta_tensors.Mask),
            (F.center_crop_mask, ta_tensors.BatchMasks),
            (F.center_crop_video, ta_tensors.Video),
            (F.center_crop_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.center_crop, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    def test_transform(self, make_input):
        check_transform(
            transforms.CenterCrop(self.OUTPUT_SIZES[0]), make_input(self.INPUT_SIZE)
        )

    @pytest.mark.parametrize("output_size", OUTPUT_SIZES)
    @pytest.mark.parametrize(
        "fn", [F.center_crop, transform_cls_to_functional(transforms.CenterCrop)]
    )
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_image_correctness(self, output_size, fn, make_input):
        image = make_input(self.INPUT_SIZE, dtype=torch.uint8, device="cpu")

        actual = fn(image, output_size)
        expected = TVF.center_crop(torch.as_tensor(image), output_size=output_size)

        assert_equal(actual, expected)

    def _reference_center_crop_bounding_boxes(
        self, bounding_boxes, output_size, is_batch
    ):
        image_height, image_width = bounding_boxes.canvas_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        elif len(output_size) == 1:
            output_size *= 2
        crop_height, crop_width = output_size

        top = int(round((image_height - crop_height) / 2))
        left = int(round((image_width - crop_width) / 2))

        affine_matrix = np.array(
            [
                [1, 0, -left],
                [0, 1, -top],
            ],
        )
        return reference_affine_bounding_boxes_helper(
            bounding_boxes,
            affine_matrix=affine_matrix,
            new_canvas_size=output_size,
            is_batch=is_batch,
        )

    @pytest.mark.parametrize("output_size", OUTPUT_SIZES)
    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize(
        "fn", [F.center_crop, transform_cls_to_functional(transforms.CenterCrop)]
    )
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_bounding_boxes_correctness(
        self, output_size, format, dtype, device, fn, make_input
    ):
        is_batch = make_input == make_batch_bounding_boxes
        bounding_boxes = make_input(
            self.INPUT_SIZE, format=format, dtype=dtype, device=device
        )

        actual = fn(bounding_boxes, output_size)
        expected = self._reference_center_crop_bounding_boxes(
            bounding_boxes, output_size, is_batch
        )

        assert_equal(actual, expected)


class TestResizedCrop:
    INPUT_SIZE = (17, 11)
    CROP_KWARGS = dict(top=2, left=2, height=5, width=7)
    OUTPUT_SIZE = (19, 32)

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.resized_crop_image, make_image),
            (F.resized_crop_image, make_batch_images),
            (F.resized_crop_bounding_boxes, make_bounding_boxes),
            (F.resized_crop_bounding_boxes, make_batch_bounding_boxes),
            (F.resized_crop_mask, make_segmentation_mask),
            (F.resized_crop_mask, make_batch_segmentation_masks),
            (F.resized_crop_mask, make_detection_masks),
            (F.resized_crop_mask, make_batch_detection_masks),
            (F.resized_crop_video, make_video),
            (F.resized_crop_video, make_batch_videos),
        ],
    )
    def test_kernel(self, kernel, make_input):
        input = make_input(self.INPUT_SIZE)
        if isinstance(input, (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes)):
            extra_kwargs = dict(format=input.format)
        elif isinstance(input, (ta_tensors.Mask, ta_tensors.BatchMasks)):
            extra_kwargs = dict()
        else:
            extra_kwargs = dict(antialias=True)

        check_kernel(
            kernel, input, **self.CROP_KWARGS, size=self.OUTPUT_SIZE, **extra_kwargs
        )

    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    def test_functional(self, make_input):
        check_functional(
            F.resized_crop,
            make_input(self.INPUT_SIZE),
            **self.CROP_KWARGS,
            size=self.OUTPUT_SIZE,
            antialias=True,
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.resized_crop_image, torch.Tensor),
            (F.resized_crop_image, ta_tensors.Image),
            (F.resized_crop_image, ta_tensors.BatchImages),
            (F.resized_crop_bounding_boxes, ta_tensors.BoundingBoxes),
            (F.resized_crop_bounding_boxes, ta_tensors.BatchBoundingBoxes),
            (F.resized_crop_mask, ta_tensors.Mask),
            (F.resized_crop_mask, ta_tensors.BatchMasks),
            (F.resized_crop_video, ta_tensors.Video),
            (F.resized_crop_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.resized_crop, kernel=kernel, input_type=input_type
        )

    @param_value_parametrization(
        scale=[(0.1, 0.2), [0.0, 1.0]],
        ratio=[(0.3, 0.7), [0.1, 5.0]],
    )
    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    def test_transform(self, param, value, make_input):
        check_transform(
            transforms.RandomResizedCrop(
                size=self.OUTPUT_SIZE, **{param: value}, antialias=True
            ),
            make_input(self.INPUT_SIZE),
        )

    @param_value_parametrization(
        scale=[(0.1, 0.2), [0.0, 1.0]],
        ratio=[(0.3, 0.7), [0.1, 5.0]],
    )
    @pytest.mark.parametrize("make_input", BATCH_IMAGES_TENSOR_AND_MAKERS)
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("num_chunks", [-1, 1, 2])
    @pytest.mark.parametrize("permute_chunks", [True, False])
    def test_batch_transform(
        self, param, value, make_input, batch_size, num_chunks, permute_chunks
    ):
        check_batch_transform(
            transforms.RandomResizedCrop(
                size=self.OUTPUT_SIZE,
                **{param: value},
                antialias=True,
                batch_transform=True,
                num_chunks=num_chunks,
                permute_chunks=permute_chunks,
            ),
            make_input(self.INPUT_SIZE, batch_dims=(batch_size,)),
            batch_size=batch_size,
        )

    # `InterpolationMode.NEAREST` is modeled after the buggy `INTER_NEAREST` interpolation of CV2.
    @pytest.mark.parametrize(
        "interpolation",
        set(INTERPOLATION_MODES) - {transforms.InterpolationMode.NEAREST},
    )
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_functional_image_correctness(self, interpolation, make_input):
        image = make_input(self.INPUT_SIZE, dtype=torch.uint8)

        actual = F.resized_crop(
            image,
            **self.CROP_KWARGS,
            size=self.OUTPUT_SIZE,
            interpolation=interpolation,
            antialias=True,
        )
        expected = TVF.resized_crop(
            torch.as_tensor(image),
            **self.CROP_KWARGS,
            size=self.OUTPUT_SIZE,
            interpolation=interpolation,
            antialias=True,
        )

        torch.testing.assert_close(actual, expected, atol=1, rtol=0)

    def _reference_resized_crop_bounding_boxes(
        self, bounding_boxes, *, top, left, height, width, size, is_batch
    ):
        new_height, new_width = size

        crop_affine_matrix = np.array(
            [
                [1, 0, -left],
                [0, 1, -top],
                [0, 0, 1],
            ],
        )
        resize_affine_matrix = np.array(
            [
                [new_width / width, 0, 0],
                [0, new_height / height, 0],
                [0, 0, 1],
            ],
        )
        affine_matrix = (resize_affine_matrix @ crop_affine_matrix)[:2, :]

        return reference_affine_bounding_boxes_helper(
            bounding_boxes,
            affine_matrix=affine_matrix,
            new_canvas_size=size,
            is_batch=is_batch,
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_functional_bounding_boxes_correctness(self, format, make_input):
        is_batch = make_input == make_batch_bounding_boxes
        bounding_boxes = make_input(self.INPUT_SIZE, format=format)

        actual = F.resized_crop(
            bounding_boxes, **self.CROP_KWARGS, size=self.OUTPUT_SIZE
        )
        expected = self._reference_resized_crop_bounding_boxes(
            bounding_boxes, **self.CROP_KWARGS, size=self.OUTPUT_SIZE, is_batch=is_batch
        )

        torch.testing.assert_close(actual, expected)
        assert_equal(F.get_size(actual), F.get_size(expected))

    def test_transform_errors_warnings(self):
        with pytest.raises(ValueError, match="provide only two dimensions"):
            transforms.RandomResizedCrop(size=(1, 2, 3))

        with pytest.raises(TypeError, match="Scale should be a sequence"):
            transforms.RandomResizedCrop(size=self.INPUT_SIZE, scale=123)

        with pytest.raises(TypeError, match="Ratio should be a sequence"):
            transforms.RandomResizedCrop(size=self.INPUT_SIZE, ratio=123)

        for param in ["scale", "ratio"]:
            with pytest.warns(match="Scale and ratio should be of kind"):
                transforms.RandomResizedCrop(size=self.INPUT_SIZE, **{param: [1, 0]})


class TestFiveTenCrop:
    INPUT_SIZE = (17, 11)
    OUTPUT_SIZE = (3, 5)

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("kernel", [F.five_crop_image, F.ten_crop_image])
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, kernel, make_input):
        check_kernel(
            kernel,
            make_input(self.INPUT_SIZE, dtype=dtype, device=device),
            size=self.OUTPUT_SIZE,
            check_batched_vs_unbatched=False,
            check_batch_kernel_leading_dims=False,
        )

    @pytest.mark.parametrize("kernel", [F.five_crop_video, F.ten_crop_video])
    @pytest.mark.parametrize("make_input", VIDEO_MAKERS)
    def test_kernel_video(self, kernel, make_input):
        check_kernel(
            kernel,
            make_input(self.INPUT_SIZE),
            size=self.OUTPUT_SIZE,
            check_batched_vs_unbatched=False,
            check_batch_kernel_leading_dims=False,
        )

    def _functional_wrapper(self, fn):
        # This wrapper is needed to make five_crop / ten_crop compatible with check_functional, since that requires a
        # single output rather than a sequence.
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            outputs = fn(*args, **kwargs)
            return outputs[0]

        return wrapper

    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image,
            make_video,
            make_batch_images,
            make_batch_videos,
        ],
    )
    @pytest.mark.parametrize("functional", [F.five_crop, F.ten_crop])
    def test_functional(self, make_input, functional):
        check_functional(
            self._functional_wrapper(functional),
            make_input(self.INPUT_SIZE),
            size=self.OUTPUT_SIZE,
            check_scripted_smoke=False,
        )

    @pytest.mark.parametrize(
        ("functional", "kernel", "input_type"),
        [
            (F.five_crop, F.five_crop_image, torch.Tensor),
            (F.five_crop, F.five_crop_image, ta_tensors.Image),
            (F.five_crop, F.five_crop_image, ta_tensors.BatchImages),
            (F.five_crop, F.five_crop_video, ta_tensors.Video),
            (F.five_crop, F.five_crop_video, ta_tensors.BatchVideos),
            (F.ten_crop, F.ten_crop_image, torch.Tensor),
            (F.ten_crop, F.ten_crop_image, ta_tensors.Image),
            (F.ten_crop, F.ten_crop_image, ta_tensors.BatchImages),
            (F.ten_crop, F.ten_crop_video, ta_tensors.Video),
            (F.ten_crop, F.ten_crop_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, functional, kernel, input_type):
        check_functional_kernel_signature_match(
            functional, kernel=kernel, input_type=input_type
        )

    class _TransformWrapper(nn.Module):
        # This wrapper is needed to make FiveCrop / TenCrop compatible with check_transform, since that requires a
        # single output rather than a sequence.

        def __init__(self, five_ten_crop_transform):
            super().__init__()
            type(self)._v1_transform_cls = type(self)
            self.five_ten_crop_transform = five_ten_crop_transform

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            outputs = self.five_ten_crop_transform(input)
            return outputs[0]

    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image,
            make_video,
            make_batch_images,
            make_batch_videos,
        ],
    )
    @pytest.mark.parametrize("transform_cls", [transforms.FiveCrop, transforms.TenCrop])
    def test_transform(self, make_input, transform_cls):
        check_transform(
            self._TransformWrapper(transform_cls(size=self.OUTPUT_SIZE)),
            make_input(self.INPUT_SIZE),
            check_sample_input=False,
        )

    @pytest.mark.parametrize(
        "make_input",
        [
            make_bounding_boxes,
            make_detection_masks,
            make_batch_bounding_boxes,
            make_batch_detection_masks,
        ],
    )
    @pytest.mark.parametrize("transform_cls", [transforms.FiveCrop, transforms.TenCrop])
    def test_transform_error(self, make_input, transform_cls):
        transform = transform_cls(size=self.OUTPUT_SIZE)

        with pytest.raises(TypeError, match="not supported"):
            transform(make_input(self.INPUT_SIZE))

    @pytest.mark.parametrize(
        "fn", [F.five_crop, transform_cls_to_functional(transforms.FiveCrop)]
    )
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_correctness_image_five_crop(self, fn, make_input):
        is_batch = make_input == make_batch_images
        image = make_input(self.INPUT_SIZE, dtype=torch.uint8, device="cpu")

        actual = fn(image, size=self.OUTPUT_SIZE)
        expected = TVF.five_crop(torch.as_tensor(image), size=self.OUTPUT_SIZE)

        assert isinstance(actual, tuple)
        assert_equal(
            actual,
            [F.to_image(e) if not is_batch else F.to_batch_images(e) for e in expected],
        )

    @pytest.mark.parametrize("fn_or_class", [F.ten_crop, transforms.TenCrop])
    @pytest.mark.parametrize("vertical_flip", [False, True])
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_correctness_image_ten_crop(self, fn_or_class, vertical_flip, make_input):
        is_batch = make_input == make_batch_images
        if fn_or_class is transforms.TenCrop:
            fn = transform_cls_to_functional(
                fn_or_class, size=self.OUTPUT_SIZE, vertical_flip=vertical_flip
            )
            kwargs = dict()
        else:
            fn = fn_or_class
            kwargs = dict(size=self.OUTPUT_SIZE, vertical_flip=vertical_flip)

        image = make_input(self.INPUT_SIZE, dtype=torch.uint8, device="cpu")

        actual = fn(image, **kwargs)
        expected = TVF.ten_crop(
            torch.as_tensor(image), size=self.OUTPUT_SIZE, vertical_flip=vertical_flip
        )

        assert isinstance(actual, tuple)
        assert_equal(
            actual,
            [F.to_image(e) if not is_batch else F.to_batch_images(e) for e in expected],
        )


class TestPad:
    EXHAUSTIVE_TYPE_PADDINGS = [
        1,
        (1,),
        (1, 2),
        (1, 2, 3, 4),
        [1],
        [1, 2],
        [1, 2, 3, 4],
    ]
    CORRECTNESS_PADDINGS = [
        padding
        for padding in EXHAUSTIVE_TYPE_PADDINGS
        if isinstance(padding, int) or isinstance(padding, list) and len(padding) > 1
    ]
    PADDING_MODES = ["constant", "symmetric", "edge", "reflect"]

    @param_value_parametrization(
        padding=EXHAUSTIVE_TYPE_PADDINGS,
        fill=EXHAUSTIVE_TYPE_FILLS,
        padding_mode=PADDING_MODES,
    )
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, param, value, dtype, device, make_input):
        if param == "fill":
            value = adapt_fill(value, dtype=dtype)
        kwargs = {param: value}
        if param != "padding":
            kwargs["padding"] = [1]

        image = make_input(dtype=dtype, device=device)

        check_kernel(
            F.pad_image,
            image,
            **kwargs,
            check_scripted_vs_eager=not (
                (param == "padding" and isinstance(value, int))
                # See https://github.com/pytorch/vision/pull/7252#issue-1585585521 for details
                or (
                    param == "fill"
                    and (
                        isinstance(value, tuple)
                        or (
                            isinstance(value, list)
                            and any(isinstance(v, int) for v in value)
                        )
                    )
                )
            ),
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_kernel_bounding_boxes(self, format, make_bounding_boxes):
        bounding_boxes = make_bounding_boxes(format=format)
        check_kernel(
            F.pad_bounding_boxes,
            bounding_boxes,
            format=bounding_boxes.format,
            canvas_size=bounding_boxes.canvas_size,
            padding=[1],
        )

    @pytest.mark.parametrize("padding_mode", ["symmetric", "edge", "reflect"])
    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_kernel_bounding_boxes_errors(self, padding_mode, make_bounding_boxes):
        bounding_boxes = make_bounding_boxes()
        with pytest.raises(ValueError, match=f"'{padding_mode}' is not supported"):
            F.pad_bounding_boxes(
                bounding_boxes,
                format=bounding_boxes.format,
                canvas_size=bounding_boxes.canvas_size,
                padding=[1],
                padding_mode=padding_mode,
            )

    @pytest.mark.parametrize(
        "make_mask",
        MASKS_MAKERS,
    )
    def test_kernel_mask(self, make_mask):
        check_kernel(F.pad_mask, make_mask(), padding=[1])

    @pytest.mark.parametrize("fill", [[1], (0,), [1, 0, 1], (0, 1, 0)])
    @pytest.mark.parametrize(
        "make_mask",
        [make_segmentation_mask, make_batch_segmentation_masks],
    )
    def test_kernel_mask_errors(self, fill, make_mask):
        with pytest.raises(ValueError, match="Non-scalar fill value is not supported"):
            F.pad_mask(make_mask(), padding=[1], fill=fill)

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(F.pad_video, make_video(), padding=[1])

    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_functional(self, make_input):
        check_functional(F.pad, make_input(), padding=[1])

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.pad_image, torch.Tensor),
            (F.pad_image, ta_tensors.Image),
            (F.pad_image, ta_tensors.BatchImages),
            (F.pad_bounding_boxes, ta_tensors.BoundingBoxes),
            (F.pad_bounding_boxes, ta_tensors.BatchBoundingBoxes),
            (F.pad_mask, ta_tensors.Mask),
            (F.pad_mask, ta_tensors.BatchMasks),
            (F.pad_video, ta_tensors.Video),
            (F.pad_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.pad, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    def test_transform(self, make_input):
        check_transform(transforms.Pad(padding=[1]), make_input())

    def test_transform_errors(self):
        with pytest.raises(TypeError, match="Got inappropriate padding arg"):
            transforms.Pad("abc")

        with pytest.raises(ValueError, match="Padding must be an int or a 1, 2, or 4"):
            transforms.Pad([-0.7, 0, 0.7])

        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.Pad(12, fill="abc")

        with pytest.raises(ValueError, match="Padding mode should be either"):
            transforms.Pad(12, padding_mode="abc")

    @pytest.mark.parametrize("padding", CORRECTNESS_PADDINGS)
    @pytest.mark.parametrize(
        ("padding_mode", "fill"),
        [
            *[("constant", fill) for fill in CORRECTNESS_FILLS],
            *[
                (padding_mode, None)
                for padding_mode in ["symmetric", "edge", "reflect"]
            ],
        ],
    )
    @pytest.mark.parametrize("fn", [F.pad, transform_cls_to_functional(transforms.Pad)])
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_image_correctness(self, padding, padding_mode, fill, fn, make_input):
        image = make_image(dtype=torch.uint8, device="cpu")

        fill = adapt_fill(fill, dtype=torch.uint8)

        actual = fn(image, padding=padding, padding_mode=padding_mode, fill=fill)
        expected = TVF.pad(
            torch.as_tensor(image),
            padding=padding,
            padding_mode=padding_mode,
            fill=fill,
        )

        assert_equal(actual, expected)

    def _reference_pad_bounding_boxes(self, bounding_boxes, *, padding, is_batch):
        if isinstance(padding, int):
            padding = [padding]
        left, top, right, bottom = padding * (4 // len(padding))

        affine_matrix = np.array(
            [
                [1, 0, left],
                [0, 1, top],
            ],
        )

        height = bounding_boxes.canvas_size[0] + top + bottom
        width = bounding_boxes.canvas_size[1] + left + right

        return reference_affine_bounding_boxes_helper(
            bounding_boxes,
            affine_matrix=affine_matrix,
            new_canvas_size=(height, width),
            is_batch=is_batch,
        )

    @pytest.mark.parametrize("padding", CORRECTNESS_PADDINGS)
    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("fn", [F.pad, transform_cls_to_functional(transforms.Pad)])
    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_bounding_boxes_correctness(
        self, padding, format, dtype, device, fn, make_bounding_boxes
    ):
        is_batch = make_bounding_boxes == make_batch_bounding_boxes
        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)

        actual = fn(bounding_boxes, padding=padding)
        expected = self._reference_pad_bounding_boxes(
            bounding_boxes, padding=padding, is_batch=is_batch
        )

        assert_equal(actual, expected)


class TestRandomZoomOut:
    # Tests are light because this largely relies on the already tested `pad` kernels.

    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    def test_transform(self, make_input):
        check_transform(transforms.RandomZoomOut(p=1), make_input())

    def test_transform_error(self):
        for side_range in [None, 1, [1, 2, 3]]:
            with pytest.raises(
                ValueError if isinstance(side_range, list) else TypeError,
                match="should be a sequence of length 2",
            ):
                transforms.RandomZoomOut(side_range=side_range)

        for side_range in [[0.5, 1.5], [2.0, 1.0]]:
            with pytest.raises(ValueError, match="Invalid side range"):
                transforms.RandomZoomOut(side_range=side_range)

    @pytest.mark.parametrize("side_range", [(1.0, 4.0), [2.0, 5.0]])
    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform_params_correctness(self, side_range, make_input, device):
        transform = transforms.RandomZoomOut(side_range=side_range)

        input = make_input()
        height, width = F.get_size(input)

        params = transform._get_params([input], 1, torch.tensor([0]))[0]
        assert "padding" in params

        padding = params["padding"]
        assert len(padding) == 4

        assert 0 <= padding[0] <= (side_range[1] - 1) * width
        assert 0 <= padding[1] <= (side_range[1] - 1) * height
        assert 0 <= padding[2] <= (side_range[1] - 1) * width
        assert 0 <= padding[3] <= (side_range[1] - 1) * height


class TestCrop:
    INPUT_SIZE = (21, 11)

    CORRECTNESS_CROP_KWARGS = [
        # center
        dict(top=5, left=5, height=10, width=5),
        # larger than input, i.e. pad
        dict(top=-5, left=-5, height=30, width=20),
        # sides: left, right, top, bottom
        dict(top=-5, left=-5, height=30, width=10),
        dict(top=-5, left=5, height=30, width=10),
        dict(top=-5, left=-5, height=20, width=20),
        dict(top=5, left=-5, height=20, width=20),
        # corners: top-left, top-right, bottom-left, bottom-right
        dict(top=-5, left=-5, height=20, width=10),
        dict(top=-5, left=5, height=20, width=10),
        dict(top=5, left=-5, height=20, width=10),
        dict(top=5, left=5, height=20, width=10),
    ]
    MINIMAL_CROP_KWARGS = CORRECTNESS_CROP_KWARGS[0]

    @pytest.mark.parametrize("kwargs", CORRECTNESS_CROP_KWARGS)
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, kwargs, dtype, device, make_input):
        check_kernel(
            F.crop_image,
            make_input(self.INPUT_SIZE, dtype=dtype, device=device),
            **kwargs,
        )

    @pytest.mark.parametrize("kwargs", CORRECTNESS_CROP_KWARGS)
    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_kernel_bounding_box(
        self, kwargs, format, dtype, device, make_bounding_boxes
    ):
        bounding_boxes = make_bounding_boxes(
            self.INPUT_SIZE, format=format, dtype=dtype, device=device
        )
        check_kernel(F.crop_bounding_boxes, bounding_boxes, format=format, **kwargs)

    @pytest.mark.parametrize(
        "make_mask",
        MASKS_MAKERS,
    )
    def test_kernel_mask(self, make_mask):
        check_kernel(
            F.crop_mask, make_mask(self.INPUT_SIZE), **self.MINIMAL_CROP_KWARGS
        )

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(
            F.crop_video, make_video(self.INPUT_SIZE), **self.MINIMAL_CROP_KWARGS
        )

    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    def test_functional(self, make_input):
        check_functional(
            F.crop, make_input(self.INPUT_SIZE), **self.MINIMAL_CROP_KWARGS
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.crop_image, torch.Tensor),
            (F.crop_image, ta_tensors.Image),
            (F.crop_image, ta_tensors.BatchImages),
            (F.crop_bounding_boxes, ta_tensors.BoundingBoxes),
            (F.crop_bounding_boxes, ta_tensors.BatchBoundingBoxes),
            (F.crop_mask, ta_tensors.Mask),
            (F.crop_mask, ta_tensors.BatchMasks),
            (F.crop_video, ta_tensors.Video),
            (F.crop_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.crop, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize("kwargs", CORRECTNESS_CROP_KWARGS)
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_functional_image_correctness(self, kwargs, make_input):
        image = make_input(self.INPUT_SIZE, dtype=torch.uint8, device="cpu")

        actual = F.crop(image, **kwargs)
        expected = TVF.crop(torch.as_tensor(image), **kwargs)

        assert_equal(actual, expected)

    @param_value_parametrization(
        size=[(10, 5), (25, 15), (25, 5), (10, 15)],
        fill=EXHAUSTIVE_TYPE_FILLS,
    )
    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    def test_transform(self, param, value, make_input):
        input = make_input(self.INPUT_SIZE)

        check_sample_input = True
        if param == "fill":
            if isinstance(value, (tuple, list)):
                if isinstance(input, (ta_tensors.Mask, ta_tensors.BatchMasks)):
                    pytest.skip("F.pad_mask doesn't support non-scalar fill.")
                else:
                    check_sample_input = False

            kwargs = dict(
                # 1. size is required
                # 2. the fill parameter only has an affect if we need padding
                size=[s + 4 for s in self.INPUT_SIZE],
                fill=adapt_fill(
                    value,
                    dtype=input.dtype
                    if isinstance(input, torch.Tensor)
                    else torch.uint8,
                ),
            )
        else:
            kwargs = {param: value}

        check_transform(
            transforms.RandomCrop(**kwargs, pad_if_needed=True),
            input,
            check_sample_input=check_sample_input,
        )

    @pytest.mark.parametrize("padding", [1, (1, 1), (1, 1, 1, 1)])
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_transform_padding(self, padding, make_input):
        inpt = make_input(self.INPUT_SIZE)

        output_size = [s + 2 for s in F.get_size(inpt)]
        transform = transforms.RandomCrop(output_size, padding=padding)

        output = transform(inpt)

        assert F.get_size(output) == output_size

    @pytest.mark.parametrize("padding", [None, 1, (1, 1), (1, 1, 1, 1)])
    def test_transform_insufficient_padding(self, padding):
        inpt = make_image(self.INPUT_SIZE)

        output_size = [s + 3 for s in F.get_size(inpt)]
        transform = transforms.RandomCrop(output_size, padding=padding)

        with pytest.raises(ValueError, match="larger than (padded )?input image size"):
            transform(inpt)

    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_transform_pad_if_needed(self, make_input):
        inpt = make_input(self.INPUT_SIZE)

        output_size = [s * 2 for s in F.get_size(inpt)]
        transform = transforms.RandomCrop(output_size, pad_if_needed=True)

        output = transform(inpt)

        assert F.get_size(output) == output_size

    @param_value_parametrization(
        size=[(10, 5), (25, 15), (25, 5), (10, 15)],
        fill=CORRECTNESS_FILLS,
        padding_mode=["constant", "edge", "reflect", "symmetric"],
    )
    @pytest.mark.parametrize("seed", list(range(5)))
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_transform_image_correctness(self, param, value, seed, make_input):
        kwargs = {param: value}
        if param != "size":
            # 1. size is required
            # 2. the fill / padding_mode parameters only have an affect if we need padding
            kwargs["size"] = [s + 4 for s in self.INPUT_SIZE]
        if param == "fill":
            kwargs["fill"] = adapt_fill(kwargs["fill"], dtype=torch.uint8)

        transform = transforms.RandomCrop(pad_if_needed=True, **kwargs)
        torchvision_transform = tv_transforms.RandomCrop(pad_if_needed=True, **kwargs)

        image = make_input(self.INPUT_SIZE)

        with freeze_rng_state():
            torch.manual_seed(seed)
            actual = transform(image)

            torch.manual_seed(seed)
            expected = torchvision_transform(torch.as_tensor(image))

        assert_equal(actual, expected)

    def _reference_crop_bounding_boxes(
        self, bounding_boxes, *, top, left, height, width, is_batch
    ):
        affine_matrix = np.array(
            [
                [1, 0, -left],
                [0, 1, -top],
            ],
        )
        return reference_affine_bounding_boxes_helper(
            bounding_boxes,
            affine_matrix=affine_matrix,
            new_canvas_size=(height, width),
            is_batch=is_batch,
        )

    @pytest.mark.parametrize("kwargs", CORRECTNESS_CROP_KWARGS)
    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_functional_bounding_box_correctness(
        self, kwargs, format, dtype, device, make_bounding_boxes
    ):
        is_batch = make_bounding_boxes == make_batch_bounding_boxes
        bounding_boxes = make_bounding_boxes(
            self.INPUT_SIZE, format=format, dtype=dtype, device=device
        )

        actual = F.crop(bounding_boxes, **kwargs)
        expected = self._reference_crop_bounding_boxes(
            bounding_boxes, **kwargs, is_batch=is_batch
        )

        assert_equal(actual, expected, atol=1, rtol=0)
        assert_equal(F.get_size(actual), F.get_size(expected))

    @pytest.mark.parametrize("output_size", [(17, 11), (11, 17), (11, 11)])
    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seed", list(range(5)))
    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_transform_bounding_boxes_correctness(
        self, output_size, format, dtype, device, seed, make_bounding_boxes
    ):
        input_size = [s * 2 for s in output_size]

        is_batch = make_bounding_boxes == make_batch_bounding_boxes
        bounding_boxes = make_bounding_boxes(
            input_size, format=format, dtype=dtype, device=device
        )

        transform = transforms.RandomCrop(output_size)

        with freeze_rng_state():
            torch.manual_seed(seed)
            params = transform._get_params([bounding_boxes], 1, torch.tensor([0]))[0]
            assert not params.pop("needs_pad")
            del params["padding"]
            assert params.pop("needs_crop")

            torch.manual_seed(seed)
            actual = transform(bounding_boxes)

        expected = self._reference_crop_bounding_boxes(
            bounding_boxes, **params, is_batch=is_batch
        )

        assert_equal(actual, expected)
        assert_equal(F.get_size(actual), F.get_size(expected))

    def test_errors(self):
        with pytest.raises(ValueError, match="Please provide only two dimensions"):
            transforms.RandomCrop([10, 12, 14])

        with pytest.raises(TypeError, match="Got inappropriate padding arg"):
            transforms.RandomCrop([10, 12], padding="abc")

        with pytest.raises(ValueError, match="Padding must be an int or a 1, 2, or 4"):
            transforms.RandomCrop([10, 12], padding=[-0.7, 0, 0.7])

        with pytest.raises(TypeError, match="Got inappropriate fill arg"):
            transforms.RandomCrop([10, 12], padding=1, fill="abc")

        with pytest.raises(ValueError, match="Padding mode should be either"):
            transforms.RandomCrop([10, 12], padding=1, padding_mode="abc")


class TestPerspective:
    COEFFICIENTS = [
        [1.2405, 0.1772, -6.9113, 0.0463, 1.251, -5.235, 0.00013, 0.0018],
        [0.7366, -0.11724, 1.45775, -0.15012, 0.73406, 2.6019, -0.0072, -0.0063],
    ]
    START_END_POINTS = [
        ([[0, 0], [33, 0], [33, 25], [0, 25]], [[3, 2], [32, 3], [30, 24], [2, 25]]),
        ([[3, 2], [32, 3], [30, 24], [2, 25]], [[0, 0], [33, 0], [33, 25], [0, 25]]),
        ([[3, 2], [32, 3], [30, 24], [2, 25]], [[5, 5], [30, 3], [33, 19], [4, 25]]),
    ]
    MINIMAL_KWARGS = dict(
        startpoints=None, endpoints=None, coefficients=COEFFICIENTS[0]
    )

    @param_value_parametrization(
        coefficients=COEFFICIENTS,
        start_end_points=START_END_POINTS,
        fill=EXHAUSTIVE_TYPE_FILLS,
    )
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, param, value, dtype, device, make_input):
        if param == "start_end_points":
            kwargs = dict(zip(["startpoints", "endpoints"], value))
        else:
            kwargs = {"startpoints": None, "endpoints": None, param: value}
        if param == "fill":
            kwargs["coefficients"] = self.COEFFICIENTS[0]

        check_kernel(
            F.perspective_image,
            make_input(dtype=dtype, device=device),
            **kwargs,
            check_scripted_vs_eager=not (
                param == "fill" and isinstance(value, (int, float))
            ),
        )

    def test_kernel_image_error(self):
        image = make_image_tensor()

        with pytest.raises(
            ValueError,
            match="startpoints/endpoints or the coefficients must have non `None` values",
        ):
            F.perspective_image(image, startpoints=None, endpoints=None)

        with pytest.raises(
            ValueError,
            match="startpoints/endpoints and the coefficients shouldn't be defined concurrently",
        ):
            startpoints, endpoints = self.START_END_POINTS[0]
            coefficients = self.COEFFICIENTS[0]
            F.perspective_image(
                image,
                startpoints=startpoints,
                endpoints=endpoints,
                coefficients=coefficients,
            )

        with pytest.raises(ValueError, match="coefficients should have 8 float values"):
            F.perspective_image(
                image, startpoints=None, endpoints=None, coefficients=list(range(7))
            )

    @param_value_parametrization(
        coefficients=COEFFICIENTS,
        start_end_points=START_END_POINTS,
    )
    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_kernel_bounding_boxes(self, param, value, format, make_bounding_boxes):
        if param == "start_end_points":
            kwargs = dict(zip(["startpoints", "endpoints"], value))
        else:
            kwargs = {"startpoints": None, "endpoints": None, param: value}

        bounding_boxes = make_bounding_boxes(format=format)

        check_kernel(
            F.perspective_bounding_boxes,
            bounding_boxes,
            format=bounding_boxes.format,
            canvas_size=bounding_boxes.canvas_size,
            **kwargs,
        )

    def test_kernel_bounding_boxes_error(self):
        bounding_boxes = make_bounding_boxes()
        format, canvas_size = bounding_boxes.format, bounding_boxes.canvas_size
        bounding_boxes = bounding_boxes.as_subclass(torch.Tensor)

        with pytest.raises(RuntimeError, match="Denominator is zero"):
            F.perspective_bounding_boxes(
                bounding_boxes,
                format=format,
                canvas_size=canvas_size,
                startpoints=None,
                endpoints=None,
                coefficients=[0.0] * 8,
            )

    @pytest.mark.parametrize("make_mask", MASKS_MAKERS)
    def test_kernel_mask(self, make_mask):
        check_kernel(F.perspective_mask, make_mask(), **self.MINIMAL_KWARGS)

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(F.perspective_video, make_video(), **self.MINIMAL_KWARGS)

    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_functional(self, make_input):
        check_functional(F.perspective, make_input(), **self.MINIMAL_KWARGS)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.perspective_image, torch.Tensor),
            (F.perspective_image, ta_tensors.Image),
            (F.perspective_image, ta_tensors.BatchImages),
            (F.perspective_bounding_boxes, ta_tensors.BoundingBoxes),
            (F.perspective_bounding_boxes, ta_tensors.BatchBoundingBoxes),
            (F.perspective_mask, ta_tensors.Mask),
            (F.perspective_mask, ta_tensors.BatchMasks),
            (F.perspective_video, ta_tensors.Video),
            (F.perspective_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.perspective, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize("distortion_scale", [0.5, 0.0, 1.0])
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_transform(self, distortion_scale, make_input):
        try:
            check_transform(
                transforms.RandomPerspective(distortion_scale=distortion_scale, p=1),
                make_input(),
            )
        except Exception as e:
            # TODO: remove this once the error message is fixed
            if (
                "The least squares solution could not be computed because the input matrix does not have full rank"
                not in str(e)
            ):
                pytest.fail("Unexpected error message: {}".format(str(e)))

    @pytest.mark.parametrize("distortion_scale", [0.5, 0.0, 1.0])
    @pytest.mark.parametrize("make_input", BATCH_IMAGES_TENSOR_AND_MAKERS)
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("p", [0, 0.5, 1])
    @pytest.mark.parametrize("batch_inplace", [True, False])
    @pytest.mark.parametrize("num_chunks", [-1, 1, 2])
    @pytest.mark.parametrize("permute_chunks", [True, False])
    def test_batch_transform(
        self,
        distortion_scale,
        make_input,
        batch_size,
        p,
        batch_inplace,
        num_chunks,
        permute_chunks,
    ):
        try:
            check_batch_transform(
                transforms.RandomPerspective(
                    distortion_scale=distortion_scale,
                    p=p,
                    batch_transform=True,
                    num_chunks=num_chunks,
                    permute_chunks=permute_chunks,
                    batch_inplace=batch_inplace,
                ),
                make_input(batch_dims=(batch_size,)),
                batch_size=batch_size,
            )
        except Exception as e:
            # TODO: remove this once the error message is fixed
            if (
                "The least squares solution could not be computed because the input matrix does not have full rank"
                not in str(e)
            ):
                pytest.fail("Unexpected error message: {}".format(str(e)))

    @pytest.mark.parametrize("distortion_scale", [-1, 2])
    def test_transform_error(self, distortion_scale):
        with pytest.raises(
            ValueError, match="distortion_scale value should be between 0 and 1"
        ):
            transforms.RandomPerspective(distortion_scale=distortion_scale)

    @pytest.mark.parametrize("coefficients", COEFFICIENTS)
    @pytest.mark.parametrize(
        "interpolation",
        [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR],
    )
    @pytest.mark.parametrize("fill", CORRECTNESS_FILLS)
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_image_functional_correctness(
        self, coefficients, interpolation, fill, make_input
    ):
        image = make_input(dtype=torch.uint8, device="cpu")

        actual = F.perspective(
            image,
            startpoints=None,
            endpoints=None,
            coefficients=coefficients,
            interpolation=interpolation,
            fill=fill,
        )
        expected = TVF.perspective(
            torch.as_tensor(image),
            startpoints=None,
            endpoints=None,
            coefficients=coefficients,
            interpolation=interpolation,
            fill=fill,
        )

        if interpolation is transforms.InterpolationMode.BILINEAR:
            abs_diff = (actual.float() - expected.float()).abs()
            assert (abs_diff > 1).float().mean() < 7e-2
            mae = abs_diff.mean()
            assert mae < 3
        else:
            assert_equal(actual, expected)

    def _reference_perspective_bounding_boxes(
        self, bounding_boxes, *, startpoints, endpoints, is_batch
    ):
        format = bounding_boxes.format
        canvas_size = bounding_boxes.canvas_size
        dtype = bounding_boxes.dtype
        device = bounding_boxes.device
        if is_batch:
            idx_sample = bounding_boxes.idx_sample

        coefficients = _get_perspective_coeffs(endpoints, startpoints)

        def perspective_bounding_boxes(bounding_boxes):
            m1 = np.array(
                [
                    [coefficients[0], coefficients[1], coefficients[2]],
                    [coefficients[3], coefficients[4], coefficients[5]],
                ]
            )
            m2 = np.array(
                [
                    [coefficients[6], coefficients[7], 1.0],
                    [coefficients[6], coefficients[7], 1.0],
                ]
            )

            # Go to float before converting to prevent precision loss in case of CXCYWH -> XYXY and W or H is 1
            input_xyxy = F.convert_bounding_box_format(
                bounding_boxes.to(dtype=torch.float64, device="cpu", copy=True),
                old_format=format,
                new_format=ta_tensors.BoundingBoxFormat.XYXY,
                inplace=True,
            )
            x1, y1, x2, y2 = input_xyxy.squeeze(0).tolist()

            points = np.array(
                [
                    [x1, y1, 1.0],
                    [x2, y1, 1.0],
                    [x1, y2, 1.0],
                    [x2, y2, 1.0],
                ]
            )

            numerator = points @ m1.T
            denominator = points @ m2.T
            transformed_points = numerator / denominator

            output_xyxy = torch.Tensor(
                [
                    float(np.min(transformed_points[:, 0])),
                    float(np.min(transformed_points[:, 1])),
                    float(np.max(transformed_points[:, 0])),
                    float(np.max(transformed_points[:, 1])),
                ]
            )

            output = F.convert_bounding_box_format(
                output_xyxy,
                old_format=ta_tensors.BoundingBoxFormat.XYXY,
                new_format=format,
            )

            # It is important to clamp before casting, especially for CXCYWH format, dtype=int64
            return F.clamp_bounding_boxes(
                output,
                format=format,
                canvas_size=canvas_size,
            ).to(dtype=dtype, device=device)

        return (
            ta_tensors.BoundingBoxes(
                torch.cat(
                    [
                        perspective_bounding_boxes(b)
                        for b in bounding_boxes.reshape(-1, 4).unbind()
                    ],
                    dim=0,
                ).reshape(bounding_boxes.shape),
                format=format,
                canvas_size=canvas_size,
            )
            if not is_batch
            else ta_tensors.BatchBoundingBoxes(
                torch.cat(
                    [
                        perspective_bounding_boxes(b)
                        for b in bounding_boxes.reshape(-1, 4).unbind()
                    ],
                    dim=0,
                ).reshape(bounding_boxes.shape),
                format=format,
                canvas_size=canvas_size,
                idx_sample=idx_sample,
            )
        )

    @pytest.mark.parametrize(("startpoints", "endpoints"), START_END_POINTS)
    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_correctness_perspective_bounding_boxes(
        self, startpoints, endpoints, format, dtype, device, make_bounding_boxes
    ):
        is_batch = make_bounding_boxes == make_batch_bounding_boxes
        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)

        actual = F.perspective(
            bounding_boxes, startpoints=startpoints, endpoints=endpoints
        )

        expected = self._reference_perspective_bounding_boxes(
            bounding_boxes,
            startpoints=startpoints,
            endpoints=endpoints,
            is_batch=is_batch,
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=1)


class TestElastic:
    def _make_displacement(self, inpt):
        return torch.rand(
            1,
            *F.get_size(inpt),
            2,
            dtype=torch.float32,
            device=inpt.device if isinstance(inpt, torch.Tensor) else "cpu",
        )

    def _make_batch_displacement(self, inpt):
        if isinstance(inpt, ta_tensors.BatchMasks):
            return torch.rand(
                inpt.batch_size,
                *F.get_size(inpt),
                2,
                dtype=torch.float32,
                device=inpt.device if isinstance(inpt, torch.Tensor) else "cpu",
            )

        return torch.rand(
            inpt.shape[0],
            *F.get_size(inpt),
            2,
            dtype=torch.float32,
            device=inpt.device if isinstance(inpt, torch.Tensor) else "cpu",
        )

    def _make_batch_bounding_boxes_displacement(self, inpt):
        return torch.rand(
            inpt.batch_size,
            *inpt.canvas_size,
            2,
            dtype=torch.float32,
            device=inpt.device if isinstance(inpt, torch.Tensor) else "cpu",
        )

    def _make_batch_masks_displacement(self, inpt):
        return torch.rand(
            inpt.batch_size,
            *F.get_size(inpt),
            2,
            dtype=torch.float32,
            device=inpt.device if isinstance(inpt, torch.Tensor) else "cpu",
        )

    @param_value_parametrization(
        interpolation=[
            transforms.InterpolationMode.NEAREST,
            transforms.InterpolationMode.BILINEAR,
        ],
        fill=EXHAUSTIVE_TYPE_FILLS,
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8, torch.float16])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, param, value, dtype, device, make_input):
        image = make_input(dtype=dtype, device=device)

        check_kernel(
            F.elastic_image,
            image,
            displacement=self._make_displacement(image),
            **{param: value},
            check_scripted_vs_eager=not (
                param == "fill" and isinstance(value, (int, float))
            ),
            check_cuda_vs_cpu=dtype is not torch.float16,
        )

    @param_value_parametrization(
        interpolation=[
            transforms.InterpolationMode.NEAREST,
            transforms.InterpolationMode.BILINEAR,
        ],
        fill=EXHAUSTIVE_TYPE_FILLS,
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8, torch.float16])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", [make_batch_images])
    def test_kernel_batch_image(self, param, value, dtype, device, make_input):
        image = make_input(dtype=dtype, device=device)
        check_kernel(
            F.elastic_batch_images,
            image,
            displacement=self._make_batch_displacement(image),
            **{param: value},
            check_scripted_vs_eager=not (
                param == "fill" and isinstance(value, (int, float))
            ),
            check_cuda_vs_cpu=dtype is not torch.float16,
            check_batch_kernel_leading_dims=True,
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_kernel_bounding_boxes(self, format, dtype, device, make_bounding_boxes):
        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)

        check_kernel(
            F.elastic_bounding_boxes,
            bounding_boxes,
            format=bounding_boxes.format,
            canvas_size=bounding_boxes.canvas_size,
            displacement=self._make_displacement(bounding_boxes),
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_bounding_boxes", [make_batch_bounding_boxes])
    def test_kernel_batch_bounding_boxes(
        self, format, dtype, device, make_bounding_boxes
    ):
        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)

        check_kernel(
            F.elastic_batch_bounding_boxes,
            bounding_boxes,
            format=bounding_boxes.format,
            canvas_size=bounding_boxes.canvas_size,
            displacement=self._make_batch_displacement(bounding_boxes),
        )

    @pytest.mark.parametrize(
        "make_mask",
        MASKS_MAKERS,
    )
    def test_kernel_mask(self, make_mask):
        mask = make_mask()
        check_kernel(F.elastic_mask, mask, displacement=self._make_displacement(mask))

    @pytest.mark.parametrize(
        "make_mask",
        [make_batch_detection_masks, make_batch_segmentation_masks],
    )
    def test_kernel_batch_masks(self, make_mask):
        mask = make_mask()
        displacement = self._make_batch_displacement(mask)
        displacement = displacement.repeat_interleave(
            torch.tensor(
                [mask.get_num_masks_sample(i) for i in range(mask.batch_size)]
            ),
            dim=0,
        )

        check_kernel(
            F.elastic_batch_masks,
            mask,
            displacement=displacement,
            check_batch_kernel_leading_dims=False,
        )

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        video = make_video()
        check_kernel(
            F.elastic_video, video, displacement=self._make_displacement(video)
        )

    @pytest.mark.parametrize("make_video", [make_batch_videos])
    def test_kernel_batch_videos(self, make_video):
        video = make_video()
        check_kernel(
            F.elastic_batch_videos,
            video,
            displacement=self._make_batch_displacement(video),
            check_batch_kernel_leading_dims=False,
        )

    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    def test_functional(self, make_input):
        if make_input == make_segmentation_mask:
            pytest.skip("Segmentation mask is not supported in the functional test")

        input = make_input()
        check_functional(F.elastic, input, displacement=self._make_displacement(input))

    @pytest.mark.parametrize("make_input", BATCH_IMAGES_TENSOR_AND_MAKERS)
    def test_batch_functional(self, make_input):
        input = make_input()
        check_functional(
            F.elastic_batch, input, displacement=self._make_batch_displacement(input)
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.elastic_image, torch.Tensor),
            (F.elastic_image, ta_tensors.Image),
            (F.elastic_bounding_boxes, ta_tensors.BoundingBoxes),
            (F.elastic_mask, ta_tensors.Mask),
            (F.elastic_video, ta_tensors.Video),
            (F.elastic_image, ta_tensors.BatchImages),
            (F.elastic_bounding_boxes, ta_tensors.BatchBoundingBoxes),
            (F.elastic_mask, ta_tensors.BatchMasks),
            (F.elastic_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.elastic, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.elastic_batch_images, torch.Tensor),
            (F.elastic_batch_images, ta_tensors.BatchImages),
            (F.elastic_batch_bounding_boxes, ta_tensors.BatchBoundingBoxes),
            (F.elastic_batch_masks, ta_tensors.BatchMasks),
            (F.elastic_batch_videos, ta_tensors.BatchVideos),
        ],
    )
    def test_batch_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.elastic_batch, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    def test_displacement_error(self, make_input):
        input = make_input()

        with pytest.raises(TypeError, match="displacement should be a Tensor"):
            F.elastic(input, displacement=None)

        with pytest.raises(ValueError, match="displacement shape should be"):
            F.elastic(input, displacement=torch.rand(F.get_size(input)))

    @pytest.mark.parametrize(
        "make_input", [make_batch_images_tensor, make_batch_images, make_batch_videos]
    )
    def test_batch_displacement_error(self, make_input):
        input = make_input()

        with pytest.raises(TypeError, match="displacement should be a Tensor"):
            F.elastic_batch(input, displacement=None)

        with pytest.raises(ValueError, match="displacement shape should be"):
            F.elastic_batch(input, displacement=torch.rand(F.get_size(input)))

    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image,
            make_bounding_boxes,
            make_segmentation_mask,
            make_video,
            make_batch_images,
            make_batch_bounding_boxes,
            make_batch_segmentation_masks,
            make_batch_videos,
        ],
    )
    # ElasticTransform needs larger images to avoid the needed internal padding being larger than the actual image
    @pytest.mark.parametrize("size", [(163, 163), (72, 333), (313, 95)])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, size, device):
        # We have to skip that test on M1 because it's flaky: Mismatched elements: 35 / 89205 (0.0%)
        # See https://github.com/pytorch/vision/issues/8154
        # All other platforms are fine, so the differences do not come from something we own in torchvision
        check_transform(
            transforms.ElasticTransform(),
            make_input(size, device=device),
        )

    @pytest.mark.parametrize(
        "make_input",
        [
            make_batch_images_tensor,
            make_batch_images,
            make_batch_bounding_boxes,
            make_batch_segmentation_masks,
            make_batch_videos,
        ],
    )
    # ElasticTransform needs larger images to avoid the needed internal padding being larger than the actual image
    @pytest.mark.parametrize("size", [(163, 163), (72, 333), (313, 95)])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("batch_inplace", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_transform(self, make_input, size, device, batch_inplace, batch_size):
        # We have to skip that test on M1 because it's flaky: Mismatched elements: 35 / 89205 (0.0%)
        # See https://github.com/pytorch/vision/issues/8154
        # All other platforms are fine, so the differences do not come from something we own in torchvision
        check_batch_transform(
            transforms.ElasticTransform(
                batch_transform=True, batch_inplace=batch_inplace
            ),
            make_input(size, device=device, batch_dims=(batch_size,)),
            batch_size=batch_size,
        )

    @pytest.mark.parametrize(
        "interpolation",
        [
            transforms.InterpolationMode.NEAREST,
            transforms.InterpolationMode.BILINEAR,
        ],
    )
    @pytest.mark.parametrize("fill", EXHAUSTIVE_TYPE_FILLS)
    @pytest.mark.parametrize(
        "make_input",
        [
            make_image,
            make_batch_images,
            make_detection_masks,
            make_segmentation_mask,
            make_video,
        ],
    )
    def test_image_correctness(self, interpolation, fill, make_input):
        image = make_image(dtype=torch.uint8)
        displacement = self._make_displacement(image)

        actual = F.elastic(image, displacement, interpolation, fill)
        expected = TVF.elastic(
            torch.as_tensor(image), displacement, interpolation=interpolation, fill=fill
        )

        torch.testing.assert_close(actual, expected, atol=1, rtol=0)

    @pytest.mark.parametrize(
        "interpolation",
        [
            transforms.InterpolationMode.NEAREST,
            transforms.InterpolationMode.BILINEAR,
        ],
    )
    @pytest.mark.parametrize("fill", EXHAUSTIVE_TYPE_FILLS)
    @pytest.mark.parametrize("make_image", [make_batch_images, make_batch_videos])
    def test_batch_images_correctness(self, interpolation, fill, make_image):
        images = make_batch_images(dtype=torch.uint8, device="cpu", batch_dims=(4,))
        displacement = self._make_batch_displacement(images)

        actual = F.elastic_batch(images, displacement, interpolation, fill)
        tensor_images = torch.as_tensor(images)

        for i, a in enumerate(actual):
            tensor_image = tensor_images[i]
            e = TVF.elastic(
                tensor_image,
                displacement[i].unsqueeze(0),
                interpolation=interpolation,
                fill=fill,
            )
            torch.testing.assert_close(a, e, rtol=0, atol=1)

    @pytest.mark.parametrize(
        "interpolation",
        [
            transforms.InterpolationMode.NEAREST,
            transforms.InterpolationMode.BILINEAR,
        ],
    )
    @pytest.mark.parametrize("fill", EXHAUSTIVE_TYPE_FILLS)
    @pytest.mark.parametrize("make_boxes", [make_batch_bounding_boxes])
    def test_batch_bounding_boxes_correctness(self, interpolation, fill, make_boxes):
        boxes = make_boxes(
            dtype=torch.uint8, device="cpu", batch_dims=(4,), num_boxes=2
        )
        displacement = self._make_batch_bounding_boxes_displacement(boxes)

        actual = F.elastic_batch(boxes, displacement, interpolation, fill)

        for i in range(boxes.batch_size):
            a = actual.get_sample(i)
            tensor_box = torch.as_tensor(boxes.get_sample(i))
            e = TVF.elastic_bounding_boxes(
                tensor_box,
                format=boxes.format,
                canvas_size=boxes.canvas_size,
                displacement=displacement[i].unsqueeze(0),
            )
            torch.testing.assert_close(a, e, rtol=0, atol=1)

    @pytest.mark.parametrize(
        "interpolation",
        [
            transforms.InterpolationMode.NEAREST,
            transforms.InterpolationMode.BILINEAR,
        ],
    )
    @pytest.mark.parametrize("fill", EXHAUSTIVE_TYPE_FILLS)
    @pytest.mark.parametrize(
        "make_masks", [make_batch_detection_masks, make_batch_segmentation_masks]
    )
    def test_batch_masks_correctness(self, interpolation, fill, make_masks):
        if fill is not None and not isinstance(fill, (int, float)) and len(fill) >= 3:
            pytest.skip("Fill value is not supported for masks")

        masks = make_masks(dtype=torch.uint8, device="cpu", batch_dims=(4,))
        displacement = self._make_batch_masks_displacement(masks)

        actual = F.elastic_batch(masks, displacement, interpolation, fill)

        for i in range(masks.batch_size):
            a = actual.get_sample(i)
            tensor_box = torch.as_tensor(masks.get_sample(i))
            e = TVF.elastic_mask(
                tensor_box, displacement=displacement[i].unsqueeze(0), fill=fill
            )
            torch.testing.assert_close(a, e, rtol=0, atol=1)


class TestRandomIoUCrop:
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("options", [[0.5, 0.9], [2.0]])
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test__get_params(self, device, options, make_image):
        orig_h, orig_w = size = (24, 32)
        image = make_image(size)
        is_batch = make_image == make_batch_images
        bboxes = (
            ta_tensors.BoundingBoxes(
                torch.tensor(
                    [[1, 1, 10, 10], [20, 20, 23, 23], [1, 20, 10, 23], [20, 1, 23, 10]]
                ),
                format="XYXY",
                canvas_size=size,
                device=device,
            )
            if not is_batch
            else ta_tensors.BatchBoundingBoxes(
                torch.tensor(
                    [
                        [1, 1, 10, 10],
                        [20, 20, 23, 23],
                        [1, 20, 10, 23],
                        [20, 1, 23, 10],
                    ],
                ),
                format="XYXY",
                canvas_size=size,
                device=device,
                idx_sample=[0, 1, 2],
            )
        )
        sample = [image, bboxes]

        transform = transforms.RandomIoUCrop(
            sampler_options=options, batch_transform=is_batch
        )

        n_samples = 5
        for _ in range(n_samples):

            list_params = transform._get_params(
                sample,
                num_chunks=1 if not is_batch else 2,
                chunks_indices=[torch.tensor([0]), torch.tensor([1])]
                if is_batch
                else [torch.tensor([0])],
            )

            for params in list_params:

                if options == [2.0]:
                    assert len(params) == 0
                    return

                assert len(params["is_within_crop_area"]) > 0
                assert params["is_within_crop_area"].dtype == torch.bool

                assert (
                    int(transform.min_scale * orig_h)
                    <= params["height"]
                    <= int(transform.max_scale * orig_h)
                )
                assert (
                    int(transform.min_scale * orig_w)
                    <= params["width"]
                    <= int(transform.max_scale * orig_w)
                )

                left, top = params["left"], params["top"]
                new_h, new_w = params["height"], params["width"]
                ious = box_iou(
                    bboxes,
                    torch.tensor(
                        [[left, top, left + new_w, top + new_h]],
                        dtype=bboxes.dtype,
                        device=bboxes.device,
                    ),
                )
                assert (
                    ious.max() >= options[0] or ious.max() >= options[1]
                ), f"{ious} vs {options}"

    def test__transform_empty_params(self, mocker):
        transform = transforms.RandomIoUCrop(sampler_options=[2.0])
        image = ta_tensors.Image(torch.rand(1, 3, 4, 4))
        bboxes = ta_tensors.BoundingBoxes(
            torch.tensor([[1, 1, 2, 2]]), format="XYXY", canvas_size=(4, 4)
        )
        label = torch.tensor([1])
        sample = [image, bboxes, label]
        # Let's mock transform._get_params to control the output:
        transform._get_params = mocker.MagicMock(return_value=[{}])
        output = transform(sample)
        torch.testing.assert_close(output, sample)

    def test_forward_assertion(self):
        transform = transforms.RandomIoUCrop()
        with pytest.raises(
            TypeError,
            match="requires input sample to contain tensor images and bounding boxes",
        ):
            transform(torch.tensor(0))

    @pytest.mark.parametrize("is_batch", [False, True])
    def test__transform(self, mocker, is_batch):
        transform = transforms.RandomIoUCrop(batch_transform=is_batch)

        size = (32, 24)
        image = make_image(size) if not is_batch else make_batch_images(size)
        bboxes = (
            make_bounding_boxes(format="XYXY", canvas_size=size, num_boxes=6)
            if not is_batch
            else make_batch_bounding_boxes(format="XYXY", canvas_size=size, num_boxes=6)
        )
        masks = (
            make_detection_masks(size, num_masks=6)
            if not is_batch
            else make_batch_detection_masks(size, num_masks=6)
        )

        sample = [image, bboxes, masks]

        is_within_crop_area = torch.tensor(
            [0, 1, 0, 1, 0, 1] * (1 if not is_batch else 2), dtype=torch.bool
        )

        params = dict(
            top=1, left=2, height=12, width=12, is_within_crop_area=is_within_crop_area
        )
        transform._get_params = mocker.MagicMock(
            return_value=[params] if not is_batch else [params, params]
        )
        output = transform(sample)

        # check number of bboxes vs number of labels:
        output_bboxes = output[1]
        assert isinstance(
            output_bboxes,
            ta_tensors.BoundingBoxes if not is_batch else ta_tensors.BatchBoundingBoxes,
        )
        assert (output_bboxes[~is_within_crop_area] == 0).all()

        output_masks = output[2]
        assert isinstance(
            output_masks, ta_tensors.Mask if not is_batch else ta_tensors.BatchMasks
        )


class TestScaleJitter:
    # Tests are light because this largely relies on the already tested `resize` kernels.

    INPUT_SIZE = (17, 11)
    TARGET_SIZE = (12, 13)

    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        check_transform(
            transforms.ScaleJitter(self.TARGET_SIZE),
            make_input(self.INPUT_SIZE, device=device),
        )

    @pytest.mark.parametrize("is_batch", [False, True])
    def test__get_params(self, is_batch):
        input_size = self.INPUT_SIZE
        target_size = self.TARGET_SIZE
        scale_range = (0.5, 1.5)

        transform = transforms.ScaleJitter(
            target_size=target_size, scale_range=scale_range
        )
        list_params = transform._get_params(
            [make_image(input_size) if not is_batch else make_batch_images(input_size)],
            num_chunks=1 if not is_batch else 2,
            chunks_indices=[torch.tensor([0]), torch.tensor([1])]
            if is_batch
            else [torch.tensor([0])],
        )

        for params in list_params:
            assert "size" in params
            size = params["size"]

            assert isinstance(size, tuple) and len(size) == 2
            height, width = size

            r_min = (
                min(target_size[1] / input_size[0], target_size[0] / input_size[1])
                * scale_range[0]
            )
            r_max = (
                min(target_size[1] / input_size[0], target_size[0] / input_size[1])
                * scale_range[1]
            )

            assert int(input_size[0] * r_min) <= height <= int(input_size[0] * r_max)
            assert int(input_size[1] * r_min) <= width <= int(input_size[1] * r_max)
