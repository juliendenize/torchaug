# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C
#
# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torchvision.transforms.v2.functional as TVF
from torchvision.transforms.v2.functional._meta import (
    _cxcywh_to_xyxy,
    _xywh_to_xyxy,
    _xyxy_to_cxcywh,
    _xyxy_to_xywh,
)

from torchaug import ta_tensors
from torchaug._utils import _log_api_usage_once

from ._utils._kernel import _get_kernel, _register_kernel_internal
from ._utils._tensor import is_pure_tensor


def get_dimensions(inpt: torch.Tensor) -> List[int]:
    if torch.jit.is_scripting():
        return get_dimensions_image(inpt)

    _log_api_usage_once(get_dimensions)

    kernel = _get_kernel(get_dimensions, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(get_dimensions, torch.Tensor)
@_register_kernel_internal(get_dimensions, ta_tensors.Image, ta_tensor_wrapper=False)
@_register_kernel_internal(get_dimensions, ta_tensors.BatchImages, ta_tensor_wrapper=False)
def get_dimensions_image(image: torch.Tensor) -> List[int]:
    return TVF.get_dimensions_image(image=image)


@_register_kernel_internal(get_dimensions, ta_tensors.Video, ta_tensor_wrapper=False)
@_register_kernel_internal(get_dimensions, ta_tensors.BatchVideos, ta_tensor_wrapper=False)
def get_dimensions_video(video: torch.Tensor) -> List[int]:
    return get_dimensions_image(image=video)


def get_num_channels(inpt: torch.Tensor) -> int:
    if torch.jit.is_scripting():
        return get_num_channels_image(inpt)

    _log_api_usage_once(get_num_channels)

    kernel = _get_kernel(get_num_channels, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(get_num_channels, torch.Tensor)
@_register_kernel_internal(get_num_channels, ta_tensors.Image, ta_tensor_wrapper=False)
@_register_kernel_internal(get_num_channels, ta_tensors.BatchImages, ta_tensor_wrapper=False)
def get_num_channels_image(image: torch.Tensor) -> int:
    return TVF.get_num_channels_image(image=image)


@_register_kernel_internal(get_num_channels, ta_tensors.Video, ta_tensor_wrapper=False)
@_register_kernel_internal(get_num_channels, ta_tensors.BatchVideos, ta_tensor_wrapper=False)
def get_num_channels_video(video: torch.Tensor) -> int:
    return get_num_channels_image(video)


def get_size(inpt: torch.Tensor) -> List[int]:
    if torch.jit.is_scripting():
        return get_size_image(inpt)

    _log_api_usage_once(get_size)

    kernel = _get_kernel(get_size, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(get_size, torch.Tensor)
@_register_kernel_internal(get_size, ta_tensors.Image, ta_tensor_wrapper=False)
@_register_kernel_internal(get_size, ta_tensors.BatchImages, ta_tensor_wrapper=False)
def get_size_image(image: torch.Tensor) -> List[int]:
    return TVF.get_size_image(image=image)


@_register_kernel_internal(get_size, ta_tensors.Video, ta_tensor_wrapper=False)
@_register_kernel_internal(get_size, ta_tensors.BatchVideos, ta_tensor_wrapper=False)
def get_size_video(video: torch.Tensor) -> List[int]:
    return get_size_image(image=video)


@_register_kernel_internal(get_size, ta_tensors.Mask, ta_tensor_wrapper=False)
@_register_kernel_internal(get_size, ta_tensors.BatchMasks, ta_tensor_wrapper=False)
def get_size_mask(mask: torch.Tensor) -> List[int]:
    return get_size_image(mask)


@_register_kernel_internal(get_size, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(get_size, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def get_size_bounding_boxes(bounding_box: ta_tensors.BoundingBoxes) -> List[int]:
    return list(bounding_box.canvas_size)


def get_num_frames(inpt: torch.Tensor) -> int:
    if torch.jit.is_scripting():
        return get_num_frames_video(inpt)

    _log_api_usage_once(get_num_frames)

    kernel = _get_kernel(get_num_frames, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(get_num_frames, torch.Tensor)
@_register_kernel_internal(get_num_frames, ta_tensors.Video, ta_tensor_wrapper=False)
@_register_kernel_internal(get_num_frames, ta_tensors.BatchVideos, ta_tensor_wrapper=False)
def get_num_frames_video(video: torch.Tensor) -> int:
    return TVF.get_num_frames_video(video=video)


def _convert_bounding_box_format(
    bounding_boxes: torch.Tensor,
    old_format: ta_tensors.BoundingBoxFormat,
    new_format: ta_tensors.BoundingBoxFormat,
    inplace: bool = False,
) -> torch.Tensor:
    if new_format == old_format:
        return bounding_boxes

    # TODO: Add _xywh_to_cxcywh and _cxcywh_to_xywh to improve performance
    if old_format == ta_tensors.BoundingBoxFormat.XYWH:
        bounding_boxes = _xywh_to_xyxy(bounding_boxes, inplace)
    elif old_format == ta_tensors.BoundingBoxFormat.CXCYWH:
        bounding_boxes = _cxcywh_to_xyxy(bounding_boxes, inplace)

    if new_format == ta_tensors.BoundingBoxFormat.XYWH:
        bounding_boxes = _xyxy_to_xywh(bounding_boxes, inplace)
    elif new_format == ta_tensors.BoundingBoxFormat.CXCYWH:
        bounding_boxes = _xyxy_to_cxcywh(bounding_boxes, inplace)

    return bounding_boxes


def convert_bounding_box_format(
    inpt: torch.Tensor,
    old_format: Optional[ta_tensors.BoundingBoxFormat] = None,
    new_format: Optional[ta_tensors.BoundingBoxFormat] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """See :func:`~torchaug.transforms.ConvertBoundingBoxFormat` for details."""
    # This being a kernel / functional hybrid, we need an option to pass `old_format` explicitly for pure tensor
    # inputs as well as extract it from `ta_tensors.BoundingBoxes` inputs. However, putting a default value on
    # `old_format` means we also need to put one on `new_format` to have syntactically correct Python. Here we
    # mimic the default error that would be thrown if `new_format` had no default value.
    if new_format is None:
        raise TypeError("convert_bounding_box_format() missing 1 required argument: 'new_format'")

    if not torch.jit.is_scripting():
        _log_api_usage_once(convert_bounding_box_format)

    if isinstance(old_format, str):
        old_format = ta_tensors.BoundingBoxFormat[old_format.upper()]  # type: ignore[misc]
    if isinstance(new_format, str):
        new_format = ta_tensors.BoundingBoxFormat[new_format.upper()]  # type: ignore[misc]

    if torch.jit.is_scripting() or is_pure_tensor(inpt):
        if old_format is None:
            raise ValueError("For pure tensor inputs, `old_format` has to be passed.")
        return _convert_bounding_box_format(inpt, old_format=old_format, new_format=new_format, inplace=inplace)
    elif isinstance(inpt, (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes)):
        if old_format is not None:
            raise ValueError("For bounding box ta_tensor inputs, `old_format` must not be passed.")
        output = _convert_bounding_box_format(
            inpt.as_subclass(torch.Tensor),
            old_format=inpt.format,
            new_format=new_format,
            inplace=inplace,
        )
        return ta_tensors.wrap(output, like=inpt, format=new_format)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or a bounding box ta_tensor, but got {type(inpt)} instead."
        )


def _clamp_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    canvas_size: Tuple[int, int],
) -> torch.Tensor:
    # TODO: Investigate if it makes sense from a performance perspective to have an implementation for every
    #  BoundingBoxFormat instead of converting back and forth
    in_dtype = bounding_boxes.dtype
    bounding_boxes = bounding_boxes.clone() if bounding_boxes.is_floating_point() else bounding_boxes.float()
    xyxy_boxes = convert_bounding_box_format(
        bounding_boxes,
        old_format=format,
        new_format=ta_tensors.BoundingBoxFormat.XYXY,
        inplace=True,
    )
    xyxy_boxes[..., 0::2].clamp_(min=0, max=canvas_size[1])
    xyxy_boxes[..., 1::2].clamp_(min=0, max=canvas_size[0])
    out_boxes = convert_bounding_box_format(
        xyxy_boxes,
        old_format=ta_tensors.BoundingBoxFormat.XYXY,
        new_format=format,
        inplace=True,
    )
    return out_boxes.to(in_dtype)


def clamp_bounding_boxes(
    inpt: torch.Tensor,
    format: Optional[ta_tensors.BoundingBoxFormat] = None,
    canvas_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """See :func:`~torchaug.transforms.ClampBoundingBoxes` for details."""
    if not torch.jit.is_scripting():
        _log_api_usage_once(clamp_bounding_boxes)

    if torch.jit.is_scripting() or is_pure_tensor(inpt):
        if format is None or canvas_size is None:
            raise ValueError("For pure tensor inputs, `format` and `canvas_size` have to be passed.")
        return _clamp_bounding_boxes(inpt, format=format, canvas_size=canvas_size)
    elif isinstance(inpt, (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes)):
        if format is not None or canvas_size is not None:
            raise ValueError("For bounding box ta_tensor inputs, `format` and `canvas_size` must not be passed.")
        output = _clamp_bounding_boxes(
            inpt.as_subclass(torch.Tensor),
            format=inpt.format,
            canvas_size=inpt.canvas_size,
        )
        return ta_tensors.wrap(output, like=inpt)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or a bounding box ta_tensor, but got {type(inpt)} instead."
        )
