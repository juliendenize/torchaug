from __future__ import annotations

from typing import List

import torch
import torchvision.transforms.v2.functional as TVF

from torchaug import ta_tensors
from torchaug.utils import _log_api_usage_once

from ._utils._kernel import _get_kernel, _register_kernel_internal


def get_dimensions(inpt: torch.Tensor) -> List[int]:
    if torch.jit.is_scripting():
        return get_dimensions_image(inpt)

    _log_api_usage_once(get_dimensions)

    kernel = _get_kernel(get_dimensions, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(get_dimensions, torch.Tensor)
@_register_kernel_internal(get_dimensions, ta_tensors.Image, ta_tensor_wrapper=False)
@_register_kernel_internal(
    get_dimensions, ta_tensors.BatchImages, ta_tensor_wrapper=False
)
def get_dimensions_image(image: torch.Tensor) -> List[int]:
    return TVF.get_dimensions_image(image=image)


@_register_kernel_internal(get_dimensions, ta_tensors.Video, ta_tensor_wrapper=False)
@_register_kernel_internal(
    get_dimensions, ta_tensors.BatchVideos, ta_tensor_wrapper=False
)
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
@_register_kernel_internal(
    get_num_channels, ta_tensors.BatchImages, ta_tensor_wrapper=False
)
def get_num_channels_image(image: torch.Tensor) -> int:
    return TVF.get_num_channels_image(image=image)


@_register_kernel_internal(get_num_channels, ta_tensors.Video, ta_tensor_wrapper=False)
@_register_kernel_internal(
    get_num_channels, ta_tensors.BatchVideos, ta_tensor_wrapper=False
)
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
@_register_kernel_internal(
    get_size, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False
)
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
@_register_kernel_internal(
    get_num_frames, ta_tensors.BatchVideos, ta_tensor_wrapper=False
)
def get_num_frames_video(video: torch.Tensor) -> int:
    return TVF.get_num_frames_video(video=video)


convert_bounding_box_format = TVF.convert_bounding_box_format
clamp_bounding_boxes = TVF.clamp_bounding_boxes
