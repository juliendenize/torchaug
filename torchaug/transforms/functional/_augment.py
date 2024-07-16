# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C
#
# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

from __future__ import annotations

import torch
import torchvision.transforms.v2.functional as TVF
from torchvision.io import decode_jpeg, encode_jpeg

from torchaug import ta_tensors
from torchaug._utils import _assert_torchvision_installed, _log_api_usage_once

from ._utils._kernel import _get_kernel, _register_kernel_internal


def erase(
    inpt: torch.Tensor,
    i: int,
    j: int,
    h: int,
    w: int,
    v: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomErasing` for details."""
    if torch.jit.is_scripting():
        return erase_image(inpt, i=i, j=j, h=h, w=w, v=v, inplace=inplace)

    _log_api_usage_once(erase)

    kernel = _get_kernel(erase, type(inpt))
    return kernel(inpt, i=i, j=j, h=h, w=w, v=v, inplace=inplace)


@_register_kernel_internal(erase, torch.Tensor)
@_register_kernel_internal(erase, ta_tensors.Image)
@_register_kernel_internal(erase, ta_tensors.BatchImages)
def erase_image(
    image: torch.Tensor,
    i: int,
    j: int,
    h: int,
    w: int,
    v: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    return TVF.erase_image(image=image, i=i, j=j, h=h, w=w, v=v, inplace=inplace)


@_register_kernel_internal(erase, ta_tensors.Video)
@_register_kernel_internal(erase, ta_tensors.BatchVideos)
def erase_video(
    video: torch.Tensor,
    i: int,
    j: int,
    h: int,
    w: int,
    v: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    return erase_image(image=video, i=i, j=j, h=h, w=w, v=v, inplace=inplace)


def jpeg(image: torch.Tensor, quality: int) -> torch.Tensor:
    """See :class:`~torchaug.transforms.JPEG` for details."""
    if torch.jit.is_scripting():
        return jpeg_image(image, quality=quality)

    _log_api_usage_once(jpeg)

    kernel = _get_kernel(jpeg, type(image))
    return kernel(image, quality=quality)


@_register_kernel_internal(jpeg, torch.Tensor)
@_register_kernel_internal(jpeg, ta_tensors.Image)
@_register_kernel_internal(jpeg, ta_tensors.BatchImages)
def jpeg_image(image: torch.Tensor, quality: int) -> torch.Tensor:
    original_shape = image.shape
    image = image.view((-1,) + image.shape[-3:])

    if image.shape[0] == 0:  # degenerate
        return image.reshape(original_shape).clone()

    image_list = [decode_jpeg(encode_jpeg(image[i], quality=quality)) for i in range(image.shape[0])]
    image = torch.stack(image_list, dim=0).view(original_shape)
    return image


@_register_kernel_internal(jpeg, ta_tensors.Video)
@_register_kernel_internal(jpeg, ta_tensors.BatchVideos)
def jpeg_video(video: torch.Tensor, quality: int) -> torch.Tensor:
    return jpeg_image(video, quality=quality)
