# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C
#
# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torchvision.transforms.v2.functional as TVF
from torch.nn.functional import conv2d
from torch.nn.functional import pad as torch_pad

from torchaug import ta_tensors
from torchaug._utils import _log_api_usage_once

from ._meta import _convert_bounding_box_format
from ._utils._kernel import _get_kernel, _register_kernel_internal
from ._utils._tensor import _transfer_tensor_on_device, is_pure_tensor


def normalize(
    inpt: torch.Tensor,
    mean: List[float],
    std: List[float],
    inplace: bool = False,
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.Normalize` for details."""
    if torch.jit.is_scripting():
        return normalize_image(inpt, mean=mean, std=std, inplace=inplace)

    _log_api_usage_once(normalize)

    kernel = _get_kernel(normalize, type(inpt))
    return kernel(inpt, mean=mean, std=std, inplace=inplace)


@_register_kernel_internal(normalize, torch.Tensor)
@_register_kernel_internal(normalize, ta_tensors.Image)
@_register_kernel_internal(normalize, ta_tensors.BatchImages)
def normalize_image(
    image: torch.Tensor,
    mean: List[float],
    std: List[float],
    inplace: bool = False,
) -> torch.Tensor:
    return TVF.normalize_image(image=image, mean=mean, std=std, inplace=inplace)


@_register_kernel_internal(normalize, ta_tensors.Video)
@_register_kernel_internal(normalize, ta_tensors.BatchVideos)
def normalize_video(video: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False) -> torch.Tensor:
    return normalize_image(image=video, mean=mean, std=std, inplace=inplace)


def gaussian_blur(
    inpt: torch.Tensor,
    kernel_size: List[int],
    sigma: Optional[List[float]] = None,
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomGaussianBlur` for details."""
    if torch.jit.is_scripting():
        return gaussian_blur_image(inpt, kernel_size=kernel_size, sigma=sigma)

    _log_api_usage_once(gaussian_blur)

    kernel = _get_kernel(gaussian_blur, type(inpt))
    return kernel(inpt, kernel_size=kernel_size, sigma=sigma)


def gaussian_blur_batch(
    inpt: torch.Tensor,
    kernel_size: List[int],
    sigma: Optional[torch.Tensor] = None,
    value_check: bool = False,
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomGaussianBlur` for details."""
    if torch.jit.is_scripting():
        return gaussian_blur_batch_images(inpt, kernel_size=kernel_size, sigma=sigma, value_check=value_check)

    _log_api_usage_once(gaussian_blur_batch)

    kernel = _get_kernel(gaussian_blur_batch, type(inpt))
    return kernel(inpt, kernel_size=kernel_size, sigma=sigma, value_check=value_check)


def _get_gaussian_kernel1d(
    kernel_size: int, sigma: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    lim = (kernel_size - 1) / (2.0 * math.sqrt(2.0))
    x = torch.linspace(-lim, lim, steps=kernel_size, dtype=dtype, device=device)

    sigma = sigma.view(-1, 1)
    x = x.view(1, -1).expand(sigma.shape[0], -1)

    kernel1d = torch.softmax((x.div(sigma)).pow_(2).neg_(), dim=1, dtype=dtype)
    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int],
    sigma: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[..., 0], dtype, device)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[..., 1], dtype, device)
    kernel2d = (kernel1d_y.view(-1, kernel_size[1], 1) * kernel1d_x.view(-1, 1, kernel_size[0])).view(
        -1, kernel_size[1], kernel_size[0]
    )
    return kernel2d


@_register_kernel_internal(gaussian_blur, torch.Tensor)
@_register_kernel_internal(gaussian_blur, ta_tensors.Image)
@_register_kernel_internal(gaussian_blur, ta_tensors.BatchImages)
def gaussian_blur_image(
    image: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None
) -> torch.Tensor:
    # TODO: consider deprecating integers from sigma on the future
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    elif len(kernel_size) != 2:
        raise ValueError(f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}")
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError(f"kernel_size should have odd and positive integers. Got {kernel_size}")

    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]
    else:
        if isinstance(sigma, (list, tuple)):
            length = len(sigma)
            if length == 1:
                s = float(sigma[0])
                sigma = [s, s]
            elif length != 2:
                raise ValueError(f"If sigma is a sequence, its length should be 2. Got {length}")
        elif isinstance(sigma, (int, float)):
            s = float(sigma)
            sigma = [s, s]
        else:
            raise TypeError(f"sigma should be either float or sequence of floats. Got {type(sigma)}")
    for s in sigma:
        if s <= 0.0:
            raise ValueError(f"sigma should have positive values. Got {sigma}")

    sigma = torch.as_tensor(sigma, device=image.device)

    if image.numel() == 0:
        return image

    dtype = image.dtype
    shape = image.shape
    ndim = image.ndim
    if ndim == 3:
        image = image.unsqueeze(dim=0)
    elif ndim > 4:
        image = image.reshape((-1,) + shape[-3:])

    fp = torch.is_floating_point(image)
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype if fp else torch.float32, device=image.device)

    kernel = kernel.expand(shape[-3], 1, kernel.shape[1], kernel.shape[2])

    output = image if fp else image.to(dtype=torch.float32)

    # padding = (left, right, top, bottom)
    padding = [
        kernel_size[0] // 2,
        kernel_size[0] // 2,
        kernel_size[1] // 2,
        kernel_size[1] // 2,
    ]
    output = torch_pad(output, padding, mode="reflect")
    output = conv2d(output, kernel, groups=shape[-3])

    if ndim == 3:
        output = output.squeeze(dim=0)
    elif ndim > 4:
        output = output.reshape(shape)

    if not fp:
        output = output.round_().to(dtype=dtype)

    return output


@_register_kernel_internal(gaussian_blur, ta_tensors.Video)
@_register_kernel_internal(gaussian_blur, ta_tensors.BatchVideos)
def gaussian_blur_video(
    video: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None
) -> torch.Tensor:
    return gaussian_blur_image(image=video, kernel_size=kernel_size, sigma=sigma)


@_register_kernel_internal(gaussian_blur_batch, torch.Tensor)
@_register_kernel_internal(gaussian_blur_batch, ta_tensors.BatchImages)
def gaussian_blur_batch_images(
    images: torch.Tensor,
    kernel_size: List[int],
    sigma: Optional[torch.Tensor] = None,
    value_check: bool = False,
) -> torch.Tensor:
    if not isinstance(sigma, torch.Tensor):
        return gaussian_blur_image(image=images, kernel_size=kernel_size, sigma=sigma)

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    elif len(kernel_size) != 2:
        raise ValueError(f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}")
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError(f"kernel_size should have odd and positive integers. Got {kernel_size}")

    if images.numel() == 0:
        return images

    sigma = _transfer_tensor_on_device(sigma, images.device, True)

    if sigma.device.type == "cpu" or value_check:
        if (sigma < 0).any():
            raise ValueError(f"sigma should have positive values. Got {sigma}")

    dtype = images.dtype
    shape = images.shape

    b, h, w = shape[0], shape[-2], shape[-1]
    images = images.reshape(b, -1, h, w)

    if sigma.ndim == 0 or sigma.ndim == 1 and (sigma.numel() == 1 or sigma.numel() > 2):
        sigma = sigma.view(-1, 1)
        sigma = sigma.expand(b, 2)
    elif sigma.ndim in [1, 2] and sigma.numel() in [1, 2]:
        sigma = sigma.view(1, -1)
        sigma = sigma.expand(b, 2)
    elif sigma.ndim == 2 and sigma.shape[0] != b:
        raise ValueError(
            f"sigma should have one element or the same length as the batch size. Got {sigma.shape[0]} and {b}"
        )
    elif sigma.ndim > 2:
        raise ValueError(f"sigma should have 1 or 2 dimensions. Got {sigma.ndim}")
    fp = torch.is_floating_point(images)
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype if fp else torch.float32, device=images.device)

    kernel = kernel[:, None, ...]
    kernel = kernel.expand(-1, images.shape[-3], kernel_size[1], kernel_size[0])
    kernel = kernel.reshape(-1, 1, kernel_size[1], kernel_size[0])

    images = images if fp else images.to(dtype=torch.float32)

    # padding = (left, right, top, bottom)
    padding = [
        kernel_size[0] // 2,
        kernel_size[0] // 2,
        kernel_size[1] // 2,
        kernel_size[1] // 2,
    ]

    output = torch_pad(images, padding, mode="reflect")
    output = output.view(-1, kernel.shape[0], output.shape[-2], output.shape[-1])
    output = conv2d(output, kernel, groups=output.shape[-3])

    output = output.reshape(shape)
    if not fp:
        output = output.round_().to(dtype=dtype)

    return output


@_register_kernel_internal(gaussian_blur_batch, ta_tensors.BatchVideos)
def gaussian_blur_batch_videos(
    videos: torch.Tensor,
    kernel_size: List[int],
    sigma: Optional[torch.Tensor] = None,
    value_check: bool = False,
) -> torch.Tensor:
    return gaussian_blur_batch_images(images=videos, kernel_size=kernel_size, sigma=sigma, value_check=value_check)


def sanitize_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: Optional[ta_tensors.BoundingBoxFormat] = None,
    canvas_size: Optional[Tuple[int, int]] = None,
    min_size: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove degenerate/invalid bounding boxes and return the corresponding indexing mask.

    This removes bounding boxes that:
    - are below a given ``min_size``: by default this also removes degenerate boxes that have e.g. X2 <= X1.
    - have any coordinate outside of their corresponding image. You may want to
      call :func:`~torchaug.transforms.functional.clamp_bounding_boxes` first to avoid undesired removals.
    It is recommended to call it at the end of a pipeline, before passing the
    input to the models. It is critical to call this transform if
    :class:`~torchaug.transforms.RandomIoUCrop` was called.
    If you want to be extra careful, you may call it after all transforms that
    may modify bounding boxes but once at the end should be enough in most
    cases.

    Args:
        bounding_boxes: The bounding boxes to be sanitized.
        format: The format of the bounding boxes.
            Must be left to none if ``bounding_boxes`` is a :class:`~torchaug.ta_tensors.BoundingBoxes` object.
        canvas_size: The canvas_size of the bounding boxes
            (size of the corresponding image/video).
            Must be left to none if ``bounding_boxes`` is a :class:`~torchaug.ta_tensors.BoundingBoxes` object.
        min_size: The size below which bounding boxes are removed.

    Returns:
        The subset of valid bounding boxes, and the corresponding indexing mask.
        The mask can then be used to subset other tensors (e.g. labels) that are associated with the bounding boxes.
    """
    if torch.jit.is_scripting() or is_pure_tensor(bounding_boxes):
        if format is None or canvas_size is None:
            raise ValueError(
                "format and canvas_size cannot be None if bounding_boxes is a pure tensor. "
                f"Got format={format} and canvas_size={canvas_size}."
                "Set those to appropriate values or pass bounding_boxes as a tv_tensors.BoundingBoxes object."
            )
        if isinstance(format, str):
            format = ta_tensors.BoundingBoxFormat[format.upper()]
        valid = _get_sanitize_bounding_boxes_mask(
            bounding_boxes, format=format, canvas_size=canvas_size, min_size=min_size
        )
        bounding_boxes = bounding_boxes[valid]
    else:
        if not isinstance(bounding_boxes, (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes)):
            raise ValueError(
                "bounding_boxes must be a ta_tensors.BoundingBoxes or "
                "ta_tensors.BatchBoundingBoxes instance or a pure tensor."
            )
        if format is not None or canvas_size is not None:
            raise ValueError(
                "format and canvas_size must be None when bounding_boxes is a "
                "ta_tensors.BoundingBoxes or ta_tensors.BatchBoundingBoxes instance. "
                f"Got format={format} and canvas_size={canvas_size}. "
                "Leave those to None or pass bouding_boxes as a pure tensor."
            )
        valid = _get_sanitize_bounding_boxes_mask(
            bounding_boxes, format=bounding_boxes.format, canvas_size=bounding_boxes.canvas_size, min_size=min_size
        )
        bounding_boxes = ta_tensors.wrap(bounding_boxes[valid], like=bounding_boxes)

    return bounding_boxes, valid


def _get_sanitize_bounding_boxes_mask(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    canvas_size: Tuple[int, int],
    min_size: float = 1.0,
) -> torch.Tensor:
    bounding_boxes = _convert_bounding_box_format(
        bounding_boxes, new_format=ta_tensors.BoundingBoxFormat.XYXY, old_format=format
    )

    image_h, image_w = canvas_size
    ws, hs = bounding_boxes[:, 2] - bounding_boxes[:, 0], bounding_boxes[:, 3] - bounding_boxes[:, 1]
    valid = (ws >= min_size) & (hs >= min_size) & (bounding_boxes >= 0).all(dim=-1)
    # TODO: Do we really need to check for out of bounds here? All
    # transforms should be clamping anyway, so this should never happen?
    image_h, image_w = canvas_size
    valid &= (bounding_boxes[:, 0] <= image_w) & (bounding_boxes[:, 2] <= image_w)
    valid &= (bounding_boxes[:, 1] <= image_h) & (bounding_boxes[:, 3] <= image_h)
    return valid


def to_dtype(inpt: torch.Tensor, dtype: torch.dtype = torch.float, scale: bool = False) -> torch.Tensor:
    """See :func:`~torchaug.transforms.ToDtype` for details."""
    if torch.jit.is_scripting():
        return to_dtype_image(inpt, dtype=dtype, scale=scale)

    _log_api_usage_once(to_dtype)

    kernel = _get_kernel(to_dtype, type(inpt))
    return kernel(inpt, dtype=dtype, scale=scale)


@_register_kernel_internal(to_dtype, torch.Tensor)
@_register_kernel_internal(to_dtype, ta_tensors.Image)
@_register_kernel_internal(to_dtype, ta_tensors.BatchImages)
def to_dtype_image(image: torch.Tensor, dtype: torch.dtype = torch.float, scale: bool = False) -> torch.Tensor:
    return TVF.to_dtype_image(image=image, dtype=dtype, scale=scale)


@_register_kernel_internal(to_dtype, ta_tensors.Video)
@_register_kernel_internal(to_dtype, ta_tensors.BatchVideos)
def to_dtype_video(video: torch.Tensor, dtype: torch.dtype = torch.float, scale: bool = False) -> torch.Tensor:
    return to_dtype_image(image=video, dtype=dtype, scale=scale)


@_register_kernel_internal(to_dtype, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(to_dtype, ta_tensors.Mask, ta_tensor_wrapper=False)
@_register_kernel_internal(to_dtype, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(to_dtype, ta_tensors.BatchMasks, ta_tensor_wrapper=False)
def _to_dtype_tensor_dispatch(inpt: torch.Tensor, dtype: torch.dtype, scale: bool = False) -> torch.Tensor:
    # We don't need to unwrap and rewrap here, since TVTensor.to() preserves the type
    return inpt.to(dtype)
