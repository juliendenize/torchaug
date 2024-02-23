from __future__ import annotations

import math
from typing import List, Optional

import torch
import torchvision.transforms.v2.functional as TVF
from torch.nn.functional import conv2d, pad as torch_pad

from torchaug import ta_tensors
from torchaug.utils import _log_api_usage_once

from ._utils._kernel import _get_kernel, _register_kernel_internal
from ._utils._tensor import _transfer_tensor_on_device


def normalize(
    inpt: torch.Tensor,
    mean: List[float],
    std: List[float],
    inplace: bool = False,
) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.Normalize` for details."""
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
    mean: List[float] | torch.Tensor,
    std: List[float] | torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    if not image.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {image.dtype}.")

    if image.ndim < 3:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (..., C, H, W). Got {image.shape}."
        )

    if isinstance(std, (tuple, list)):
        divzero = not all(std)
    elif isinstance(std, (int, float)):
        divzero = std == 0
    else:
        divzero = False
    if divzero:
        raise ValueError("std evaluated to zero, leading to division by zero.")

    dtype = image.dtype
    device = image.device

    if not isinstance(mean, torch.Tensor):
        mean = _transfer_tensor_on_device(mean, device, True)
    else:
        mean = torch.as_tensor(mean, dtype=dtype, device=device)

    if isinstance(std, torch.Tensor):
        std = _transfer_tensor_on_device(std, device, True)
    else:
        std = torch.as_tensor(std, dtype=dtype, device=device)

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    if inplace:
        image = image.sub_(mean)
    else:
        image = image.sub(mean)

    return image.div_(std)


@_register_kernel_internal(normalize, ta_tensors.Video)
@_register_kernel_internal(normalize, ta_tensors.BatchVideos)
def normalize_video(
    video: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False
) -> torch.Tensor:
    return normalize_image(image=video, mean=mean, std=std, inplace=inplace)


def gaussian_blur(
    inpt: torch.Tensor,
    kernel_size: List[int],
    sigma: List[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.GaussianBlur` for details."""
    if torch.jit.is_scripting():
        return gaussian_blur_image(inpt, kernel_size=kernel_size, sigma=sigma)

    _log_api_usage_once(gaussian_blur)

    kernel = _get_kernel(gaussian_blur, type(inpt))
    return kernel(inpt, kernel_size=kernel_size, sigma=sigma)


def gaussian_blur_batch(
    inpt: torch.Tensor,
    kernel_size: List[int],
    sigma: List[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.GaussianBlur` for details."""
    if torch.jit.is_scripting():
        return gaussian_blur_batch_images(inpt, kernel_size=kernel_size, sigma=sigma)

    _log_api_usage_once(gaussian_blur_batch)

    kernel = _get_kernel(gaussian_blur_batch, type(inpt))
    return kernel(inpt, kernel_size=kernel_size, sigma=sigma)


def _get_gaussian_kernel1d(
    kernel_size: int, sigma: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    lim = (kernel_size - 1) / (2.0 * math.sqrt(2.0))
    x = torch.linspace(-lim, lim, steps=kernel_size, dtype=dtype, device=device)
    if sigma.numel() == 1:
        sigma = sigma.view(1)
    else:
        x = x.view(1, -1).expand(sigma.shape[0], -1)
    kernel1d = torch.softmax((x / sigma).pow_(2).neg_(), dim=0)
    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int],
    sigma: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    kernel1d_x = _get_gaussian_kernel1d(
        kernel_size[0], sigma[..., 0, None], dtype, device
    )[..., None]
    kernel1d_y = _get_gaussian_kernel1d(
        kernel_size[1], sigma[..., 1, None], dtype, device
    )[..., None]
    kernel2d = kernel1d_y * kernel1d_x.view(-1, 1, kernel_size[0])
    return kernel2d


@_register_kernel_internal(gaussian_blur, torch.Tensor)
@_register_kernel_internal(gaussian_blur, ta_tensors.Image)
@_register_kernel_internal(gaussian_blur, ta_tensors.BatchImages)
def gaussian_blur_image(
    image: torch.Tensor, kernel_size: List[int], sigma: List[float] | None = None
) -> torch.Tensor:
    # TODO: consider deprecating integers from sigma on the future
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    elif len(kernel_size) != 2:
        raise ValueError(
            f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}"
        )
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError(
                f"kernel_size should have odd and positive integers. Got {kernel_size}"
            )

    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]
    else:
        if isinstance(sigma, (list, tuple)):
            length = len(sigma)
            if length == 1:
                s = float(sigma[0])
                sigma = [s, s]
            elif length != 2:
                raise ValueError(
                    f"If sigma is a sequence, its length should be 2. Got {length}"
                )
        elif isinstance(sigma, (int, float)):
            s = float(sigma)
            sigma = [s, s]
        elif isinstance(sigma, torch.Tensor):
            sigma = _transfer_tensor_on_device(sigma, image.device, True)
        else:
            raise TypeError(
                f"sigma should be either float or sequence of floats. Got {type(sigma)}"
            )
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
    kernel = _get_gaussian_kernel2d(
        kernel_size, sigma, dtype=dtype if fp else torch.float32, device=image.device
    )
    kernel = kernel.expand(shape[-3], 1, kernel.shape[0], kernel.shape[1])

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
    sigma: List[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(sigma, (int, float, None)) or sigma.numel() == 2:
        return gaussian_blur_image(image=images, kernel_size=kernel_size, sigma=sigma)

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    elif len(kernel_size) != 2:
        raise ValueError(
            f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}"
        )
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError(
                f"kernel_size should have odd and positive integers. Got {kernel_size}"
            )

    if images.numel() == 0:
        return images

    sigma = _transfer_tensor_on_device(sigma, images.device, True)

    dtype = images.dtype
    shape = images.shape

    b, *rest_dims, h, w = shape
    images = images.reshape(b, math.prod(rest_dims), h, w)

    fp = torch.is_floating_point(images)
    kernel = _get_gaussian_kernel2d(
        kernel_size, sigma, dtype=dtype if fp else torch.float32, device=images.device
    )
    kernel = kernel[:, None, ...]
    kernel = kernel.expand(-1, images.shape[-3], kernel_size[0], kernel_size[1])
    kernel = kernel.reshape(-1, 1, kernel_size[0], kernel_size[1])

    images = images if fp else images.to(dtype=torch.float32)

    # padding = (left, right, top, bottom)
    padding = [
        kernel_size[0] // 2,
        kernel_size[0] // 2,
        kernel_size[1] // 2,
        kernel_size[1] // 2,
    ]

    output = torch_pad(images, padding, mode="reflect")
    output = conv2d(output, kernel, groups=shape[-3])

    output = output.reshape(b, *rest_dims, h, w)
    if not fp:
        output = output.round_().to(dtype=dtype)

    return output


@_register_kernel_internal(gaussian_blur_batch, ta_tensors.BatchVideos)
def gaussian_blur_batch_videos(
    videos: torch.Tensor,
    kernel_size: List[int],
    sigma: float | None | torch.Tensor = None,
) -> torch.Tensor:
    return gaussian_blur_batch_images(
        images=videos, kernel_size=kernel_size, sigma=sigma
    )


def to_dtype(
    inpt: torch.Tensor, dtype: torch.dtype = torch.float, scale: bool = False
) -> torch.Tensor:
    """See :func:`~torchvision.transforms.v2.ToDtype` for details."""
    if torch.jit.is_scripting():
        return to_dtype_image(inpt, dtype=dtype, scale=scale)

    _log_api_usage_once(to_dtype)

    kernel = _get_kernel(to_dtype, type(inpt))
    return kernel(inpt, dtype=dtype, scale=scale)


@_register_kernel_internal(to_dtype, torch.Tensor)
@_register_kernel_internal(to_dtype, ta_tensors.Image)
@_register_kernel_internal(to_dtype, ta_tensors.BatchImages)
def to_dtype_image(
    image: torch.Tensor, dtype: torch.dtype = torch.float, scale: bool = False
) -> torch.Tensor:
    return TVF.to_dtype_image(image=image, dtype=dtype, scale=scale)


@_register_kernel_internal(to_dtype, ta_tensors.Video)
@_register_kernel_internal(to_dtype, ta_tensors.BatchVideos)
def to_dtype_video(
    video: torch.Tensor, dtype: torch.dtype = torch.float, scale: bool = False
) -> torch.Tensor:
    return to_dtype_image(image=video, dtype=dtype, scale=scale)


@_register_kernel_internal(to_dtype, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(to_dtype, ta_tensors.Mask, ta_tensor_wrapper=False)
@_register_kernel_internal(
    to_dtype, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False
)
@_register_kernel_internal(to_dtype, ta_tensors.BatchMasks, ta_tensor_wrapper=False)
def _to_dtype_tensor_dispatch(
    inpt: torch.Tensor, dtype: torch.dtype, scale: bool = False
) -> torch.Tensor:
    # We don't need to unwrap and rewrap here, since TVTensor.to() preserves the type
    return inpt.to(dtype)
