from typing import List

import torch
from torch import Tensor
from torch.nn.functional import conv2d
from torch.nn.functional import pad as torch_pad
from torchvision.transforms._functional_tensor import (_assert_channels,
                                                       _assert_image_tensor,
                                                       _cast_squeeze_in,
                                                       _cast_squeeze_out,
                                                       _max_value, invert)
from torchvision.utils import _log_api_usage_once

from torchaug.transforms._utils import _assert_tensor, is_tensor_on_cpu


def _get_gaussian_kernel1d(
    kernel_size: int, sigma: float, dtype: torch.dtype, device: torch.device
) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(
        -ksize_half, ksize_half, steps=kernel_size, dtype=dtype, device=device
    )
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0], dtype, device)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1], dtype, device)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def gaussian_blur(
    img: Tensor,
    kernel_size: List[int],
    sigma: int | float | List[int] | List[float] | torch.Tensor | None = None,
    value_check: bool = False,
) -> Tensor:
    """Performs Gaussian blurring on the image by given kernel. If is expected to have [..., H, W] shape, where ...
    means an arbitrary number of leading dimensions.

    Args:
        img (Tensor): Image to be blurred
        kernel_size (sequence of ints or int): Gaussian kernel size. Can be a sequence of integers
            like ``(kx, ky)`` or a single integer for square kernels.

            .. note::
                In torchscript mode kernel_size as single int is not supported, use a sequence of
                length 1: ``[ksize, ]``.
        sigma (sequence of floats or int or Tensor, optional): Gaussian kernel standard deviation. Can be a
            sequence of floats like ``(sigma_x, sigma_y)`` or a single float to define the
            same sigma in both X/Y directions. If None, then it is computed using
            ``kernel_size`` as ``sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8``.
            Default, None.
        value_check (bool, optional): Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization. Default, False.

    Returns:
        Tensor: Gaussian Blurred version of the image.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(gaussian_blur)
    if not isinstance(kernel_size, (int, list, tuple)):
        raise TypeError(
            f"kernel_size should be int or a sequence of integers. Got {type(kernel_size)}."
        )

    _assert_tensor(img)
    _assert_image_tensor(img)

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if len(kernel_size) != 2:
        raise ValueError(
            f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}."
        )
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError(
                f"kernel_size should have odd and positive integers. Got {kernel_size}."
            )

    if sigma is None:
        sigma_t = torch.tensor(
            [ksize * 0.15 + 0.35 for ksize in kernel_size], device=img.device
        )
    else:
        if isinstance(sigma, (list, tuple)):
            length = len(sigma)
            if length == 1:
                s = float(sigma[0])
                sigma_t = torch.tensor([s, s], device=img.device)
            elif length != 2:
                raise ValueError(
                    f"If sigma is a sequence, its length should be 2. Got {length}."
                )
            else:
                sigma_t = torch.tensor(sigma, device=img.device)
        elif isinstance(sigma, (int, float)):
            s = float(sigma)
            sigma_t = torch.tensor([s, s], device=img.device)
        elif isinstance(sigma, torch.Tensor):
            sigma_t = sigma.to(device=img.device, non_blocking=True)

            dim_sigma = sigma_t.ndim + 1 if sigma_t.ndim == 0 else sigma_t.ndim
            len_sigma = 1 if sigma_t.ndim == 0 else sigma_t.shape[0]
            if dim_sigma != 1:
                raise ValueError(
                    f"If sigma is a tensor, its dimension should be 1. Got {dim_sigma}."
                )
            if len_sigma == 1:
                sigma_t = sigma_t.expand(2)
            elif len_sigma != 2:
                raise ValueError(
                    f"If sigma is a tensor of multiple values, its length should be 2. Got {len_sigma}."
                )
        else:
            raise TypeError(
                f"sigma should be either int, float or sequence of floats or int or tensor. Got {type(sigma)}."
            )

    if (
        (isinstance(sigma, (float, int)) and sigma <= 0)
        or (isinstance(sigma, (list, tuple)) and any([s <= 0 for s in sigma]))
        or (
            isinstance(sigma, (torch.Tensor))
            and (value_check or is_tensor_on_cpu(sigma))
            and not torch.all(torch.gt(sigma, 0))
        )
    ):
        raise ValueError(f"sigma should have positive values. Got {sigma}.")

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    device = img.device
    kernel = _get_gaussian_kernel2d(kernel_size, sigma_t, dtype=dtype, device=device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype])

    # padding = (left, right, top, bottom)
    padding = [
        kernel_size[0] // 2,
        kernel_size[0] // 2,
        kernel_size[1] // 2,
        kernel_size[1] // 2,
    ]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def normalize(
    tensor: Tensor,
    mean: List[float] | Tensor,
    std: List[float] | Tensor,
    inplace: bool = True,
    value_check: bool = False,
) -> Tensor:
    """Normalize a float tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence or Tensor): Sequence of means for each channel.
        std (sequence or Tensor): Sequence of standard deviations for each channel.
        inplace(bool, optional): Bool to make this operation inplace.
        value_check (bool, optional): Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization. Default, False.

    Returns:
        Tensor: Normalized Tensor image.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(normalize)

    _assert_tensor(tensor)
    _assert_image_tensor(tensor)

    if not tensor.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)

    if not inplace:
        tensor = tensor.clone()

    if (value_check or is_tensor_on_cpu(std)) and not torch.all(torch.gt(std, 0)):
        raise ValueError(
            f"std evaluated to zero after conversion to {dtype}, leading to division by zero."
        )

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    return tensor.sub_(mean).div_(std)


def solarize(
    img: Tensor, threshold: int | float | torch.Tensor, value_check: bool = False
) -> Tensor:
    """Solarize an RGB/grayscale image by inverting all pixel values above a threshold.

    Args:
        img (Tensor): Image to have its colors inverted. It is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".
        threshold (int, float, Tensor): All pixels equal or above this value are inverted.
        value_check (bool, optional): Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization. Default, False.
    Returns:
        Tensor: Solarized image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(solarize)

    _assert_tensor(img)
    _assert_image_tensor(img)

    if not isinstance(threshold, (int, float)) and (
        not isinstance(threshold, torch.Tensor) or threshold.numel() > 1
    ):
        raise TypeError("threshold should be a float or a tensor of one element.")

    if img.ndim < 3:
        raise TypeError(
            f"Input image tensor should have at least 3 dimensions, but found {img.ndim}."
        )

    _assert_channels(img, [1, 3])

    if (
        value_check
        or isinstance(threshold, (int, float) or is_tensor_on_cpu(threshold))
    ) and threshold > _max_value(img.dtype):
        raise ValueError("Threshold should be less than bound of img.")

    inverted_img = invert(img)
    return torch.where(img >= threshold, inverted_img, img)
