from typing import List

import torch
from torch import Tensor
from torch.nn.functional import conv2d
from torch.nn.functional import pad as torch_pad
from torchvision.transforms._functional_tensor import (_assert_channels,
                                                       _assert_image_tensor,
                                                       _cast_squeeze_in,
                                                       _cast_squeeze_out,
                                                       invert)


def solarize(img: Tensor, threshold: float) -> Tensor:
    _assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError(
            f"Input image tensor should have at least 3 dimensions, but found {img.ndim}"
        )

    _assert_channels(img, [1, 3])

    inverted_img = invert(img)
    return torch.where(img >= threshold, inverted_img, img)


def normalize(
    tensor: Tensor,
    mean: List[float] | Tensor,
    std: List[float] | Tensor,
    inplace: bool = True,
) -> Tensor:
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)

    if not inplace:
        tensor = tensor.clone()

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    return tensor.sub_(mean).div_(std)


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


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    _assert_image_tensor(img)

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    device = img.device
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=device)
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
