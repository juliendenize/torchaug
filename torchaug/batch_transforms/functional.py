import math
from typing import List

import torch
from torch import Tensor
from torch.nn.functional import conv2d
from torch.nn.functional import pad as torch_pad
from torchvision.transforms._functional_tensor import (_assert_channels,
                                                       _assert_image_tensor,
                                                       _cast_squeeze_in,
                                                       _cast_squeeze_out,
                                                       _hsv2rgb, _max_value,
                                                       _rgb2hsv,
                                                       convert_image_dtype,
                                                       get_dimensions,
                                                       rgb_to_grayscale)


def _get_batch_gaussian_kernel1d(
    kernel_size: int, sigma: float, dtype: torch.dtype, device: torch.device
) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = (
        torch.linspace(
            -ksize_half, ksize_half, steps=kernel_size, dtype=dtype, device=device
        )
        .view(1, -1)
        .expand(sigma.shape[0], -1)
    )
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum(-1, keepdim=True)

    return kernel1d


def _get_batch_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_batch_gaussian_kernel1d(
        kernel_size[0], sigma[:, 0, None], dtype, device
    )[..., None]
    kernel1d_y = _get_batch_gaussian_kernel1d(
        kernel_size[1], sigma[:, 1, None], dtype, device
    )[..., None]
    kernel2d = kernel1d_y * kernel1d_x.view(-1, 1, kernel_size[0])
    return kernel2d


def _batch_blend(img1: Tensor, img2: Tensor, ratio: Tensor) -> Tensor:
    ratio = ratio.float()
    bound = _max_value(img1.dtype)

    while len(ratio.shape) != len(img1.shape):
        ratio = ratio[..., None]

    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def batch_adjust_brightness(img: Tensor, brightness_factor: Tensor) -> Tensor:
    _assert_image_tensor(img)

    _assert_channels(img, [1, 3])

    return _batch_blend(img, torch.zeros_like(img), brightness_factor)


def batch_adjust_contrast(img: Tensor, contrast_factor: Tensor) -> Tensor:
    _assert_image_tensor(img)

    _assert_channels(img, [3, 1])
    c = get_dimensions(img)[0]
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    if c == 3:
        mean = torch.mean(
            rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True
        )
    else:
        mean = torch.mean(img.to(dtype), dim=(-3, -2, -1), keepdim=True)

    return _batch_blend(img, mean, contrast_factor)


def batch_adjust_hue(img: Tensor, hue_factor: Tensor) -> Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError("Input img should be Tensor image")

    _assert_image_tensor(img)

    _assert_channels(img, [1, 3])
    if get_dimensions(img)[0] == 1:  # Match PIL behaviour
        return img

    orig_dtype = img.dtype
    img = convert_image_dtype(img, torch.float32)

    img = _rgb2hsv(img)
    h, s, v = img.unbind(dim=-3)

    while len(hue_factor.shape) != len(h.shape):
        hue_factor = hue_factor[..., None]

    h = (h + hue_factor) % 1.0
    img = torch.stack((h, s, v), dim=-3)
    img_hue_adj = _hsv2rgb(img)

    return convert_image_dtype(img_hue_adj, orig_dtype)


def batch_adjust_saturation(img: Tensor, saturation_factor: Tensor) -> Tensor:
    _assert_image_tensor(img)

    _assert_channels(img, [1, 3])

    if get_dimensions(img)[0] == 1:  # Match PIL behaviour
        return img

    return _batch_blend(img, rgb_to_grayscale(img), saturation_factor)


def batch_gaussian_blur(imgs: Tensor, kernel_size: List[int], sigma: Tensor) -> Tensor:
    if not (isinstance(imgs, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(imgs)}")

    _assert_image_tensor(imgs)

    dtype = imgs.dtype if torch.is_floating_point(imgs) else torch.float32

    b, *rest_dims, h, w = imgs.shape
    imgs = imgs.reshape(b, math.prod(rest_dims), h, w)

    device = imgs.device
    kernel = _get_batch_gaussian_kernel2d(
        kernel_size, sigma, dtype=dtype, device=device
    )
    kernel = kernel[:, None, ...]
    kernel = kernel.expand(-1, imgs.shape[-3], kernel_size[0], kernel_size[1])
    kernel = kernel.reshape(-1, 1, kernel_size[0], kernel_size[1])

    imgs, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(imgs, [kernel.dtype])

    # padding = (left, right, top, bottom)
    padding = [
        kernel_size[0] // 2,
        kernel_size[0] // 2,
        kernel_size[1] // 2,
        kernel_size[1] // 2,
    ]
    imgs = torch_pad(imgs, padding, mode="reflect")
    imgs = imgs.view(-1, kernel.size(0), imgs.size(-2), imgs.size(-1))
    imgs = conv2d(imgs, kernel, groups=imgs.shape[-3])

    imgs = _cast_squeeze_out(imgs, need_cast, need_squeeze, out_dtype)

    imgs = imgs.reshape(b, *rest_dims, h, w)

    return imgs
