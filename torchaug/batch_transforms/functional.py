from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn.functional import conv2d
from torch.nn.functional import pad as torch_pad
from torchvision.transforms._functional_tensor import (_assert_image_tensor,
                                                       _cast_squeeze_in,
                                                       _cast_squeeze_out,
                                                       _hsv2rgb, _max_value,
                                                       _rgb2hsv,
                                                       convert_image_dtype,
                                                       rgb_to_grayscale)

from torchaug.batch_transforms._utils import (_assert_batch_channels,
                                              _assert_batch_images_tensor,
                                              get_batched_img_dimensions)
from torchaug.transforms._utils import (_assert_tensor,
                                        transfer_tensor_on_device)
from torchaug.utils import _log_api_usage_once


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
    kernel_size: list[int], sigma: list[float], dtype: torch.dtype, device: torch.device
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

    ratio = ratio.view(-1, *[1 for _ in range(img1.ndim - 1)])

    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def batch_adjust_brightness(
    imgs: Tensor, brightness_factor: float | Tensor, value_check: bool = False
) -> Tensor:
    """Adjust brightness of a batch of images.

    Args:
        imgs: Batch of images to be adjusted. It is expected to be in [B, ..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of dimensions.
        brightness_factor:  How much to adjust the brightness. Can be
            any 1 or B non-negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization or large batch size.

    Returns:
        Brightness adjusted batch of images.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batch_adjust_brightness)

    _assert_tensor(imgs)
    _assert_batch_images_tensor(imgs)
    _assert_batch_channels(imgs, [1, 3])

    batch_size = imgs.shape[0]

    if isinstance(brightness_factor, float):
        if brightness_factor < 0.0:
            raise ValueError(f"brightness_factor is not non-negative.")
        brightness_factor = torch.tensor(brightness_factor, device=imgs.device).expand(
            batch_size
        )
    elif isinstance(brightness_factor, Tensor):
        brightness_factor = transfer_tensor_on_device(
            brightness_factor, imgs.device, True
        )
        if value_check and not torch.all(torch.ge(brightness_factor, 0)):
            raise ValueError(f"brightness_factor is not non-negative.")
        if brightness_factor.numel() == 1:
            brightness_factor = brightness_factor.expand(batch_size)
        elif brightness_factor.numel() != batch_size:
            raise ValueError(
                f"brightness_factor tensor should contain 1 or B elements."
            )
    else:
        raise TypeError(f"brightness_factor should be a float or Tensor.")

    return _batch_blend(imgs, torch.zeros_like(imgs), brightness_factor)


def batch_adjust_contrast(
    imgs: Tensor, contrast_factor: Tensor, value_check: bool = False
) -> Tensor:
    """Adjust contrast of a batch of images.

    Args:
        imgs: Batch of images to be adjusted. It is expected to be in [B, ..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number dimensions.
        contrast_factor: How much to adjust the contrast. Can be any
            1 or B non-negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization.

    Returns:
        Contrast adjusted batch of images.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batch_adjust_contrast)

    _assert_tensor(imgs)
    _assert_batch_images_tensor(imgs)
    _assert_batch_channels(imgs, [1, 3])

    batch_size = imgs.shape[0]

    if isinstance(contrast_factor, float):
        if contrast_factor < 0.0:
            raise ValueError(f"contrast_factor is not non-negative.")
        contrast_factor = torch.tensor(contrast_factor, device=imgs.device).expand(
            batch_size
        )
    elif isinstance(contrast_factor, Tensor):
        contrast_factor = transfer_tensor_on_device(contrast_factor, imgs.device, True)
        if value_check and not torch.all(torch.ge(contrast_factor, 0)):
            raise ValueError(f"contrast_factor is not non-negative.")
        if contrast_factor.numel() == 1:
            contrast_factor = contrast_factor.expand(batch_size)
        elif contrast_factor.numel() != batch_size:
            raise ValueError(f"contrast_factor tensor should contain 1 or B elements.")
    else:
        raise TypeError(f"contrast_factor should be a float or Tensor.")

    c = get_batched_img_dimensions(imgs)[1]
    dtype = imgs.dtype if torch.is_floating_point(imgs) else torch.float32
    if c == 3:
        mean = torch.mean(
            rgb_to_grayscale(imgs).to(dtype), dim=(-3, -2, -1), keepdim=True
        )
    else:
        mean = torch.mean(imgs.to(dtype), dim=(-3, -2, -1), keepdim=True)

    return _batch_blend(imgs, mean, contrast_factor)


def batch_adjust_hue(
    imgs: Tensor, hue_factor: float | Tensor, value_check: bool = False
) -> Tensor:
    """Adjust hue of a batch of images.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        imgs: Batch of images to be adjusted. It is expected to be in [B, ..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of dimensions.
            Note: the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
        hue_factor:  How much to shift the hue channel. Can be 1 or B elements in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization or large batch size.

    Returns:
        Hue adjusted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batch_adjust_hue)

    _assert_tensor(imgs)
    _assert_image_tensor(imgs)
    _assert_batch_channels(imgs, [1, 3])

    if get_batched_img_dimensions(imgs)[1] == 1:  # Match PIL behaviour
        return imgs

    batch_size = imgs.shape[0]

    if isinstance(hue_factor, float):
        if not -0.5 <= hue_factor <= 0.5:
            raise ValueError(f"hue_factor is not between -0.5 and 0.5.")
        hue_factor = torch.tensor(hue_factor, device=imgs.device).expand(batch_size)
    elif isinstance(hue_factor, Tensor):
        hue_factor = transfer_tensor_on_device(hue_factor, imgs.device, True)
        if value_check and not torch.all(
            torch.logical_and(torch.ge(hue_factor, -0.5), torch.le(hue_factor, 0.5))
        ):
            raise ValueError(f"hue_factor is not between -0.5 and 0.5.")
        if hue_factor.numel() == 1:
            hue_factor = hue_factor.expand(batch_size)
        elif hue_factor.numel() != batch_size:
            raise ValueError(f"hue_factor tensor should contain 1 or B elements.")
    else:
        raise TypeError(f"hue_factor should be a float or Tensor.")

    orig_dtype = imgs.dtype
    imgs = convert_image_dtype(imgs, torch.float32)

    imgs = _rgb2hsv(imgs)
    h, s, v = imgs.unbind(dim=-3)

    hue_factor = hue_factor.view(-1, *[1 for _ in range(h.ndim - 1)])

    h = (h + hue_factor) % 1.0
    imgs = torch.stack((h, s, v), dim=-3)
    imgs_hue_adj = _hsv2rgb(imgs)

    return convert_image_dtype(imgs_hue_adj, orig_dtype)


def batch_adjust_saturation(
    imgs: Tensor, saturation_factor: float | Tensor, value_check: bool = False
) -> Tensor:
    """Adjust color saturation of a batch of images.

    Args:
        imgs: Batch of images to be adjusted. It is expected to be in [B, ..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of dimensions.
        saturation_factor:  How much to adjust the saturation. Can be 1 or B non-negative elements. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization or large batch size.

    Returns:
        Saturation adjusted batch of images.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batch_adjust_saturation)

    _assert_tensor(imgs)
    _assert_image_tensor(imgs)
    _assert_batch_channels(imgs, [1, 3])

    if get_batched_img_dimensions(imgs)[1] == 1:  # Match PIL behaviour
        return imgs

    batch_size = imgs.shape[0]

    if isinstance(saturation_factor, float):
        if saturation_factor < 0.0:
            raise ValueError(f"saturation_factor is not non-negative.")
        saturation_factor = torch.tensor(saturation_factor, device=imgs.device).expand(
            batch_size
        )
    elif isinstance(saturation_factor, Tensor):
        saturation_factor = transfer_tensor_on_device(
            saturation_factor, imgs.device, True
        )
        if value_check and not torch.all(torch.ge(saturation_factor, 0)):
            raise ValueError(f"saturation_factor is not non-negative.")
        if saturation_factor.numel() == 1:
            saturation_factor = saturation_factor.expand(batch_size)
        elif saturation_factor.numel() != batch_size:
            raise ValueError(
                f"saturation_factor tensor should contain 1 or B elements."
            )
    else:
        raise TypeError(f"saturation_factor should be a float or Tensor.")

    return _batch_blend(imgs, rgb_to_grayscale(imgs), saturation_factor)


def batch_gaussian_blur(
    imgs: Tensor,
    kernel_size: int | list[int],
    sigma: int | float | list[int] | list[float] | Tensor | None = None,
    value_check: bool = False,
) -> Tensor:
    """Performs Gaussian blurring on the batch of images by given kernel. It is expected to have [B, ..., C, H, W]
    shape, where ... means an arbitrary number of dimensions.

    Args:
        img: Image to be blurred
        kernel_size (sequence of ints or int): Gaussian kernel size. Can be a sequence of integers
            like ``(kx, ky)`` or a single integer for square kernels.

            .. note::
                In torchscript mode kernel_size as single int is not supported, use a sequence of
                length 1: ``[ksize, ]``.
        sigma: Gaussian kernel standard deviation.
            Can be a sequence of floats like ``(sigma_x, sigma_y)`` or a single float to define the
            same sigma in both X/Y directions. If None, then it is computed using
            ``kernel_size`` as ``sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8``.
            If Tensor it is expected to have [B] shape.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization or large batch size.

    Returns:
        Gaussian Blurred version of the image.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batch_gaussian_blur)

    _assert_tensor(imgs)
    _assert_batch_images_tensor(imgs)

    if not isinstance(kernel_size, (int, list, tuple)):
        raise TypeError(
            f"kernel_size should be int or a sequence of integers. Got {type(kernel_size)}."
        )
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

    batch_size = imgs.shape[0]
    if sigma is None:
        sigma_t = torch.tensor(
            [[ksize * 0.15 + 0.35 for ksize in kernel_size]], device=imgs.device
        ).expand((batch_size, 2))
    else:
        if isinstance(sigma, (list, tuple)):
            length = len(sigma)
            if length == 1:
                s = float(sigma[0])
                sigma_t = torch.tensor([[s, s]], device=imgs.device).expand(
                    (batch_size, 2)
                )
            elif length != 2:
                raise ValueError(
                    f"If sigma is a sequence, its length should be 2. Got {length}."
                )
            else:
                sigma_t = torch.tensor([list(sigma)], device=imgs.device).expand(
                    (batch_size, 2)
                )
        elif isinstance(sigma, (int, float)):
            s = float(sigma)
            sigma_t = torch.tensor([[s, s]], device=imgs.device).expand((batch_size, 2))
        elif isinstance(sigma, Tensor):
            sigma_t = transfer_tensor_on_device(sigma, imgs.device, non_blocking=True)

            dim_sigma = sigma_t.ndim
            if dim_sigma in [0, 1]:
                sigma_t = sigma_t.view(-1, 1)
            elif dim_sigma != 2:
                raise ValueError(
                    f"If sigma is a tensor, its dimension should be 0, 1 or 2. Got {dim_sigma}."
                )

            len_sigma, feature_dim_sigma = sigma_t.shape

            if len_sigma == 1:
                sigma_t = sigma_t.view(1, -1).expand((batch_size, feature_dim_sigma))
            elif len_sigma != batch_size:
                raise ValueError(
                    f"If sigma is a tensor, it should contain one or batch size elements. Got {len_sigma}."
                )

            if feature_dim_sigma == 1:
                sigma_t = sigma_t.view(-1, 1).expand((batch_size, 2))
            elif feature_dim_sigma != 2:
                raise ValueError(
                    f"If sigma is a tensor, it should have a feature dim of size one or two. Got {feature_dim_sigma}."
                )

        else:
            raise TypeError(
                f"sigma should be either int, float or sequence of floats or int or tensor. Got {type(sigma)}."
            )

    if (
        (isinstance(sigma, (float, int)) and sigma <= 0)
        or (isinstance(sigma, (list, tuple)) and any([s <= 0 for s in sigma]))
        or (
            isinstance(sigma, (Tensor))
            and value_check
            and not torch.all(torch.gt(sigma, 0))
        )
    ):
        raise ValueError(f"sigma should have positive values.")

    dtype = imgs.dtype if torch.is_floating_point(imgs) else torch.float32

    b, *rest_dims, h, w = imgs.shape
    imgs = imgs.reshape(b, math.prod(rest_dims), h, w)

    device = imgs.device
    kernel = _get_batch_gaussian_kernel2d(
        kernel_size, sigma_t, dtype=dtype, device=device
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


def batch_mixup(
    tensor_1: Tensor, tensor_2: Tensor, lam: float | Tensor, inplace: bool = False
) -> Tensor:
    """Mix two tensors with linear interpolation.

    The shape of the tensors is expected to be [B, ...] with ... any number of dimensions.
    The tensors shoud be floats.

    Args:
        tensor_1: First tensor.
        tensor_2: Second tensor.
        lam: Mixing coefficient. If float, same coefficient is applied on whole batch. If tensor,
            its expecting shape is [B, 1] or [B] or [1].
        inplace: Whether to perform the operation inplace.

    Returns:
        The mixed tensor.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batch_mixup)

    _assert_tensor(tensor_1)
    _assert_tensor(tensor_2)

    if not tensor_1.is_floating_point() or not tensor_2.is_floating_point():
        raise TypeError(
            f"Tensors should be float. Got {tensor_1.dtype} and {tensor_2.dtype}."
        )

    device = tensor_1.device
    batch_size = tensor_1.shape[0]

    if isinstance(lam, float):
        lam = torch.tensor(lam, device=device).expand(batch_size)
    elif isinstance(lam, Tensor):
        lam = transfer_tensor_on_device(lam, device, True)
        dim_lam = lam.ndim
        if dim_lam in [0, 1]:
            lam = lam.view(-1, 1)
        elif dim_lam == 2:
            if lam.shape[1] > 1:
                raise ValueError(
                    f"If lam is a two dimensional tensor, its second dimension should be 1. Got {lam.shape[1]}."
                )
        else:
            raise ValueError(
                f"If lam is a tensor, its dimension should be 0, 1 or 2. Got {dim_lam}."
            )

        len_lam = lam.shape[0]

        if len_lam == 1:
            lam = lam.view(1, 1).expand(batch_size, 1)
        elif len_lam != batch_size:
            raise ValueError(
                f"If lam is a tensor, it should contain one or batch size elements. Got {len_lam}."
            )
    else:
        raise TypeError(f"lam should be either float or tensor. Got {type(lam)}.")

    lam = lam.view(-1, *[1 for _ in range(tensor_1.ndim - 1)])

    tensor_1 = tensor_1 if inplace else tensor_1.clone()
    tensor_2 = tensor_2 if inplace else tensor_2.clone()
    return tensor_1.mul_(lam).add_(tensor_2.mul_(1 - lam))
