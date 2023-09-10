from __future__ import annotations

import torch
from torch import Tensor
from torchvision.transforms._functional_tensor import (_max_value, _rgb2hsv,
                                                       convert_image_dtype,
                                                       rgb_to_grayscale)

from torchaug.batch_transforms._utils import (_assert_batch_channels,
                                              _assert_batch_images_tensor,
                                              get_batch_channels_height_width)
from torchaug.transforms._utils import (_assert_image_tensor, _assert_tensor,
                                        transfer_tensor_on_device)
from torchaug.transforms.functional._color import _hsv2rgb
from torchaug.utils import _log_api_usage_once


def _batch_blend(img1: Tensor, img2: Tensor, ratio: Tensor) -> Tensor:
    ratio = ratio.float()
    bound = _max_value(img1.dtype)

    ratio = ratio.view(-1, *[1 for _ in range(img1.ndim - 1)])

    output = img1.mul(ratio).add_(img2.mul(1.0 - ratio)).clamp_(0, bound)

    fp = img1.is_floating_point()

    return output if fp else output.to(img1.dtype)


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

    brightness_factor = brightness_factor.float()
    bound = _max_value(imgs.dtype)

    brightness_factor = brightness_factor.view(-1, *[1 for _ in range(imgs.ndim - 1)])

    output = imgs.mul(brightness_factor).clamp_(0, bound)
    return output if imgs.is_floating_point() else output.to(imgs.dtype)


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

    c = get_batch_channels_height_width(imgs)[1]
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
        Hue adjusted batch of images.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batch_adjust_hue)

    _assert_tensor(imgs)
    _assert_image_tensor(imgs)
    _assert_batch_channels(imgs, [1, 3])

    if get_batch_channels_height_width(imgs)[1] == 1:  # Match PIL behaviour
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

    if get_batch_channels_height_width(imgs)[1] == 1:  # Match PIL behaviour
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
