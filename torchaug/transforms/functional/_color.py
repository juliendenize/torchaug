from __future__ import annotations

from typing import List, Union

import torch
import torchvision.transforms.v2.functional as TVF
from torchvision.transforms.v2.functional._color import _hsv_to_rgb, _rgb_to_hsv

from torchaug import ta_tensors
from torchaug._utils import _log_api_usage_once

from ._misc import to_dtype_image
from ._utils._kernel import _get_kernel, _register_kernel_internal
from ._utils._tensor import _get_batch_factor, _max_value


def _rgb_to_grayscale_image(
    image: torch.Tensor, num_output_channels: int = 1, preserve_dtype: bool = True
) -> torch.Tensor:
    # TODO: Maybe move the validation that num_output_channels is 1 or 3 to this function instead of callers.
    if image.shape[-3] == 1 and num_output_channels == 1:
        return image.clone()
    if image.shape[-3] == 1 and num_output_channels == 3:
        s = [1] * len(image.shape)
        s[-3] = 3
        return image.repeat(s)
    r, g, b = image.unbind(dim=-3)
    l_img = r.mul(0.2989).add_(g, alpha=0.587).add_(b, alpha=0.114)
    l_img = l_img.unsqueeze(dim=-3)
    if preserve_dtype:
        l_img = l_img.to(image.dtype)
    if num_output_channels == 3:
        l_img = l_img.expand(image.shape)
    return l_img


def rgb_to_grayscale(inpt: torch.Tensor, num_output_channels: int = 1) -> torch.Tensor:
    """See :class:`~torchaug.transforms.Grayscale` for details."""
    if torch.jit.is_scripting():
        return rgb_to_grayscale_image(inpt, num_output_channels=num_output_channels)

    _log_api_usage_once(rgb_to_grayscale)

    kernel = _get_kernel(rgb_to_grayscale, type(inpt))
    return kernel(inpt, num_output_channels=num_output_channels)


@_register_kernel_internal(rgb_to_grayscale, torch.Tensor)
@_register_kernel_internal(rgb_to_grayscale, ta_tensors.Image)
@_register_kernel_internal(rgb_to_grayscale, ta_tensors.BatchImages)
def rgb_to_grayscale_image(image: torch.Tensor, num_output_channels: int = 1) -> torch.Tensor:
    return _rgb_to_grayscale_image(image=image, num_output_channels=num_output_channels, preserve_dtype=True)


@_register_kernel_internal(rgb_to_grayscale, ta_tensors.Video)
@_register_kernel_internal(rgb_to_grayscale, ta_tensors.BatchVideos)
def rgb_to_grayscale_video(video: torch.Tensor, num_output_channels: int = 1) -> torch.Tensor:
    return rgb_to_grayscale_image(image=video, num_output_channels=num_output_channels)


def _batch_blend(images1: torch.Tensor, images2: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    ratio = ratio.float()
    fp = images1.is_floating_point()
    bound = _max_value(images1.dtype)
    while ratio.ndim < images1.ndim:
        ratio = ratio.unsqueeze(-1)
    output = images1.mul(ratio).add_(images2.mul(1.0 - ratio)).clamp_(0, bound)
    return output if fp else output.to(images1.dtype)


def adjust_brightness(inpt: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    """Adjust brightness."""
    if torch.jit.is_scripting():
        return adjust_brightness_image(inpt, brightness_factor=brightness_factor)

    _log_api_usage_once(adjust_brightness)

    kernel = _get_kernel(adjust_brightness, type(inpt))
    return kernel(inpt, brightness_factor=brightness_factor)


def adjust_brightness_batch(
    inpt: torch.Tensor,
    brightness_factor: Union[float, torch.Tensor],
    value_check: bool = False,
) -> torch.Tensor:
    """Adjust brightness."""
    if torch.jit.is_scripting():
        return adjust_brightness_batch_images(inpt, brightness_factor=brightness_factor, value_check=value_check)

    _log_api_usage_once(adjust_brightness_batch)

    kernel = _get_kernel(adjust_brightness_batch, type(inpt))
    return kernel(inpt, brightness_factor=brightness_factor, value_check=value_check)


@_register_kernel_internal(adjust_brightness, torch.Tensor)
@_register_kernel_internal(adjust_brightness, ta_tensors.Image)
@_register_kernel_internal(adjust_brightness, ta_tensors.BatchImages)
def adjust_brightness_image(image: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    return TVF.adjust_brightness_image(image=image, brightness_factor=brightness_factor)


@_register_kernel_internal(adjust_brightness, ta_tensors.Video)
@_register_kernel_internal(adjust_brightness, ta_tensors.BatchVideos)
def adjust_brightness_video(video: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    return adjust_brightness_image(image=video, brightness_factor=brightness_factor)


@_register_kernel_internal(adjust_brightness_batch, torch.Tensor)
@_register_kernel_internal(adjust_brightness_batch, ta_tensors.BatchImages)
def adjust_brightness_batch_images(
    images: torch.Tensor,
    brightness_factor: Union[float, torch.Tensor],
    value_check: bool = False,
) -> torch.Tensor:
    c = images.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are 1 or 3, but found {c}")

    brightness_factor = _get_batch_factor(brightness_factor, images.shape[0], images.device, None, value_check)

    fp = images.is_floating_point()
    brightness_factor = brightness_factor.float()
    bound = _max_value(images.dtype)

    while brightness_factor.ndim < images.ndim:
        brightness_factor = brightness_factor.unsqueeze(-1)

    output = images.mul(brightness_factor).clamp_(0, bound)

    return output if fp else output.to(images.dtype)


@_register_kernel_internal(adjust_brightness_batch, ta_tensors.BatchVideos)
def adjust_brightness_batch_videos(
    videos: torch.Tensor,
    brightness_factor: Union[float, torch.Tensor],
    value_check: bool = False,
):
    return adjust_brightness_batch_images(images=videos, brightness_factor=brightness_factor, value_check=value_check)


def adjust_saturation(inpt: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    """Adjust saturation."""
    if torch.jit.is_scripting():
        return adjust_saturation_image(inpt, saturation_factor=saturation_factor)

    _log_api_usage_once(adjust_saturation)

    kernel = _get_kernel(adjust_saturation, type(inpt))
    return kernel(inpt, saturation_factor=saturation_factor)


def adjust_saturation_batch(
    inpt: torch.Tensor,
    saturation_factor: Union[float, torch.Tensor],
    value_check: bool = False,
) -> torch.Tensor:
    """Adjust saturation."""
    if torch.jit.is_scripting():
        return adjust_saturation_batch_images(inpt, saturation_factor=saturation_factor, value_check=value_check)

    _log_api_usage_once(adjust_saturation_batch)

    kernel = _get_kernel(adjust_saturation_batch, type(inpt))
    return kernel(inpt, saturation_factor=saturation_factor, value_check=value_check)


@_register_kernel_internal(adjust_saturation, torch.Tensor)
@_register_kernel_internal(adjust_saturation, ta_tensors.Image)
@_register_kernel_internal(adjust_saturation, ta_tensors.BatchImages)
def adjust_saturation_image(image: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    return TVF.adjust_saturation_image(image=image, saturation_factor=saturation_factor)


@_register_kernel_internal(adjust_saturation, ta_tensors.Video)
@_register_kernel_internal(adjust_saturation, ta_tensors.BatchVideos)
def adjust_saturation_video(video: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    return adjust_saturation_image(image=video, saturation_factor=saturation_factor)


@_register_kernel_internal(adjust_saturation_batch, torch.Tensor)
@_register_kernel_internal(adjust_saturation_batch, ta_tensors.BatchImages)
def adjust_saturation_batch_images(
    images: torch.Tensor,
    saturation_factor: Union[float, torch.Tensor],
    value_check: bool = False,
) -> torch.Tensor:
    c = images.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are 1 or 3, but found {c}")

    if c == 1:  # Match PIL behaviour
        return images

    saturation_factor = _get_batch_factor(saturation_factor, images.shape[0], images.device, None, value_check)

    return _batch_blend(images, rgb_to_grayscale(images), saturation_factor)


@_register_kernel_internal(adjust_saturation_batch, ta_tensors.BatchVideos)
def adjust_saturation_batch_videos(
    video: torch.Tensor,
    saturation_factor: Union[float, torch.Tensor],
    value_check: bool = False,
) -> torch.Tensor:
    return adjust_saturation_batch_images(video, saturation_factor=saturation_factor, value_check=value_check)


def adjust_contrast(inpt: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    """Adjust contrast."""
    if torch.jit.is_scripting():
        return adjust_contrast_image(inpt, contrast_factor=contrast_factor)

    _log_api_usage_once(adjust_contrast)

    kernel = _get_kernel(adjust_contrast, type(inpt))
    return kernel(inpt, contrast_factor=contrast_factor)


def adjust_contrast_batch(
    inpt: torch.Tensor, contrast_factor: Union[float, torch.Tensor], value_check: bool = False
) -> torch.Tensor:
    """Adjust contrast."""
    if torch.jit.is_scripting():
        return adjust_contrast_batch_images(inpt, contrast_factor=contrast_factor, value_check=value_check)

    _log_api_usage_once(adjust_contrast_batch)

    kernel = _get_kernel(adjust_contrast_batch, type(inpt))
    return kernel(inpt, contrast_factor=contrast_factor, value_check=value_check)


@_register_kernel_internal(adjust_contrast, torch.Tensor)
@_register_kernel_internal(adjust_contrast, ta_tensors.Image)
@_register_kernel_internal(adjust_contrast, ta_tensors.BatchImages)
def adjust_contrast_image(image: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    return TVF.adjust_contrast_image(image=image, contrast_factor=contrast_factor)


@_register_kernel_internal(adjust_contrast, ta_tensors.Video)
@_register_kernel_internal(adjust_contrast, ta_tensors.BatchVideos)
def adjust_contrast_video(video: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    return adjust_contrast_image(image=video, contrast_factor=contrast_factor)


@_register_kernel_internal(adjust_contrast_batch, torch.Tensor)
@_register_kernel_internal(adjust_contrast_batch, ta_tensors.BatchImages)
def adjust_contrast_batch_images(
    images: torch.Tensor,
    contrast_factor: Union[float, torch.Tensor],
    value_check: bool = False,
) -> torch.Tensor:
    c = images.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f"Input image tensor permitted channel values are 1 or 3, but found {c}")

    contrast_factor = _get_batch_factor(contrast_factor, images.shape[0], images.device, None, value_check)

    fp = images.is_floating_point()
    if c == 3:
        grayscale_images = _rgb_to_grayscale_image(images, num_output_channels=1, preserve_dtype=False)
        if not fp:
            grayscale_images = grayscale_images.floor_()
    else:
        grayscale_images = images if fp else images.to(torch.float32)
    mean = torch.mean(grayscale_images, dim=(-3, -2, -1), keepdim=True)

    return _batch_blend(images, mean, contrast_factor)


@_register_kernel_internal(adjust_contrast_batch, ta_tensors.BatchVideos)
def adjust_contrast_batch_videos(videos: torch.Tensor, contrast_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    return adjust_contrast_batch_images(videos, contrast_factor=contrast_factor)


def adjust_sharpness(inpt: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    """See :class:`torchvision.transforms.RandomAdjustSharpness`."""
    if torch.jit.is_scripting():
        return adjust_sharpness_image(inpt, sharpness_factor=sharpness_factor)

    _log_api_usage_once(adjust_sharpness)

    kernel = _get_kernel(adjust_sharpness, type(inpt))
    return kernel(inpt, sharpness_factor=sharpness_factor)


@_register_kernel_internal(adjust_sharpness, torch.Tensor)
@_register_kernel_internal(adjust_sharpness, ta_tensors.Image)
@_register_kernel_internal(adjust_sharpness, ta_tensors.BatchImages)
def adjust_sharpness_image(image: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    return TVF.adjust_sharpness_image(image=image, sharpness_factor=sharpness_factor)


@_register_kernel_internal(adjust_sharpness, ta_tensors.Video)
@_register_kernel_internal(adjust_sharpness, ta_tensors.BatchVideos)
def adjust_sharpness_video(video: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    return TVF.adjust_sharpness_image(image=video, sharpness_factor=sharpness_factor)


def adjust_hue(inpt: torch.Tensor, hue_factor: float) -> torch.Tensor:
    """Adjust hue."""
    if torch.jit.is_scripting():
        return adjust_hue_image(inpt, hue_factor=hue_factor)

    _log_api_usage_once(adjust_hue)

    kernel = _get_kernel(adjust_hue, type(inpt))
    return kernel(inpt, hue_factor=hue_factor)


def adjust_hue_batch(
    inpt: torch.Tensor, hue_factor: Union[float, torch.Tensor], value_check: bool = False
) -> torch.Tensor:
    """Adjust hue."""
    if torch.jit.is_scripting():
        return adjust_hue_batch_images(inpt, hue_factor=hue_factor, value_check=value_check)

    _log_api_usage_once(adjust_hue_batch)

    kernel = _get_kernel(adjust_hue_batch, type(inpt))
    return kernel(inpt, hue_factor=hue_factor, value_check=value_check)


@_register_kernel_internal(adjust_hue, torch.Tensor)
@_register_kernel_internal(adjust_hue, ta_tensors.Image)
@_register_kernel_internal(adjust_hue, ta_tensors.BatchImages)
def adjust_hue_image(image: torch.Tensor, hue_factor: float) -> torch.Tensor:
    return TVF.adjust_hue_image(image=image, hue_factor=hue_factor)


@_register_kernel_internal(adjust_hue, ta_tensors.Video)
@_register_kernel_internal(adjust_hue, ta_tensors.BatchVideos)
def adjust_hue_video(video: torch.Tensor, hue_factor: float) -> torch.Tensor:
    return adjust_hue_image(image=video, hue_factor=hue_factor)


@_register_kernel_internal(adjust_hue_batch, torch.Tensor)
@_register_kernel_internal(adjust_hue_batch, ta_tensors.BatchImages)
def adjust_hue_batch_images(
    images: torch.Tensor, hue_factor: Union[float, torch.Tensor], value_check: bool = False
) -> torch.Tensor:
    c = images.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f"Input images tensor permitted channel values are 1 or 3, but found {c}")

    if c == 1:  # Match PIL behaviour
        return images

    if images.numel() == 0:
        # exit earlier on empty images
        return images

    hue_factor = _get_batch_factor(
        hue_factor,
        images.shape[0],
        images.device,
        None,
        value_check,
        min_value=-0.5,
        max_value=0.5,
    )

    orig_dtype = images.dtype
    images = to_dtype_image(images, torch.float32, scale=True)

    images = _rgb_to_hsv(images)
    h, s, v = images.unbind(dim=-3)

    while hue_factor.ndim < (images.ndim - 1):
        hue_factor = hue_factor.unsqueeze(-1)

    h = (h + hue_factor) % 1.0
    images = torch.stack((h, s, v), dim=-3)
    images_hue_adj = _hsv_to_rgb(images)

    return to_dtype_image(images_hue_adj, orig_dtype, scale=True)


@_register_kernel_internal(adjust_hue_batch, ta_tensors.BatchVideos)
def adjust_hue_batch_videos(
    videos: torch.Tensor, hue_factor: Union[float, torch.Tensor], value_check: bool = False
) -> torch.Tensor:
    return adjust_hue_batch_images(images=videos, hue_factor=hue_factor, value_check=value_check)


def adjust_gamma(
    inpt: torch.Tensor,
    gamma: float,
    gain: float = 1,
) -> torch.Tensor:
    """Adjust gamma."""
    if torch.jit.is_scripting():
        return adjust_gamma_image(inpt, gamma=gamma, gain=gain)

    _log_api_usage_once(adjust_gamma)

    kernel = _get_kernel(adjust_gamma, type(inpt))
    return kernel(inpt, gamma=gamma, gain=gain)


@_register_kernel_internal(adjust_gamma, torch.Tensor)
@_register_kernel_internal(adjust_gamma, ta_tensors.Image)
@_register_kernel_internal(adjust_gamma, ta_tensors.BatchImages)
def adjust_gamma_image(image: torch.Tensor, gamma: float, gain: float = 1.0) -> torch.Tensor:
    return TVF.adjust_gamma_image(image=image, gamma=gamma, gain=gain)


@_register_kernel_internal(adjust_gamma, ta_tensors.Video)
@_register_kernel_internal(adjust_gamma, ta_tensors.BatchVideos)
def adjust_gamma_video(video: torch.Tensor, gamma: float, gain: float = 1) -> torch.Tensor:
    return adjust_gamma_image(image=video, gamma=gamma, gain=gain)


def posterize(inpt: torch.Tensor, bits: int) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomPosterize` for details."""
    if torch.jit.is_scripting():
        return posterize_image(inpt, bits=bits)

    _log_api_usage_once(posterize)

    kernel = _get_kernel(posterize, type(inpt))
    return kernel(inpt, bits=bits)


@_register_kernel_internal(posterize, torch.Tensor)
@_register_kernel_internal(posterize, ta_tensors.Image)
@_register_kernel_internal(posterize, ta_tensors.BatchImages)
def posterize_image(image: torch.Tensor, bits: int) -> torch.Tensor:
    return TVF.posterize_image(image=image, bits=bits)


@_register_kernel_internal(posterize, ta_tensors.Video)
@_register_kernel_internal(posterize, ta_tensors.BatchVideos)
def posterize_video(video: torch.Tensor, bits: int) -> torch.Tensor:
    return posterize_image(image=video, bits=bits)


def solarize(inpt: torch.Tensor, threshold: float) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomSolarize` for details."""
    if torch.jit.is_scripting():
        return solarize_image(inpt, threshold=threshold)

    _log_api_usage_once(solarize)

    kernel = _get_kernel(solarize, type(inpt))
    return kernel(inpt, threshold=threshold)


@_register_kernel_internal(solarize, torch.Tensor)
@_register_kernel_internal(solarize, ta_tensors.Image)
@_register_kernel_internal(solarize, ta_tensors.BatchImages)
def solarize_image(image: torch.Tensor, threshold: float) -> torch.Tensor:
    return TVF.solarize_image(image=image, threshold=threshold)


@_register_kernel_internal(solarize, ta_tensors.Video)
@_register_kernel_internal(solarize, ta_tensors.BatchVideos)
def solarize_video(video: torch.Tensor, threshold: float) -> torch.Tensor:
    return solarize_image(image=video, threshold=threshold)


def autocontrast(inpt: torch.Tensor) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomAutocontrast` for details."""
    if torch.jit.is_scripting():
        return autocontrast_image(inpt)

    _log_api_usage_once(autocontrast)

    kernel = _get_kernel(autocontrast, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(autocontrast, torch.Tensor)
@_register_kernel_internal(autocontrast, ta_tensors.Image)
@_register_kernel_internal(autocontrast, ta_tensors.BatchImages)
def autocontrast_image(image: torch.Tensor) -> torch.Tensor:
    return TVF.autocontrast_image(image=image)


@_register_kernel_internal(autocontrast, ta_tensors.Video)
@_register_kernel_internal(autocontrast, ta_tensors.BatchVideos)
def autocontrast_video(video: torch.Tensor) -> torch.Tensor:
    return autocontrast_image(image=video)


def equalize(inpt: torch.Tensor) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomEqualize` for details."""
    if torch.jit.is_scripting():
        return equalize_image(inpt)

    _log_api_usage_once(equalize)

    kernel = _get_kernel(equalize, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(equalize, torch.Tensor)
@_register_kernel_internal(equalize, ta_tensors.Image)
@_register_kernel_internal(equalize, ta_tensors.BatchImages)
def equalize_image(image: torch.Tensor) -> torch.Tensor:
    return TVF.equalize_image(image=image)


@_register_kernel_internal(equalize, ta_tensors.Video)
@_register_kernel_internal(equalize, ta_tensors.BatchVideos)
def equalize_video(video: torch.Tensor) -> torch.Tensor:
    return equalize_image(image=video)


def invert(inpt: torch.Tensor) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomInvert`."""
    if torch.jit.is_scripting():
        return invert_image(inpt)

    _log_api_usage_once(invert)

    kernel = _get_kernel(invert, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(invert, torch.Tensor)
@_register_kernel_internal(invert, ta_tensors.Image)
@_register_kernel_internal(invert, ta_tensors.BatchImages)
def invert_image(image: torch.Tensor) -> torch.Tensor:
    return TVF.invert_image(image=image)


@_register_kernel_internal(invert, ta_tensors.Video)
@_register_kernel_internal(invert, ta_tensors.BatchVideos)
def invert_video(video: torch.Tensor) -> torch.Tensor:
    return invert_image(image=video)


def permute_channels(inpt: torch.Tensor, permutation: List[int]) -> torch.Tensor:
    """Permute the channels of the input according to the given permutation."""
    if torch.jit.is_scripting():
        return permute_channels_image(inpt, permutation=permutation)

    _log_api_usage_once(permute_channels)

    kernel = _get_kernel(permute_channels, type(inpt))
    return kernel(inpt, permutation=permutation)


@_register_kernel_internal(permute_channels, torch.Tensor)
@_register_kernel_internal(permute_channels, ta_tensors.Image)
@_register_kernel_internal(permute_channels, ta_tensors.BatchImages)
def permute_channels_image(image: torch.Tensor, permutation: List[int]) -> torch.Tensor:
    return TVF.permute_channels_image(image=image, permutation=permutation)


@_register_kernel_internal(permute_channels, ta_tensors.Video)
@_register_kernel_internal(permute_channels, ta_tensors.BatchVideos)
def permute_channels_video(video: torch.Tensor, permutation: List[int]) -> torch.Tensor:
    return permute_channels_image(image=video, permutation=permutation)
