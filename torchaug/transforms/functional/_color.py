from __future__ import annotations

import torch
from torch import Tensor
from torchvision.transforms.v2.functional import invert, convert_image_dtype

from torchaug.transforms._utils import (
    _assert_channels,
    _assert_image_tensor,
    _assert_tensor,
    _max_value,
    get_channels_height_width,
    is_tensor_on_cpu,
)
from torchaug.utils import _log_api_usage_once


def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    # From Torchvision
    r, g, _ = image.unbind(dim=-3)

    # Implementation is based on
    # https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/src/libImaging/Convert.c#L330
    minc, maxc = torch.aminmax(image, dim=-3)

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occurring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    channels_range = maxc - minc
    # Since `eqc => channels_range = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = channels_range / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    channels_range_divisor = torch.where(eqc, ones, channels_range).unsqueeze_(dim=-3)
    rc, gc, bc = ((maxc.unsqueeze(dim=-3) - image) / channels_range_divisor).unbind(
        dim=-3
    )

    mask_maxc_neq_r = maxc != r
    mask_maxc_eq_g = maxc == g

    hg = rc.add(2.0).sub_(bc).mul_(mask_maxc_eq_g & mask_maxc_neq_r)
    hr = bc.sub_(gc).mul_(~mask_maxc_neq_r)
    hb = (
        gc.add_(4.0)
        .sub_(rc)
        .mul_(mask_maxc_neq_r.logical_and_(mask_maxc_eq_g.logical_not_()))
    )

    h = hr.add_(hg).add_(hb)
    h = h.mul_(1.0 / 6.0).add_(1.0).fmod_(1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv_to_rgb(img: Tensor) -> Tensor:
    # From Torchvision nlgranger pull request to speed conversion: https://github.com/pytorch/vision/pull/7754
    h, s, v = img.unbind(dim=-3)
    h6 = h.mul(6)
    i = torch.floor(h6)
    f = h6.sub_(i)
    i = i.to(dtype=torch.int32)

    sxf = s * f
    one_minus_s = 1.0 - s
    q = (1.0 - sxf).mul_(v).clamp_(0.0, 1.0)
    t = sxf.add_(one_minus_s).mul_(v).clamp_(0.0, 1.0)
    p = one_minus_s.mul_(v).clamp_(0.0, 1.0)
    i.remainder_(6)

    vpqt = torch.stack((v, p, q, t), dim=-3)

    # vpqt -> rgb mapping based on i
    select = torch.tensor(
        [[0, 2, 1, 1, 3, 0], [3, 0, 0, 2, 1, 1], [1, 1, 3, 0, 0, 2]], dtype=torch.long
    )
    select = select.to(device=img.device, non_blocking=True)

    select = select[:, i]
    if select.ndim > 3:
        # if input.shape is (B, ..., C, H, W) then
        # select.shape is (C, B, ...,  H, W)
        # thus we move C axis to get (B, ..., C, H, W)
        select = select.moveaxis(0, -3)

    return vpqt.gather(-3, select)


def adjust_hue(img: Tensor, hue_factor: float) -> Tensor:
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        imgs: Image to be adjusted. It is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of dimensions.
            Note: the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
        hue_factor:  How much to shift the hue channel. Can be 1 or B elements in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        Hue adjusted image.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(adjust_hue)

    if not isinstance(hue_factor, float):
        try:
            hue_factor = float(hue_factor)
        except ValueError:
            raise TypeError(
                f"hue_factor should be a float or convertible to float. Got {type(hue_factor)}."
            )

    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError("hue_factor is not in [-0.5, 0.5].")

    _assert_image_tensor(img)

    _assert_channels(img, [1, 3])
    if get_channels_height_width(img)[0] == 1:  # Match PIL behaviour
        return img

    orig_dtype = img.dtype
    img = convert_image_dtype(img, torch.float32)

    img = _rgb_to_hsv(img)
    h, s, v = img.unbind(dim=-3)
    h = (h + hue_factor) % 1.0
    img = torch.stack((h, s, v), dim=-3)
    img_hue_adj = _hsv_to_rgb(img)

    return convert_image_dtype(img_hue_adj, orig_dtype)


def solarize(
    img: Tensor, threshold: int | float | Tensor, value_check: bool = False
) -> Tensor:
    """Solarize an RGB/grayscale image by inverting all pixel values above a threshold.

    Args:
        img: Image to have its colors inverted. It is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".
        threshold: All pixels equal or above this value are inverted.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization.
    Returns:
        Solarized image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(solarize)

    _assert_tensor(img)
    _assert_image_tensor(img)
    _assert_channels(img, [1, 3])

    if not isinstance(threshold, (int, float)) and (
        not isinstance(threshold, Tensor) or threshold.numel() > 1
    ):
        raise TypeError("threshold should be a float or a tensor of one element.")

    if (
        value_check
        or isinstance(threshold, (int, float) or is_tensor_on_cpu(threshold))
    ) and threshold > _max_value(img.dtype):
        raise ValueError("Threshold should be less than bound of img.")

    inverted_img = invert(img)
    return torch.where(img >= threshold, inverted_img, img)
