from __future__ import annotations

import torch
import torchvision.transforms.functional as F_tv
from torch import Tensor

import torchaug.transforms.functional as F
from torchaug.transforms._utils import _check_input
from torchaug.utils import _log_api_usage_once

from ._transform import RandomTransform


class RandomColorJitter(RandomTransform):
    """Randomly change the brightness, contrast, saturation and hue to images.

    The images is expected to have [..., 1 or 3, H, W] shape, where ...
    means an arbitrary number of leading dimensions.

    Args:
        brightness: How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast: How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non-negative numbers.
        saturation: How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue: How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
        p: Probability to apply color jitter.
    """

    def __init__(
        self,
        brightness: float | tuple[float, float] = 0,
        contrast: float | tuple[float, float] = 0,
        saturation: float | tuple[float, float] = 0,
        hue: float | tuple[float, float] = 0,
        p: float = 0.0,
    ):
        super().__init__(p=p)
        _log_api_usage_once(self)

        self.p = p
        self.brightness = _check_input(brightness, "brightness")
        self.contrast = _check_input(contrast, "contrast")
        self.saturation = _check_input(saturation, "saturation")
        self.hue = _check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )

    @staticmethod
    def get_params(
        brightness: list[float] | None,
        contrast: list[float] | None,
        saturation: list[float] | None,
        hue: list[float] | None,
    ) -> tuple[Tensor, float | None, float | None, float | None, float | None]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness: The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast: The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation: The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue: The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            The parameters used to apply the randomized transform along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = (
            None
            if brightness is None
            else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        )
        c = (
            None
            if contrast is None
            else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        )
        s = (
            None
            if saturation is None
            else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        )
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def apply_transform(self, img: Tensor) -> Tensor:
        """Color jitter the image.

        Args:
            img: Input image.

        Returns:
            Color jittered image.
        """
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F_tv.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F_tv.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F_tv.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue}"
            f", p={self.p})"
        )
        return s


class RandomGrayscale(RandomTransform):
    """Convert an image to grayscale.

    The image is expected to be of shape [..., 1 or 3, H, W]
    where ... means an arbitrary number of dimensions.

    Args:
        p: Probability of the images to be grayscaled.
        num_output_channels: Number of channels of the output image. Can be ``1`` or ``3``.
    """

    def __init__(
        self,
        p: float = 0.5,
        num_output_channels: int = 3,
    ):
        super().__init__(p=p)
        _log_api_usage_once(self)

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        self.num_output_channels = num_output_channels

    def apply_transform(self, img: Tensor) -> Tensor:
        """Apply grayscale on the image.

        Args:
            img: Image to be grayscaled.

        Returns:
            Grayscaled image.
        """
        img = F_tv.rgb_to_grayscale(img, self.num_output_channels)
        return img

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"p={self.p}"
            f", num_output_channels={self.num_output_channels})"
        )


class RandomSolarize(RandomTransform):
    """Solarize the image randomly with a given probability by inverting all pixel values above a threshold.

    The image is expected to be in [..., 1 or 3, H, W] format, where ... means it can have an arbitrary number of
    leading dimensions.

    Args:
        threshold: all pixels equal or above this value are inverted.
        p: probability of the image being solarized.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization.
    """

    def __init__(
        self,
        threshold: float,
        p: float = 0.5,
        value_check: bool = False,
    ):
        super().__init__(p=p)
        _log_api_usage_once(self)

        self.register_buffer("threshold", torch.as_tensor(threshold))
        self.value_check = value_check

    def apply_transform(self, img: Tensor):
        """Solarize the image.

        Args:
            img: Image to be solarized.

        Returns:
            Solarized image.
        """
        return F.solarize(img, self.threshold, self.value_check)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(threshold={self.threshold.item()}"
            f", p={self.p}"
            f", value_check={self.value_check})"
        )
