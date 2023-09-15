from __future__ import annotations

from itertools import permutations
from math import ceil

import torch
import torchvision.transforms.functional as F_tv
from torch import Tensor

import torchaug.batch_transforms.functional as F_b
import torchaug.transforms.functional as F
from torchaug.transforms._utils import _check_input
from torchaug.utils import _log_api_usage_once

from ._transform import BatchRandomTransform


class BatchRandomColorJitter(BatchRandomTransform):
    """Randomly change the brightness, contrast, saturation and hue to a batch of images.

    The batch is expected to have [B, ..., 1 or 3, H, W] shape,
    where ... means an arbitrary number of dimensions.

    Args:
        brightness: How much to jitter brightness.
            brightness factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast: How much to jitter contrast.
            contrast factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non-negative numbers.
        saturation: How much to jitter saturation.
            saturation factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue: How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
        p: Probability to apply color jitter.
        num_rand_calls: Number of random calls performed to apply augmentations at different orders on sub-batches.
            If -1, B calls are performed. The maximum is 24 = 4!, adjusted automatically if num_rand_calls > 24.
        inplace: If True, perform inplace operation to save memory and time.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization or large batch size.
    """

    def __init__(
        self,
        brightness: float | tuple[float, float] = 0,
        contrast: float | tuple[float, float] = 0,
        saturation: float | tuple[float, float] = 0,
        hue: float | tuple[float, float] = 0,
        p: float = 0.5,
        num_rand_calls: int = -1,
        inplace: bool = False,
        value_check: bool = False,
    ) -> None:
        super().__init__(p=p, inplace=inplace)
        _log_api_usage_once(self)

        if not isinstance(num_rand_calls, int) or not num_rand_calls >= -1:
            raise ValueError(
                f"num_rand_calls attribute should be an int superior to -1, {num_rand_calls} given."
            )

        self.p = p
        self.num_rand_calls = num_rand_calls
        self.inplace = inplace

        brightness = _check_input(brightness, "brightness")
        if brightness is not None:
            self.register_buffer("brightness", torch.as_tensor(brightness))
        else:
            self.brightness = None

        contrast = _check_input(contrast, "contrast")
        if contrast is not None:
            self.register_buffer("contrast", torch.as_tensor(contrast))
        else:
            self.contrast = None

        saturation = _check_input(saturation, "saturation")
        if saturation is not None:
            self.register_buffer("saturation", torch.as_tensor(saturation))
        else:
            self.saturation = None

        hue = _check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        if hue is not None:
            self.register_buffer("hue", torch.as_tensor(hue))
        else:
            self.hue = None

        self.value_check = value_check

    @staticmethod
    def get_params(
        brightness: Tensor | None,
        contrast: Tensor | None,
        saturation: Tensor | None,
        hue: Tensor | None,
        batch_size: int,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tensor (min, max), optional): The range from which the brightness factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tensor (min, max), optional): The range from which the contrast factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tensor (min, max), optional): The range from which the saturation factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue: The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.
            batch_size: The number of samples to draw.
        """

        def _get_uniform_values(batch_size: int, bounds: Tensor):
            return (bounds[0] - bounds[1]) * torch.rand(
                (batch_size,), device=bounds.device
            ) + bounds[1]

        b = None if brightness is None else _get_uniform_values(batch_size, brightness)
        c = None if contrast is None else _get_uniform_values(batch_size, contrast)
        s = None if saturation is None else _get_uniform_values(batch_size, saturation)
        h = None if hue is None else _get_uniform_values(batch_size, hue)

        return b, c, s, h

    def apply_transform(self, imgs: Tensor) -> Tensor:
        """Color jitter the batch of images.

        .. note:: Apply different transformation orders based on :attr:`~num_rand_calls`.

        Args:
            imgs: Batch of images to jitter.

        Returns:
            Randomly color jittered batch of images.
        """
        if self.num_rand_calls == 0:
            return imgs

        batch_size = imgs.shape[0]
        # At most 24 (= 4!) different combinations possible.
        num_combination = min(
            batch_size,
            self.num_rand_calls if self.num_rand_calls != -1 else batch_size,
            24,
        )

        combinations = list(permutations(range(0, 4)))
        idx_perms = torch.randperm(len(combinations))[:num_combination]

        num_apply_per_combination = ceil(batch_size / num_combination)

        (
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue,
            batch_size,
        )

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)

        for i, idx_perm in enumerate(idx_perms):
            indices_combination = indices_do_apply[
                i * num_apply_per_combination : (i + 1) * num_apply_per_combination
            ]

            fn_idx = combinations[idx_perm]
            imgs_combination = imgs[indices_combination]

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    imgs_combination = F_b.batch_adjust_brightness(
                        imgs_combination,
                        brightness_factor[
                            i
                            * num_apply_per_combination : (i + 1)
                            * num_apply_per_combination
                        ],
                        self.value_check,
                    )
                elif fn_id == 1 and contrast_factor is not None:
                    imgs_combination = F_b.batch_adjust_contrast(
                        imgs_combination,
                        contrast_factor[
                            i
                            * num_apply_per_combination : (i + 1)
                            * num_apply_per_combination
                        ],
                        self.value_check,
                    )
                elif fn_id == 2 and saturation_factor is not None:
                    imgs_combination = F_b.batch_adjust_saturation(
                        imgs_combination,
                        saturation_factor[
                            i
                            * num_apply_per_combination : (i + 1)
                            * num_apply_per_combination
                        ],
                        self.value_check,
                    )
                elif fn_id == 3 and hue_factor is not None:
                    imgs_combination = F_b.batch_adjust_hue(
                        imgs_combination,
                        hue_factor[
                            i
                            * num_apply_per_combination : (i + 1)
                            * num_apply_per_combination
                        ],
                        self.value_check,
                    )

            imgs[indices_combination] = imgs_combination

        return imgs

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness.tolist() if self.brightness is not None else None}"
            f", contrast={self.contrast.tolist() if self.contrast is not None else None}"
            f", saturation={self.saturation.tolist() if self.saturation is not None else None}"
            f", hue={self.hue.tolist() if self.hue is not None else None}"
            f", p={self.p}"
            f", num_rand_calls={self.num_rand_calls}"
            f", inplace={self.inplace}"
            f", value_check={self.value_check})"
        )
        return s


class BatchRandomGrayscale(BatchRandomTransform):
    """Convert batch of images to grayscale.

    The batch of images is expected to be of shape [B, ..., 1 or 3, H, W]
    where ... means an arbitrary number of dimensions.

    Args:
        p: Probability of the images to be grayscaled.
        num_output_channels: Number of channels of the output image. Can be ``1`` or ``3``.
        inplace: If True, perform inplace operation to save memory and time.
    """

    def __init__(
        self,
        p: float = 0.5,
        num_output_channels: int = 3,
        inplace: bool = False,
    ):
        super().__init__(p=p, inplace=inplace)
        _log_api_usage_once(self)

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        self.num_output_channels = num_output_channels

    def apply_transform(self, imgs: Tensor) -> Tensor:
        """Apply grayscale on the batch of images.

        Args:
            imgs: Batch of images to be grayscaled.

        Returns:
            Grayscaled batch of images.
        """
        imgs = F_tv.rgb_to_grayscale(imgs, self.num_output_channels)
        return imgs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"p={self.p}"
            f", num_output_channels={self.num_output_channels}"
            f", inplace={self.inplace})"
        )


class BatchRandomSolarize(BatchRandomTransform):
    """Solarize the batch of images randomly with a given probability by inverting all pixel values above a
    threshold.

    The batch if images it is expected to be in [B, ..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of dimensions.

    Args:
        threshold: all pixels equal or above this value are inverted.
        p: probability of the image being solarized.
        inplace: If True, perform inplace operation to save memory and time.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization or large batch size.
    """

    def __init__(
        self,
        threshold: float,
        p: float = 0.5,
        inplace: bool = False,
        value_check: bool = False,
    ) -> None:
        super().__init__(p=p, inplace=inplace)
        _log_api_usage_once(self)

        self.register_buffer("threshold", torch.as_tensor(threshold))
        self.value_check = value_check

    def apply_transform(self, imgs: Tensor) -> Tensor:
        """Solarize the batch of images.

        Args:
            imgs: Batch of images to be solarized.

        Returns:
            Randomly solarized batch of images.
        """
        imgs = F.solarize(imgs, self.threshold, self.value_check)
        return imgs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold.item()}"
            f", p={self.p}"
            f", inplace={self.inplace}"
            f", value_check={self.value_check})"
        )
