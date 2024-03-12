from __future__ import annotations

import collections.abc
from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from . import functional as F
from ._transform import RandomApplyTransform, Transform
from ._utils import query_chw


class Grayscale(Transform):
    """Convert images or videos to grayscale.

    If the input is a :class:`torch.Tensor`, it is expected
    to have [..., 3 or 1, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        num_output_channels: (1 or 3) number of channels desired for output image.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(self, num_output_channels: int = 1, batch_transform: bool = False):
        super().__init__(
            batch_transform=batch_transform,
        )
        self.num_output_channels = num_output_channels

    @property
    def _reshape_transform(self) -> bool:
        return True

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(
            F.rgb_to_grayscale,
            inpt,
            num_output_channels=self.num_output_channels,
        )

    def extra_repr(self) -> str:  # type: ignore[override]
        return super().extra_repr(exclude_names=["num_chunks", "permute_chunks"])


class RandomGrayscale(RandomApplyTransform):
    """Randomly convert image or videos to grayscale with a probability of p.

    The input is expected to have [..., 3 or 1, H, W] shape,
    where ... means an arbitrary number of leading dimensions

    The output has the same number of channels as the input.

    Args:
        p: probability that image should be converted to grayscale.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(self, p: float = 0.1, batch_inplace: bool = False, batch_transform: bool = False) -> None:
        super().__init__(
            p=p,
            batch_inplace=batch_inplace,
            batch_transform=batch_transform,
        )

    def _get_params(self, flat_inputs: List[Any], *args, **kwargs) -> List[Dict[str, Any]]:
        num_input_channels, *_ = query_chw(flat_inputs)
        return [{"num_input_channels": num_input_channels}]

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(
            F.rgb_to_grayscale,
            inpt,
            num_output_channels=params["num_input_channels"],
        )

    def extra_repr(self) -> str:  # type: ignore[override]
        return super().extra_repr(exclude_names=["num_chunks", "permute_chunks"])


class RandomColorJitter(RandomApplyTransform):
    """Randomly change the brightness, contrast, saturation and hue of an image or video.

    The input is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

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
        p: probability of image being color jittered.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        num_chunks: number of chunks to split the input into.
        permute_chunks: whether to permute the chunks.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(
        self,
        brightness: Optional[Union[float, Sequence[float]]] = None,
        contrast: Optional[Union[float, Sequence[float]]] = None,
        saturation: Optional[Union[float, Sequence[float]]] = None,
        hue: Optional[Union[float, Sequence[float]]] = None,
        p: float = 0.5,
        batch_inplace: bool = False,
        num_chunks: int = 1,
        permute_chunks: bool = False,
        batch_transform: bool = False,
    ) -> None:
        super().__init__(
            p=p,
            batch_inplace=batch_inplace,
            num_chunks=num_chunks,
            permute_chunks=permute_chunks,
            batch_transform=batch_transform,
        )
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self._combinations = list(permutations(range(0, 4)))
        self.num_chunks = num_chunks

    @property
    def num_chunks(self) -> int:
        return self._num_chunks

    @num_chunks.setter
    def num_chunks(self, value: int) -> None:
        if value == -1 or value > 24:
            value = 24
        self._num_chunks = value

    def _check_input(
        self,
        value: Optional[Union[float, Sequence[float]]],
        name: str,
        center: float = 1.0,
        bound: Tuple[float, float] = (0, float("inf")),
        clip_first_on_zero: bool = True,
    ) -> Optional[Tuple[float, float]]:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, collections.abc.Sequence) and len(value) == 2:
            value = [float(v) for v in value]
        else:
            raise TypeError(f"{name}={value} should be a single number or a sequence with length 2.")

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}, but got {value}.")

        return None if value[0] == value[1] == center else (float(value[0]), float(value[1]))

    @staticmethod
    def _generate_value(
        left: float,
        right: float,
        chunk_size: float,
        device: torch.device,
        batch_transform: bool,
    ) -> float:
        if not batch_transform:
            return torch.empty((1,), device="cpu").uniform_(left, right).item()

        return torch.empty((chunk_size,), device=device).uniform_(left, right)

    def _get_params(
        self,
        flat_inputs: List[Any],
        num_chunks: int,
        chunks_indices: Tuple[torch.Tensor],
    ) -> List[Dict[str, Any]]:
        if num_chunks == 1:
            idx_perms = [torch.randint(0, len(self._combinations), (1,)).item()]
        else:
            idx_perms = torch.randperm(len(self._combinations))[:num_chunks].tolist()

        params = []
        device = flat_inputs[0].device

        for i in range(num_chunks):
            chunk_batch_size = chunks_indices[i].shape[0]

            b = (
                None
                if self.brightness is None
                else self._generate_value(
                    self.brightness[0],
                    self.brightness[1],
                    chunk_batch_size,
                    device,
                    self.batch_transform,
                )
            )
            c = (
                None
                if self.contrast is None
                else self._generate_value(
                    self.contrast[0],
                    self.contrast[1],
                    chunk_batch_size,
                    device,
                    self.batch_transform,
                )
            )
            s = (
                None
                if self.saturation is None
                else self._generate_value(
                    self.saturation[0],
                    self.saturation[1],
                    chunk_batch_size,
                    device,
                    self.batch_transform,
                )
            )
            h = (
                None
                if self.hue is None
                else self._generate_value(
                    self.hue[0],
                    self.hue[1],
                    chunk_batch_size,
                    device,
                    self.batch_transform,
                )
            )

            params.append(
                {
                    "fn_idx": self._combinations[idx_perms[i]],
                    "brightness_factor": b,
                    "contrast_factor": c,
                    "saturation_factor": s,
                    "hue_factor": h,
                }
            )

        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        output = inpt
        brightness_factor = params["brightness_factor"]
        contrast_factor = params["contrast_factor"]
        saturation_factor = params["saturation_factor"]
        hue_factor = params["hue_factor"]
        for fn_id in params["fn_idx"]:
            if fn_id == 0 and brightness_factor is not None:
                output = self._call_kernel(
                    F.adjust_brightness_batch if self.batch_transform else F.adjust_brightness,  # type: ignore[arg-type]
                    output,
                    brightness_factor=brightness_factor,
                )
            elif fn_id == 1 and contrast_factor is not None:
                output = self._call_kernel(
                    F.adjust_contrast_batch if self.batch_transform else F.adjust_contrast,  # type: ignore[arg-type]
                    output,
                    contrast_factor=contrast_factor,
                )
            elif fn_id == 2 and saturation_factor is not None:
                output = self._call_kernel(
                    F.adjust_saturation_batch if self.batch_transform else F.adjust_saturation,  # type: ignore[arg-type]
                    output,
                    saturation_factor=saturation_factor,
                )
            elif fn_id == 3 and hue_factor is not None:
                output = self._call_kernel(
                    F.adjust_hue_batch if self.batch_transform else F.adjust_hue,  # type: ignore[arg-type]
                    output,
                    hue_factor=hue_factor,
                )
        return output


class ColorJitter(RandomColorJitter):
    """Randomly change the brightness, contrast, saturation and hue of an image or video.

    The input is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

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
        p: probability of image being color jittered.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        num_chunks: number of chunks to split the input into.
        permute_chunks: whether to permute the chunks.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(
        self,
        brightness: Optional[Union[float, Sequence[float]]] = None,
        contrast: Optional[Union[float, Sequence[float]]] = None,
        saturation: Optional[Union[float, Sequence[float]]] = None,
        hue: Optional[Union[float, Sequence[float]]] = None,
        batch_inplace: bool = False,
        num_chunks: int = 1,
        permute_chunks: bool = False,
        batch_transform: bool = False,
    ) -> None:
        super().__init__(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=1.0,
            batch_inplace=batch_inplace,
            num_chunks=num_chunks,
            permute_chunks=permute_chunks,
            batch_transform=batch_transform,
        )

    def extra_repr(self) -> str:  # type: ignore[override]
        return super().extra_repr(exclude_names=["p"])


class RandomChannelPermutation(RandomApplyTransform):
    """Randomly permute the channels of an image or video.

    Args:
        p: probability of the image being channel permuted.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        num_chunks: number of chunks to split the batched input into.
        permute_chunks: whether to permute the chunks.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(
        self,
        p: float = 1.0,
        batch_inplace: bool = False,
        num_chunks: int = 1,
        permute_chunks: bool = False,
        batch_transform: bool = False,
    ):
        super().__init__(
            p=p,
            batch_inplace=batch_inplace,
            num_chunks=num_chunks,
            permute_chunks=permute_chunks,
            batch_transform=batch_transform,
        )

    def _get_params(
        self,
        flat_inputs: List[Any],
        num_chunks: int,
        chunks_indices: Tuple[torch.Tensor],
    ) -> List[Dict[str, Any]]:
        num_channels, *_ = query_chw(flat_inputs)

        params = []
        for _ in range(num_chunks):
            params.append({"permutation": torch.randperm(num_channels).tolist()})
        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.permute_channels, inpt, params["permutation"])


class RandomPhotometricDistort(RandomApplyTransform):
    """Randomly distorts the image or video as used in `SSD: Single Shot
    MultiBox Detector <https://arxiv.org/abs/1512.02325>`_.

    This transform relies on :class:`~torchaug.transforms.ColorJitter`
    under the hood to adjust the contrast, saturation, hue, brightness, and also
    randomly permutes channels.

    Args:
        brightness: How much to jitter brightness.
            brightness_factor is chosen uniformly from [min, max]. Should be non negative numbers.
        contrast: How much to jitter contrast.
            contrast_factor is chosen uniformly from [min, max]. Should be non-negative numbers.
        saturation: How much to jitter saturation.
            saturation_factor is chosen uniformly from [min, max]. Should be non negative numbers.
        hue: How much to jitter hue.
            hue_factor is chosen uniformly from [min, max].  Should have -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
        p_transform: probability each distortion operation (contrast, saturation, ...) to be applied.
        p: probability of the image being photometrically distorted.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        num_chunks: number of chunks to split the batched input into.
        permute_chunks: whether to permute the chunks.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(
        self,
        brightness: Tuple[float, float] = (0.875, 1.125),
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.5, 1.5),
        hue: Tuple[float, float] = (-0.05, 0.05),
        p_transform: float = 0.5,
        p: float = 0.5,
        batch_inplace: bool = False,
        num_chunks: int = 1,
        permute_chunks: bool = False,
        batch_transform: bool = False,
    ):
        super().__init__(
            p=p,
            batch_inplace=batch_inplace,
            num_chunks=num_chunks,
            permute_chunks=permute_chunks,
            batch_transform=batch_transform,
        )
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.p_transform = p_transform

    def _get_params(
        self, flat_inputs: List[Any], num_chunks: int, chunks_indices: torch.Tensor
    ) -> List[Dict[str, Any]]:
        num_channels, *_ = query_chw(flat_inputs)

        params = []
        device = flat_inputs[0].device

        for i in range(num_chunks):
            chunk_params: Dict[str, Any] = {
                key: ColorJitter._generate_value(
                    range[0],
                    range[1],
                    chunks_indices[i].shape[0],
                    device,
                    self.batch_transform,
                )
                if torch.rand(1) < self.p_transform
                else None
                for key, range in [
                    ("brightness_factor", self.brightness),
                    ("contrast_factor", self.contrast),
                    ("saturation_factor", self.saturation),
                    ("hue_factor", self.hue),
                ]
            }
            chunk_params["contrast_before"] = bool(torch.rand(()) < 0.5)
            chunk_params["channel_permutation"] = (
                torch.randperm(num_channels) if torch.rand(1) < self.p_transform else None
            )

            params.append(chunk_params)

        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["brightness_factor"] is not None:
            inpt = self._call_kernel(
                F.adjust_brightness_batch if self.batch_transform else F.adjust_brightness_batch,  # type: ignore[arg-type]
                inpt,
                brightness_factor=params["brightness_factor"],
            )
        if params["contrast_factor"] is not None and params["contrast_before"]:
            inpt = self._call_kernel(
                F.adjust_contrast_batch if self.batch_transform else F.adjust_contrast,  # type: ignore[arg-type]
                inpt,
                contrast_factor=params["contrast_factor"],
            )
        if params["saturation_factor"] is not None:
            inpt = self._call_kernel(
                F.adjust_saturation_batch if self.batch_transform else F.adjust_saturation,  # type: ignore[arg-type]
                inpt,
                saturation_factor=params["saturation_factor"],
            )
        if params["hue_factor"] is not None:
            inpt = self._call_kernel(
                F.adjust_hue_batch if self.batch_transform else F.adjust_hue,  # type: ignore[arg-type]
                inpt,
                hue_factor=params["hue_factor"],
            )
        if params["contrast_factor"] is not None and not params["contrast_before"]:
            inpt = self._call_kernel(
                F.adjust_contrast_batch if self.batch_transform else F.adjust_contrast,  # type: ignore[arg-type]
                inpt,
                contrast_factor=params["contrast_factor"],
            )
        if params["channel_permutation"] is not None:
            inpt = self._call_kernel(
                F.permute_channels,
                inpt,
                permutation=params["channel_permutation"],
            )
        return inpt


class RandomEqualize(RandomApplyTransform):
    """Equalize the histogram of the given image or video with a given probability.

    The input is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        p: probability of the image being equalized.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(self, p: float = 0.5, batch_inplace: bool = False, batch_transform: bool = False) -> None:
        super().__init__(
            p=p,
            batch_inplace=batch_inplace,
            batch_transform=batch_transform,
        )

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.equalize, inpt)

    def extra_repr(self) -> str:  # type: ignore[override]
        return super().extra_repr(exclude_names=["num_chunks", "permute_chunks"])


class RandomInvert(RandomApplyTransform):
    """Inverts the colors of the given image or video with a given probability.

    The input is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.

    Args:
        p: probability of the image being color inverted.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(self, p: float = 0.5, batch_inplace: bool = False, batch_transform: bool = False) -> None:
        super().__init__(
            p=p,
            batch_inplace=batch_inplace,
            batch_transform=batch_transform,
        )

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.invert, inpt)

    def extra_repr(self) -> str:  # type: ignore[override]
        return super().extra_repr(exclude_names=["num_chunks", "permute_chunks"])


class RandomPosterize(RandomApplyTransform):
    """Posterize the image or video with a given probability by reducing the
    number of bits for each color channel.

    The input should be of type torch.uint8 and it is expected to have [..., 1 or 3, H, W] shape, where ...
    means an arbitrary number of leading dimensions.

    Args:
        bits: number of bits to keep for each channel (0-8)
        p: probability of the image being posterized.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(
        self,
        bits: int,
        p: float = 0.5,
        batch_inplace: bool = False,
        batch_transform: bool = False,
    ) -> None:
        super().__init__(
            p=p,
            batch_inplace=batch_inplace,
            batch_transform=batch_transform,
        )
        self.bits = bits

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(
            F.posterize,
            inpt,
            bits=self.bits,
        )

    def extra_repr(self) -> str:  # type: ignore[override]
        return super().extra_repr(exclude_names=["num_chunks", "permute_chunks"])


class RandomSolarize(RandomApplyTransform):
    """Solarize the image or video with a given probability by inverting all pixel
    values above a threshold.

    The input is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.

    Args:
        threshold: all pixels equal or above this value are inverted.
        p: probability of the image being solarized.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(
        self,
        threshold: float,
        p: float = 0.5,
        batch_inplace: bool = False,
        batch_transform: bool = False,
    ) -> None:
        super().__init__(p=p, batch_inplace=batch_inplace, batch_transform=batch_transform)
        self.threshold = threshold

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(
            F.solarize,
            inpt,
            threshold=self.threshold,
        )

    def extra_repr(self) -> str:  # type: ignore[override]
        return super().extra_repr(exclude_names=["num_chunks", "permute_chunks"])


class RandomAutocontrast(RandomApplyTransform):
    """Autocontrast the pixels of the given image or video with a given probability.

    The input is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        p: probability of the image being autocontrasted.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(self, p: float = 0.5, batch_inplace: bool = False, batch_transform: bool = False) -> None:
        super().__init__(
            p=p,
            batch_inplace=batch_inplace,
            batch_transform=batch_transform,
        )

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.autocontrast, inpt)

    def extra_repr(self) -> str:  # type: ignore[override]
        return super().extra_repr(exclude_names=["num_chunks", "permute_chunks"])


class RandomAdjustSharpness(RandomApplyTransform):
    """Adjust the sharpness of the image or video with a given probability.

    The input is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness_factor: How much to adjust the sharpness. Can be any non-negative number. 0 gives a blurred image,
            1 gives the original image while 2 increases the sharpness by a factor of 2.
        p: probability of the image being sharpened.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(
        self,
        sharpness_factor: float,
        p: float = 0.5,
        batch_inplace: bool = False,
        batch_transform: bool = False,
    ) -> None:
        super().__init__(
            p=p,
            batch_inplace=batch_inplace,
            batch_transform=batch_transform,
        )
        self.sharpness_factor = sharpness_factor

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(
            F.adjust_sharpness,
            inpt,
            sharpness_factor=self.sharpness_factor,
        )

    def extra_repr(self) -> str:  # type: ignore[override]
        return super().extra_repr(exclude_names=["num_chunks", "permute_chunks"])
