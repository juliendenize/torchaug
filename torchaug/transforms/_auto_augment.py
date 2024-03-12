# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C
#
# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten
from torchvision.transforms import AutoAugmentPolicy, InterpolationMode
from torchvision.transforms.v2._utils import _get_fill, _setup_fill_arg, check_type
from torchvision.transforms.v2.functional._geometry import _check_interpolation

from torchaug import ta_tensors

from . import functional as F
from ._transform import Transform
from ._utils import is_pure_tensor
from .functional._meta import get_size
from .functional._utils._kernel import _FillType, _FillTypeJIT
from .functional._utils._tensor import _max_value


ImageOrVideo = Union[
    torch.Tensor,
    ta_tensors.Image,
    ta_tensors.Video,
    ta_tensors.BatchImages,
    ta_tensors.BatchVideos,
]


class _AutoAugmentBase(Transform):
    def __init__(
        self,
        *,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = None,
        batch_transform: bool = False,
    ) -> None:
        super().__init__(batch_transform=batch_transform)
        self.interpolation = _check_interpolation(interpolation)
        self.fill = fill
        self._fill = _setup_fill_arg(fill)

    def _get_random_item(self, dct: Dict[str, Tuple[Callable, bool]]) -> Tuple[str, Tuple[Callable, bool]]:
        keys = tuple(dct.keys())
        key = keys[int(torch.randint(len(keys), ()))]
        return key, dct[key]

    def _flatten_and_extract_image_or_video(
        self,
        inputs: Any,
        unsupported_types: Tuple[Type, ...] = (
            ta_tensors.Mask,
            ta_tensors.BoundingBoxes,
            ta_tensors.BatchBoundingBoxes,
            ta_tensors.BatchMasks,
        ),
    ) -> Tuple[Tuple[List[Any], TreeSpec, int], ImageOrVideo]:
        if self.batch_transform:
            unsupported_types = tuple(list(unsupported_types) + [ta_tensors.Image, ta_tensors.Video])
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)

        image_or_videos = []
        for idx, (inpt, needs_transform) in enumerate(zip(flat_inputs, needs_transform_list)):
            if needs_transform and check_type(
                inpt,
                (
                    ta_tensors.BatchImages,
                    is_pure_tensor,
                    ta_tensors.BatchVideos,
                )
                if self.batch_transform
                else (
                    ta_tensors.Image,
                    ta_tensors.Video,
                    is_pure_tensor,
                    ta_tensors.BatchImages,
                    ta_tensors.BatchVideos,
                ),
            ):
                image_or_videos.append((idx, inpt))
            elif isinstance(inpt, unsupported_types):
                raise TypeError(f"Inputs of type {type(inpt).__name__} are not supported by {type(self).__name__}()")

        if not image_or_videos:
            raise TypeError("Found no image in the sample.")
        if len(image_or_videos) > 1:
            raise TypeError(
                f"Auto augment transformations are only properly defined for a single image or video, "
                f"but found {len(image_or_videos)}."
            )

        idx, image_or_video = image_or_videos[0]
        return (flat_inputs, spec, idx), image_or_video

    def _unflatten_and_insert_image_or_video(
        self,
        flat_inputs_with_spec: Tuple[List[Any], TreeSpec, int],
        image_or_video: ImageOrVideo,
    ) -> Any:
        flat_inputs, spec, idx = flat_inputs_with_spec
        flat_inputs[idx] = image_or_video
        return tree_unflatten(flat_inputs, spec)

    def _apply_image_or_video_transform(
        self,
        image: ImageOrVideo,
        transform_id: str,
        magnitude: float,
        interpolation: Union[InterpolationMode, int],
        fill: Dict[Union[Type, str], _FillTypeJIT],
    ) -> ImageOrVideo:
        fill_ = _get_fill(fill, type(image))

        if transform_id == "Identity":
            return image
        elif transform_id == "ShearX":
            # magnitude should be arctan(magnitude)
            # official autoaug: (1, level, 0, 0, 1, 0)
            # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
            # compared to
            # torchvision:      (1, tan(level), 0, 0, 1, 0)
            # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976

            return F.affine(
                image,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[math.degrees(math.atan(magnitude)), 0.0],
                interpolation=interpolation,
                fill=fill_,
                center=[0, 0],
            )
        elif transform_id == "ShearY":
            # magnitude should be arctan(magnitude)
            # See above
            return F.affine(
                image,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0.0, math.degrees(math.atan(magnitude))],
                interpolation=interpolation,
                fill=fill_,
                center=[0, 0],
            )
        elif transform_id == "TranslateX":
            return F.affine(
                image,
                angle=0.0,
                translate=[int(magnitude), 0],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill_,
            )
        elif transform_id == "TranslateY":
            return F.affine(
                image,
                angle=0.0,
                translate=[0, int(magnitude)],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill_,
            )
        elif transform_id == "Rotate":
            return F.rotate(image, angle=magnitude, interpolation=interpolation, fill=fill_)
        elif transform_id == "Brightness":
            brightness = F.adjust_brightness_batch if self.batch_transform else F.adjust_brightness
            return brightness(image, brightness_factor=1.0 + magnitude)  # type: ignore[operator]
        elif transform_id == "Color":
            saturation = F.adjust_saturation_batch if self.batch_transform else F.adjust_saturation
            return saturation(image, saturation_factor=1.0 + magnitude)  # type: ignore[operator]
        elif transform_id == "Contrast":
            contrast = F.adjust_contrast_batch if self.batch_transform else F.adjust_contrast
            return contrast(image, contrast_factor=1.0 + magnitude)  # type: ignore[operator]
        elif transform_id == "Sharpness":
            sharpness = F.adjust_sharpness
            return sharpness(image, sharpness_factor=1.0 + magnitude)
        elif transform_id == "Posterize":
            return F.posterize(image, bits=int(magnitude))
        elif transform_id == "Solarize":
            bound = _max_value(image.dtype) if isinstance(image, torch.Tensor) else 255.0
            return F.solarize(image, threshold=bound * magnitude)
        elif transform_id == "AutoContrast":
            return F.autocontrast(image)
        elif transform_id == "Equalize":
            return F.equalize(image)
        elif transform_id == "Invert":
            return F.invert(image)
        else:
            raise ValueError(f"No transform available for {transform_id}")


class AutoAugment(_AutoAugmentBase):
    r"""AutoAugment data augmentation method based on
    `"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.

    This transformation works on images and videos only.

    If the input is :class:`torch.Tensor`, it should be of type ``torch.uint8``, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        policy: Desired policy enum defined by
            :class:`torchvision.transforms.autoaugment.AutoAugmentPolicy`.
        interpolation: Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill: Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    _AUGMENTATION_SPACE = {
        "ShearX": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins),
            True,
        ),
        "ShearY": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins),
            True,
        ),
        "TranslateX": (
            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * width, num_bins),
            True,
        ),
        "TranslateY": (
            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * height, num_bins),
            True,
        ),
        "Rotate": (
            lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins),
            True,
        ),
        "Brightness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Color": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Contrast": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Sharpness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Posterize": (
            lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4))).round().int(),
            False,
        ),
        "Solarize": (
            lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins),
            False,
        ),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
        "Invert": (lambda num_bins, height, width: None, False),
    }

    def __init__(
        self,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = None,
    ) -> None:
        super().__init__(interpolation=interpolation, fill=fill)
        self.policy = policy
        self._policies = self._get_policies(policy)

    def _get_policies(
        self, policy: AutoAugmentPolicy
    ) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [
                (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [
                (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
                (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
                (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
                (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
                (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
                (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
                (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
                (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
                (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
                (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
                (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
                (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
                (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
                (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
                (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
                (("Color", 0.9, 9), ("Equalize", 0.6, None)),
                (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
                (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
                (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
                (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
                (("Equalize", 0.8, None), ("Invert", 0.1, None)),
                (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
            ]
        elif policy == AutoAugmentPolicy.SVHN:
            return [
                (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
                (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
                (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
                (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
                (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
                (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
                (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
                (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
                (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
                (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
                (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
                (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
                (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
                (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
            ]
        else:
            raise ValueError(f"The provided policy {policy} is not recognized.")

    def forward(self, *inputs: Any) -> Any:
        (
            flat_inputs_with_spec,
            image_or_video,
        ) = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(image_or_video)

        policy = self._policies[int(torch.randint(len(self._policies), ()))]

        for transform_id, probability, magnitude_idx in policy:
            if not torch.rand(()) <= probability:
                continue

            magnitudes_fn, signed = self._AUGMENTATION_SPACE[transform_id]

            magnitudes = magnitudes_fn(10, height, width)
            if magnitudes is not None:
                magnitude = float(magnitudes[magnitude_idx])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0

            image_or_video = self._apply_image_or_video_transform(
                image_or_video,
                transform_id,
                magnitude,
                interpolation=self.interpolation,
                fill=self._fill,
            )

        return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)


class RandAugment(_AutoAugmentBase):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.

    This transformation works on images and videos only.

    If the input is :class:`torch.Tensor`, it should be of type ``torch.uint8``, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops: Number of augmentation transformations to apply sequentially.
        magnitude: Magnitude for all the transformations.
        num_magnitude_bins: The number of different magnitude values.
        interpolation: Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill: Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, height, width: None, False),
        "ShearX": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins),
            True,
        ),
        "ShearY": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins),
            True,
        ),
        "TranslateX": (
            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * width, num_bins),
            True,
        ),
        "TranslateY": (
            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * height, num_bins),
            True,
        ),
        "Rotate": (
            lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins),
            True,
        ),
        "Brightness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Color": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Contrast": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Sharpness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Posterize": (
            lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4))).round().int(),
            False,
        ),
        "Solarize": (
            lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins),
            False,
        ),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
    }

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = None,
    ) -> None:
        super().__init__(interpolation=interpolation, fill=fill)
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins

    def forward(self, *inputs: Any) -> Any:
        (
            flat_inputs_with_spec,
            image_or_video,
        ) = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(image_or_video)

        for _ in range(self.num_ops):
            transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)
            magnitudes = magnitudes_fn(self.num_magnitude_bins, height, width)
            if magnitudes is not None:
                magnitude = float(magnitudes[self.magnitude])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0
            image_or_video = self._apply_image_or_video_transform(
                image_or_video,
                transform_id,
                magnitude,
                interpolation=self.interpolation,
                fill=self._fill,
            )

        return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)


class TrivialAugmentWide(_AutoAugmentBase):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`_.

    This transformation works on images and videos only.

    If the input is :class:`torch.Tensor`, it should be of type ``torch.uint8``, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins: The number of different magnitude values.
        interpolation: Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill: Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, height, width: None, False),
        "ShearX": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins),
            True,
        ),
        "ShearY": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins),
            True,
        ),
        "TranslateX": (
            lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins),
            True,
        ),
        "TranslateY": (
            lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins),
            True,
        ),
        "Rotate": (
            lambda num_bins, height, width: torch.linspace(0.0, 135.0, num_bins),
            True,
        ),
        "Brightness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins),
            True,
        ),
        "Color": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins),
            True,
        ),
        "Contrast": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins),
            True,
        ),
        "Sharpness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins),
            True,
        ),
        "Posterize": (
            lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6))).round().int(),
            False,
        ),
        "Solarize": (
            lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins),
            False,
        ),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
    }

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = None,
    ):
        super().__init__(interpolation=interpolation, fill=fill)
        self.num_magnitude_bins = num_magnitude_bins

    def forward(self, *inputs: Any) -> Any:
        (
            flat_inputs_with_spec,
            image_or_video,
        ) = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(image_or_video)

        transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)

        magnitudes = magnitudes_fn(self.num_magnitude_bins, height, width)
        if magnitudes is not None:
            magnitude = float(magnitudes[int(torch.randint(self.num_magnitude_bins, ()))])
            if signed and torch.rand(()) <= 0.5:
                magnitude *= -1
        else:
            magnitude = 0.0

        image_or_video = self._apply_image_or_video_transform(
            image_or_video,
            transform_id,
            magnitude,
            interpolation=self.interpolation,
            fill=self._fill,
        )
        return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)


class AugMix(_AutoAugmentBase):
    r"""AugMix data augmentation method based on
    `"AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty" <https://arxiv.org/abs/1912.02781>`_.

    This transformation works on images and videos only.

    If the input is :class:`torch.Tensor`, it should be of type ``torch.uint8``, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        severity: The severity of base augmentation operators.
        mixture_width: The number of augmentation chains.
        chain_depth: The depth of augmentation chains. A negative value denotes stochastic depth sampled
            from the interval [1, 3].
        alpha: The hyperparameter for the probability distributions.
        all_ops: Use all operations (including brightness, contrast, color and sharpness).
        interpolation: Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill: Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    _PARTIAL_AUGMENTATION_SPACE = {
        "ShearX": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins),
            True,
        ),
        "ShearY": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins),
            True,
        ),
        "TranslateX": (
            lambda num_bins, height, width: torch.linspace(0.0, width / 3.0, num_bins),
            True,
        ),
        "TranslateY": (
            lambda num_bins, height, width: torch.linspace(0.0, height / 3.0, num_bins),
            True,
        ),
        "Rotate": (
            lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins),
            True,
        ),
        "Posterize": (
            lambda num_bins, height, width: (4 - (torch.arange(num_bins) / ((num_bins - 1) / 4))).round().int(),
            False,
        ),
        "Solarize": (
            lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins),
            False,
        ),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
    }
    _AUGMENTATION_SPACE: Dict[str, Tuple[Callable[[int, int, int], Optional[torch.Tensor]], bool]] = {
        **_PARTIAL_AUGMENTATION_SPACE,
        "Brightness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Color": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Contrast": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Sharpness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
    }

    def __init__(
        self,
        severity: int = 3,
        mixture_width: int = 3,
        chain_depth: int = -1,
        alpha: float = 1.0,
        all_ops: bool = True,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = None,
    ) -> None:
        super().__init__(interpolation=interpolation, fill=fill, batch_transform=True)
        self._PARAMETER_MAX = 10
        if not (1 <= severity <= self._PARAMETER_MAX):
            raise ValueError(f"The severity must be between [1, {self._PARAMETER_MAX}]. Got {severity} instead.")
        self.severity = severity
        self.mixture_width = mixture_width
        self.chain_depth = chain_depth
        self.alpha = alpha
        self.all_ops = all_ops

    def _sample_dirichlet(self, params: torch.Tensor) -> torch.Tensor:
        # Must be on a separate method so that we can overwrite it in tests.
        return torch._sample_dirichlet(params)

    def forward(self, *inputs: Any) -> Any:
        (
            flat_inputs_with_spec,
            image_or_video,
        ) = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(image_or_video)

        augmentation_space = self._AUGMENTATION_SPACE if self.all_ops else self._PARTIAL_AUGMENTATION_SPACE

        orig_dims = list(image_or_video.shape)
        expected_ndim = 5 if isinstance(image_or_video, ta_tensors.BatchVideos) else 4
        batch = image_or_video.reshape([1] * max(expected_ndim - image_or_video.ndim, 0) + orig_dims)
        batch_dims = [batch.size(0)] + [1] * (batch.ndim - 1)

        # Sample the beta weights for combining the original and augmented image or video. To get Beta, we use a
        # Dirichlet with 2 parameters. The 1st column stores the weights of the original and the 2nd the ones of
        # augmented image or video.
        m = self._sample_dirichlet(
            torch.tensor([self.alpha, self.alpha], device=batch.device).expand(batch_dims[0], -1)
        )

        # Sample the mixing weights and combine them with the ones sampled from Beta for the augmented
        # images or videos.
        combined_weights = self._sample_dirichlet(
            torch.tensor([self.alpha] * self.mixture_width, device=batch.device).expand(batch_dims[0], -1)
        ) * m[:, 1].reshape([batch_dims[0], -1])

        mix = m[:, 0].reshape(batch_dims) * batch
        for i in range(self.mixture_width):
            aug = batch
            depth = self.chain_depth if self.chain_depth > 0 else int(torch.randint(low=1, high=4, size=(1,)).item())
            for _ in range(depth):
                transform_id, (magnitudes_fn, signed) = self._get_random_item(augmentation_space)

                magnitudes = magnitudes_fn(self._PARAMETER_MAX, height, width)
                if magnitudes is not None:
                    magnitude = float(magnitudes[int(torch.randint(self.severity, ()))])
                    if signed and torch.rand(()) <= 0.5:
                        magnitude *= -1
                else:
                    magnitude = 0.0

                aug = self._apply_image_or_video_transform(
                    aug,
                    transform_id,
                    magnitude,
                    interpolation=self.interpolation,
                    fill=self._fill,
                )
            mix.add_(combined_weights[:, i].reshape(batch_dims) * aug)
        mix = mix.reshape(orig_dims).to(dtype=image_or_video.dtype)

        if isinstance(image_or_video, (ta_tensors.BatchImages, ta_tensors.BatchVideos)):
            mix = ta_tensors.wrap(mix, like=image_or_video)

        return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, mix)
