# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C
#
# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch
import torchvision.transforms.v2.functional as TVF
from torch.nn.functional import grid_sample
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2.functional._geometry import _check_interpolation

from torchaug import ta_tensors
from torchaug._utils import _log_api_usage_once
from torchaug.ta_tensors._bounding_boxes import _convert_ta_format_to_tv_format

from ._meta import clamp_bounding_boxes, convert_bounding_box_format
from ._utils._kernel import (
    _FillTypeJIT,
    _get_kernel,
    _register_five_ten_crop_kernel_internal,
    _register_kernel_internal,
)


def horizontal_flip(inpt: torch.Tensor) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomHorizontalFlip` for details."""
    if torch.jit.is_scripting():
        return horizontal_flip_image(inpt)

    _log_api_usage_once(horizontal_flip)

    kernel = _get_kernel(horizontal_flip, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(horizontal_flip, torch.Tensor)
@_register_kernel_internal(horizontal_flip, ta_tensors.Image)
@_register_kernel_internal(horizontal_flip, ta_tensors.BatchImages)
def horizontal_flip_image(image: torch.Tensor) -> torch.Tensor:
    return TVF.horizontal_flip_image(image=image)


@_register_kernel_internal(horizontal_flip, ta_tensors.Mask)
@_register_kernel_internal(horizontal_flip, ta_tensors.BatchMasks)
def horizontal_flip_mask(mask: torch.Tensor) -> torch.Tensor:
    return horizontal_flip_image(mask)


def horizontal_flip_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    canvas_size: Tuple[int, int],
) -> torch.Tensor:
    return TVF.horizontal_flip_bounding_boxes(
        bounding_boxes=bounding_boxes, format=_convert_ta_format_to_tv_format(format), canvas_size=canvas_size
    )


@_register_kernel_internal(horizontal_flip, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(horizontal_flip, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def _horizontal_flip_bounding_boxes_dispatch(
    inpt: ta_tensors.BoundingBoxes,
) -> ta_tensors.BoundingBoxes:
    output = horizontal_flip_bounding_boxes(
        inpt.as_subclass(torch.Tensor), format=inpt.format, canvas_size=inpt.canvas_size
    )
    return ta_tensors.wrap(output, like=inpt)


@_register_kernel_internal(horizontal_flip, ta_tensors.Video)
@_register_kernel_internal(horizontal_flip, ta_tensors.BatchVideos)
def horizontal_flip_video(video: torch.Tensor) -> torch.Tensor:
    return horizontal_flip_image(video)


def vertical_flip(inpt: torch.Tensor) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomVerticalFlip` for details."""
    if torch.jit.is_scripting():
        return vertical_flip_image(inpt)

    _log_api_usage_once(vertical_flip)

    kernel = _get_kernel(vertical_flip, type(inpt))
    return kernel(inpt)


@_register_kernel_internal(vertical_flip, torch.Tensor)
@_register_kernel_internal(vertical_flip, ta_tensors.Image)
@_register_kernel_internal(vertical_flip, ta_tensors.BatchImages)
def vertical_flip_image(image: torch.Tensor) -> torch.Tensor:
    return TVF.vertical_flip_image(image=image)


@_register_kernel_internal(vertical_flip, ta_tensors.Mask)
@_register_kernel_internal(vertical_flip, ta_tensors.BatchMasks)
def vertical_flip_mask(mask: torch.Tensor) -> torch.Tensor:
    return vertical_flip_image(mask)


def vertical_flip_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    canvas_size: Tuple[int, int],
) -> torch.Tensor:
    return TVF.vertical_flip_bounding_boxes(
        bounding_boxes=bounding_boxes, format=_convert_ta_format_to_tv_format(format), canvas_size=canvas_size
    )


@_register_kernel_internal(vertical_flip, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(vertical_flip, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def _vertical_flip_bounding_boxes_dispatch(
    inpt: ta_tensors.BoundingBoxes,
) -> ta_tensors.BoundingBoxes:
    output = vertical_flip_bounding_boxes(
        inpt.as_subclass(torch.Tensor), format=inpt.format, canvas_size=inpt.canvas_size
    )
    return ta_tensors.wrap(output, like=inpt)


@_register_kernel_internal(vertical_flip, ta_tensors.Video)
@_register_kernel_internal(vertical_flip, ta_tensors.BatchVideos)
def vertical_flip_video(video: torch.Tensor) -> torch.Tensor:
    return vertical_flip_image(video)


def resize(
    inpt: torch.Tensor,
    size: List[int],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: bool = True,
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.Resize` for details."""
    if torch.jit.is_scripting():
        return resize_image(
            inpt,
            size=size,
            interpolation=interpolation,
            max_size=max_size,
            antialias=antialias,
        )

    _log_api_usage_once(resize)

    kernel = _get_kernel(resize, type(inpt))
    return kernel(
        inpt,
        size=size,
        interpolation=interpolation,
        max_size=max_size,
        antialias=antialias,
    )


@_register_kernel_internal(resize, torch.Tensor)
@_register_kernel_internal(resize, ta_tensors.Image)
@_register_kernel_internal(resize, ta_tensors.BatchImages)
def resize_image(
    image: torch.Tensor,
    size: List[int],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: bool = True,
) -> torch.Tensor:
    return TVF.resize_image(
        image=image,
        size=size,
        interpolation=interpolation,
        max_size=max_size,
        antialias=antialias,
    )


def resize_mask(mask: torch.Tensor, size: List[int], max_size: Optional[int] = None) -> torch.Tensor:
    return TVF.resize_mask(mask=mask, size=size, max_size=max_size)


@_register_kernel_internal(resize, ta_tensors.Mask, ta_tensor_wrapper=False)
@_register_kernel_internal(resize, ta_tensors.BatchMasks, ta_tensor_wrapper=False)
def _resize_mask_dispatch(
    inpt: ta_tensors.Mask,
    size: List[int],
    max_size: Optional[int] = None,
    **kwargs: Any,
) -> ta_tensors.Mask:
    output = resize_mask(inpt.as_subclass(torch.Tensor), size, max_size=max_size)
    return ta_tensors.wrap(output, like=inpt)


def resize_bounding_boxes(
    bounding_boxes: torch.Tensor,
    canvas_size: Tuple[int, int],
    size: List[int],
    max_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    return TVF.resize_bounding_boxes(
        bounding_boxes=bounding_boxes,
        canvas_size=canvas_size,
        size=size,
        max_size=max_size,
    )


@_register_kernel_internal(resize, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(resize, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def _resize_bounding_boxes_dispatch(
    inpt: ta_tensors.BoundingBoxes,
    size: List[int],
    max_size: Optional[int] = None,
    **kwargs: Any,
) -> ta_tensors.BoundingBoxes:
    output, canvas_size = resize_bounding_boxes(
        inpt.as_subclass(torch.Tensor), inpt.canvas_size, size, max_size=max_size
    )
    return ta_tensors.wrap(output, like=inpt, canvas_size=canvas_size)


@_register_kernel_internal(resize, ta_tensors.Video)
@_register_kernel_internal(resize, ta_tensors.BatchVideos)
def resize_video(
    video: torch.Tensor,
    size: List[int],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: bool = True,
) -> torch.Tensor:
    return resize_image(
        image=video,
        size=size,
        interpolation=interpolation,
        max_size=max_size,
        antialias=antialias,
    )


def affine(
    inpt: torch.Tensor,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    fill: _FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomAffine` for details."""
    if torch.jit.is_scripting():
        return affine_image(
            inpt,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=fill,
            center=center,
        )

    _log_api_usage_once(affine)

    kernel = _get_kernel(affine, type(inpt))
    return kernel(
        inpt,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        interpolation=interpolation,
        fill=fill,
        center=center,
    )


@_register_kernel_internal(affine, torch.Tensor)
@_register_kernel_internal(affine, ta_tensors.Image)
@_register_kernel_internal(affine, ta_tensors.BatchImages)
def affine_image(
    image: torch.Tensor,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    fill: _FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    return TVF.affine_image(
        image=image,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        interpolation=interpolation,
        fill=fill,
        center=center,
    )


def affine_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    canvas_size: Tuple[int, int],
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    return TVF.affine_bounding_boxes(
        bounding_boxes=bounding_boxes,
        format=_convert_ta_format_to_tv_format(format),
        canvas_size=canvas_size,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        center=center,
    )


@_register_kernel_internal(affine, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(affine, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def _affine_bounding_boxes_dispatch(
    inpt: ta_tensors.BoundingBoxes,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    center: Optional[List[float]] = None,
    **kwargs,
) -> ta_tensors.BoundingBoxes:
    output = affine_bounding_boxes(
        inpt.as_subclass(torch.Tensor),
        format=inpt.format,
        canvas_size=inpt.canvas_size,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        center=center,
    )
    return ta_tensors.wrap(output, like=inpt)


def affine_mask(
    mask: torch.Tensor,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    fill: _FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    return TVF.affine_mask(
        mask=mask,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        fill=fill,
        center=center,
    )


@_register_kernel_internal(affine, ta_tensors.Mask, ta_tensor_wrapper=False)
@_register_kernel_internal(affine, ta_tensors.BatchMasks, ta_tensor_wrapper=False)
def _affine_mask_dispatch(
    inpt: ta_tensors.Mask,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    fill: _FillTypeJIT = None,
    center: Optional[List[float]] = None,
    **kwargs,
) -> ta_tensors.Mask:
    output = affine_mask(
        inpt.as_subclass(torch.Tensor),
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        fill=fill,
        center=center,
    )
    return ta_tensors.wrap(output, like=inpt)


@_register_kernel_internal(affine, ta_tensors.Video)
@_register_kernel_internal(affine, ta_tensors.BatchVideos)
def affine_video(
    video: torch.Tensor,
    angle: Union[int, float],
    translate: List[float],
    scale: float,
    shear: List[float],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    fill: _FillTypeJIT = None,
    center: Optional[List[float]] = None,
) -> torch.Tensor:
    return affine_image(
        video,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        interpolation=interpolation,
        fill=fill,
        center=center,
    )


def rotate(
    inpt: torch.Tensor,
    angle: float,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    expand: bool = False,
    center: Optional[List[float]] = None,
    fill: _FillTypeJIT = None,
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomRotation` for details."""
    if torch.jit.is_scripting():
        return rotate_image(
            inpt,
            angle=angle,
            interpolation=interpolation,
            expand=expand,
            fill=fill,
            center=center,
        )

    _log_api_usage_once(rotate)

    kernel = _get_kernel(rotate, type(inpt))
    return kernel(
        inpt,
        angle=angle,
        interpolation=interpolation,
        expand=expand,
        fill=fill,
        center=center,
    )


@_register_kernel_internal(rotate, torch.Tensor)
@_register_kernel_internal(rotate, ta_tensors.Image)
@_register_kernel_internal(rotate, ta_tensors.BatchImages)
def rotate_image(
    image: torch.Tensor,
    angle: float,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    expand: bool = False,
    center: Optional[List[float]] = None,
    fill: _FillTypeJIT = None,
) -> torch.Tensor:
    return TVF.rotate_image(
        image=image,
        angle=angle,
        interpolation=interpolation,
        expand=expand,
        center=center,
        fill=fill,
    )


def rotate_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    canvas_size: Tuple[int, int],
    angle: float,
    expand: bool = False,
    center: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    return TVF.rotate_bounding_boxes(
        bounding_boxes=bounding_boxes,
        format=_convert_ta_format_to_tv_format(format),
        canvas_size=canvas_size,
        angle=angle,
        expand=expand,
        center=center,
    )


@_register_kernel_internal(rotate, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(rotate, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def _rotate_bounding_boxes_dispatch(
    inpt: ta_tensors.BoundingBoxes,
    angle: float,
    expand: bool = False,
    center: Optional[List[float]] = None,
    **kwargs,
) -> ta_tensors.BoundingBoxes:
    output, canvas_size = rotate_bounding_boxes(
        inpt.as_subclass(torch.Tensor),
        format=inpt.format,
        canvas_size=inpt.canvas_size,
        angle=angle,
        expand=expand,
        center=center,
    )
    return ta_tensors.wrap(output, like=inpt, canvas_size=canvas_size)


def rotate_mask(
    mask: torch.Tensor,
    angle: float,
    expand: bool = False,
    center: Optional[List[float]] = None,
    fill: _FillTypeJIT = None,
) -> torch.Tensor:
    return TVF.rotate_mask(
        mask=mask,
        angle=angle,
        expand=expand,
        center=center,
        fill=fill,
    )


@_register_kernel_internal(rotate, ta_tensors.Mask, ta_tensor_wrapper=False)
@_register_kernel_internal(rotate, ta_tensors.BatchMasks, ta_tensor_wrapper=False)
def _rotate_mask_dispatch(
    inpt: ta_tensors.Mask,
    angle: float,
    expand: bool = False,
    center: Optional[List[float]] = None,
    fill: _FillTypeJIT = None,
    **kwargs,
) -> ta_tensors.Mask:
    output = rotate_mask(
        inpt.as_subclass(torch.Tensor),
        angle=angle,
        expand=expand,
        fill=fill,
        center=center,
    )
    return ta_tensors.wrap(output, like=inpt)


@_register_kernel_internal(rotate, ta_tensors.Video)
@_register_kernel_internal(rotate, ta_tensors.BatchVideos)
def rotate_video(
    video: torch.Tensor,
    angle: float,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
    expand: bool = False,
    center: Optional[List[float]] = None,
    fill: _FillTypeJIT = None,
) -> torch.Tensor:
    return rotate_image(
        image=video,
        angle=angle,
        interpolation=interpolation,
        expand=expand,
        fill=fill,
        center=center,
    )


def pad(
    inpt: torch.Tensor,
    padding: List[int],
    fill: Optional[Union[int, float, List[float]]] = None,
    padding_mode: str = "constant",
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.Pad` for details."""
    if torch.jit.is_scripting():
        return pad_image(inpt, padding=padding, fill=fill, padding_mode=padding_mode)

    _log_api_usage_once(pad)

    kernel = _get_kernel(pad, type(inpt))
    return kernel(inpt, padding=padding, fill=fill, padding_mode=padding_mode)


@_register_kernel_internal(pad, torch.Tensor)
@_register_kernel_internal(pad, ta_tensors.Image)
@_register_kernel_internal(pad, ta_tensors.BatchImages)
def pad_image(
    image: torch.Tensor,
    padding: List[int],
    fill: Optional[Union[int, float, List[float]]] = None,
    padding_mode: str = "constant",
) -> torch.Tensor:
    return TVF.pad_image(image=image, padding=padding, fill=fill, padding_mode=padding_mode)


@_register_kernel_internal(pad, ta_tensors.Mask)
@_register_kernel_internal(pad, ta_tensors.BatchMasks)
def pad_mask(
    mask: torch.Tensor,
    padding: List[int],
    fill: Optional[Union[int, float, List[float]]] = None,
    padding_mode: str = "constant",
) -> torch.Tensor:
    return TVF.pad_mask(mask=mask, padding=padding, fill=fill, padding_mode=padding_mode)


def pad_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    canvas_size: Tuple[int, int],
    padding: List[int],
    padding_mode: str = "constant",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    return TVF.pad_bounding_boxes(
        bounding_boxes=bounding_boxes,
        format=_convert_ta_format_to_tv_format(format),
        canvas_size=canvas_size,
        padding=padding,
        padding_mode=padding_mode,
    )


@_register_kernel_internal(pad, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(pad, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def _pad_bounding_boxes_dispatch(
    inpt: ta_tensors.BoundingBoxes,
    padding: List[int],
    padding_mode: str = "constant",
    **kwargs,
) -> ta_tensors.BoundingBoxes:
    output, canvas_size = pad_bounding_boxes(
        inpt.as_subclass(torch.Tensor),
        format=inpt.format,
        canvas_size=inpt.canvas_size,
        padding=padding,
        padding_mode=padding_mode,
    )
    return ta_tensors.wrap(output, like=inpt, canvas_size=canvas_size)


@_register_kernel_internal(pad, ta_tensors.Video)
@_register_kernel_internal(pad, ta_tensors.BatchVideos)
def pad_video(
    video: torch.Tensor,
    padding: List[int],
    fill: Optional[Union[int, float, List[float]]] = None,
    padding_mode: str = "constant",
) -> torch.Tensor:
    return pad_image(image=video, padding=padding, fill=fill, padding_mode=padding_mode)


def crop(inpt: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomCrop` for details."""
    if torch.jit.is_scripting():
        return crop_image(inpt, top=top, left=left, height=height, width=width)

    _log_api_usage_once(crop)

    kernel = _get_kernel(crop, type(inpt))
    return kernel(inpt, top=top, left=left, height=height, width=width)


@_register_kernel_internal(crop, torch.Tensor)
@_register_kernel_internal(crop, ta_tensors.Image)
@_register_kernel_internal(crop, ta_tensors.BatchImages)
def crop_image(image: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    return TVF.crop_image(image=image, top=top, left=left, height=height, width=width)


def crop_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    top: int,
    left: int,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    return TVF.crop_bounding_boxes(
        bounding_boxes=bounding_boxes,
        format=_convert_ta_format_to_tv_format(format),
        top=top,
        left=left,
        height=height,
        width=width,
    )


@_register_kernel_internal(crop, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(crop, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def _crop_bounding_boxes_dispatch(
    inpt: ta_tensors.BoundingBoxes, top: int, left: int, height: int, width: int
) -> ta_tensors.BoundingBoxes:
    output, canvas_size = crop_bounding_boxes(
        inpt.as_subclass(torch.Tensor),
        format=inpt.format,
        top=top,
        left=left,
        height=height,
        width=width,
    )
    return ta_tensors.wrap(output, like=inpt, canvas_size=canvas_size)


@_register_kernel_internal(crop, ta_tensors.Mask)
@_register_kernel_internal(crop, ta_tensors.BatchMasks)
def crop_mask(mask: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    return TVF.crop_mask(mask=mask, top=top, left=left, height=height, width=width)


@_register_kernel_internal(crop, ta_tensors.Video)
@_register_kernel_internal(crop, ta_tensors.BatchVideos)
def crop_video(video: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    return crop_image(image=video, top=top, left=left, height=height, width=width)


def perspective(
    inpt: torch.Tensor,
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: _FillTypeJIT = None,
    coefficients: Optional[List[float]] = None,
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomPerspective` for details."""
    if torch.jit.is_scripting():
        return perspective_image(
            inpt,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=interpolation,
            fill=fill,
            coefficients=coefficients,
        )

    _log_api_usage_once(perspective)

    kernel = _get_kernel(perspective, type(inpt))
    return kernel(
        inpt,
        startpoints=startpoints,
        endpoints=endpoints,
        interpolation=interpolation,
        fill=fill,
        coefficients=coefficients,
    )


@_register_kernel_internal(perspective, torch.Tensor)
@_register_kernel_internal(perspective, ta_tensors.Image)
@_register_kernel_internal(perspective, ta_tensors.BatchImages)
def perspective_image(
    image: torch.Tensor,
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: _FillTypeJIT = None,
    coefficients: Optional[List[float]] = None,
) -> torch.Tensor:
    return TVF.perspective_image(
        image=image,
        startpoints=startpoints,
        endpoints=endpoints,
        interpolation=interpolation,
        fill=fill,
        coefficients=coefficients,
    )


def perspective_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    canvas_size: Tuple[int, int],
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    coefficients: Optional[List[float]] = None,
) -> torch.Tensor:
    return TVF.perspective_bounding_boxes(
        bounding_boxes=bounding_boxes,
        format=_convert_ta_format_to_tv_format(format),
        canvas_size=canvas_size,
        startpoints=startpoints,
        endpoints=endpoints,
        coefficients=coefficients,
    )


@_register_kernel_internal(perspective, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(perspective, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def _perspective_bounding_boxes_dispatch(
    inpt: ta_tensors.BoundingBoxes,
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    coefficients: Optional[List[float]] = None,
    **kwargs,
) -> ta_tensors.BoundingBoxes:
    output = perspective_bounding_boxes(
        inpt.as_subclass(torch.Tensor),
        format=inpt.format,
        canvas_size=inpt.canvas_size,
        startpoints=startpoints,
        endpoints=endpoints,
        coefficients=coefficients,
    )
    return ta_tensors.wrap(output, like=inpt)


def perspective_mask(
    mask: torch.Tensor,
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    fill: _FillTypeJIT = None,
    coefficients: Optional[List[float]] = None,
) -> torch.Tensor:
    return TVF.perspective_mask(
        mask=mask,
        startpoints=startpoints,
        endpoints=endpoints,
        fill=fill,
        coefficients=coefficients,
    )


@_register_kernel_internal(perspective, ta_tensors.Mask, ta_tensor_wrapper=False)
@_register_kernel_internal(perspective, ta_tensors.BatchMasks, ta_tensor_wrapper=False)
def _perspective_mask_dispatch(
    inpt: ta_tensors.Mask,
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    fill: _FillTypeJIT = None,
    coefficients: Optional[List[float]] = None,
    **kwargs,
) -> ta_tensors.Mask:
    output = perspective_mask(
        inpt.as_subclass(torch.Tensor),
        startpoints=startpoints,
        endpoints=endpoints,
        fill=fill,
        coefficients=coefficients,
    )
    return ta_tensors.wrap(output, like=inpt)


@_register_kernel_internal(perspective, ta_tensors.Video)
@_register_kernel_internal(perspective, ta_tensors.BatchVideos)
def perspective_video(
    video: torch.Tensor,
    startpoints: Optional[List[List[int]]],
    endpoints: Optional[List[List[int]]],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: _FillTypeJIT = None,
    coefficients: Optional[List[float]] = None,
) -> torch.Tensor:
    return perspective_image(
        image=video,
        startpoints=startpoints,
        endpoints=endpoints,
        interpolation=interpolation,
        fill=fill,
        coefficients=coefficients,
    )


def elastic(
    inpt: torch.Tensor,
    displacement: torch.Tensor,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: _FillTypeJIT = None,
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.ElasticTransform` for details."""
    if torch.jit.is_scripting():
        return elastic_image(inpt, displacement=displacement, interpolation=interpolation, fill=fill)

    _log_api_usage_once(elastic)

    kernel = _get_kernel(elastic, type(inpt))
    return kernel(inpt, displacement=displacement, interpolation=interpolation, fill=fill)


def elastic_batch(
    inpt: torch.Tensor,
    displacement: torch.Tensor,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: _FillTypeJIT = None,
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.ElasticTransform` for details."""
    if torch.jit.is_scripting():
        return elastic_batch_images(
            inpt,
            displacement=displacement,
            interpolation=interpolation,
            fill=fill,
        )

    _log_api_usage_once(elastic_batch)

    kernel = _get_kernel(elastic_batch, type(inpt))
    return kernel(inpt, displacement=displacement, interpolation=interpolation, fill=fill)


@_register_kernel_internal(elastic, torch.Tensor)
@_register_kernel_internal(elastic, ta_tensors.Image)
@_register_kernel_internal(elastic, ta_tensors.BatchImages)
def elastic_image(
    image: torch.Tensor,
    displacement: torch.Tensor,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: _FillTypeJIT = None,
) -> torch.Tensor:
    return TVF.elastic_image(image=image, displacement=displacement, interpolation=interpolation, fill=fill)


def _create_identity_grid_batch(
    batch_size: int, size: Tuple[int, int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    sy, sx = size
    base_grid = torch.empty(batch_size, sy, sx, 2, device=device, dtype=dtype)
    x_grid = torch.linspace((-sx + 1) / sx, (sx - 1) / sx, sx, device=device, dtype=dtype)
    base_grid[..., 0].copy_(x_grid)

    y_grid = torch.linspace((-sy + 1) / sy, (sy - 1) / sy, sy, device=device, dtype=dtype).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)

    return base_grid


def _apply_grid_transform_batch(
    images: torch.Tensor, grid: torch.Tensor, mode: str, fill: _FillTypeJIT
) -> torch.Tensor:
    input_shape = images.shape
    output_height, output_width = grid.shape[1], grid.shape[2]
    batch_size = input_shape[0]
    num_channels, input_height, input_width = input_shape[-3:]
    output_shape = input_shape[:-3] + (num_channels, output_height, output_width)

    if images.numel() == 0:
        return images.reshape(output_shape)

    images = images.reshape(batch_size, -1, num_channels, input_height, input_width)
    squashed_dim = images.shape[1]
    images = images.reshape(-1, num_channels, input_height, input_width)

    # We are using context knowledge that grid should have float dtype
    fp = images.dtype == grid.dtype
    float_images = images if fp else images.to(grid.dtype)
    if squashed_dim > 1:
        # Apply same grid to a batch of images
        grid = grid.repeat_interleave(squashed_dim, dim=0)
    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice

    if fill is not None:
        mask = torch.ones(
            (
                batch_size * squashed_dim,
                1,
                input_height,
                input_width,
            ),
            dtype=float_images.dtype,
            device=float_images.device,
        )
        float_images = torch.cat((float_images, mask), dim=1)

    float_images = grid_sample(float_images, grid, mode=mode, padding_mode="zeros", align_corners=False)

    # Fill with required color
    if fill is not None:
        float_images, mask = torch.tensor_split(float_images, indices=(-1,), dim=-3)
        mask = mask.expand_as(float_images)
        fill_list = fill if isinstance(fill, (tuple, list)) else [float(fill)]  # type: ignore[arg-type]
        fill_images = torch.tensor(fill_list, dtype=float_images.dtype, device=float_images.device).view(1, -1, 1, 1)
        if mode == "nearest":
            bool_mask = mask < 0.5
            float_images[bool_mask] = fill_images.expand_as(float_images)[bool_mask]
        else:  # 'bilinear'
            # The following is mathematically equivalent to:
            # images * mask + (1.0 - mask) * fill = images * mask - fill * mask + fill = mask * (images - fill) + fill
            float_images = float_images.sub_(fill_images).mul_(mask).add_(fill_images)

    images = float_images.round_().to(images.dtype) if not fp else float_images

    return images.reshape(output_shape)


@_register_kernel_internal(elastic_batch, torch.Tensor)
@_register_kernel_internal(elastic_batch, ta_tensors.BatchImages)
def elastic_batch_images(
    images: torch.Tensor,
    displacement: torch.Tensor,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: _FillTypeJIT = None,
) -> torch.Tensor:
    if not isinstance(displacement, torch.Tensor):
        raise TypeError("Argument displacement should be a Tensor")

    interpolation = _check_interpolation(interpolation)

    batch_size = images.shape[0]
    height, width = images.shape[-2:]
    device = images.device
    dtype = images.dtype if torch.is_floating_point(images) else torch.float32

    # Patch: elastic transform should support (cpu,f16) input
    is_cpu_half = device.type == "cpu" and dtype == torch.float16
    if is_cpu_half:
        images = images.to(torch.float32)
        dtype = torch.float32

    # We are aware that if input images dtype is uint8 and displacement is float64 then
    # displacement will be cast to float32 and all computations will be done with float32
    # We can fix this later if needed
    expected_shape = (batch_size, height, width, 2)
    if expected_shape != displacement.shape:
        raise ValueError(f"Argument displacement shape should be {expected_shape}, but given {displacement.shape}")

    grid = _create_identity_grid_batch(batch_size, (height, width), device=device, dtype=dtype).add_(
        displacement.to(dtype=dtype, device=device)
    )
    output = _apply_grid_transform_batch(images, grid, interpolation.value, fill=fill)

    if is_cpu_half:
        output = output.to(torch.float16)

    return output


def elastic_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    canvas_size: Tuple[int, int],
    displacement: torch.Tensor,
) -> torch.Tensor:
    return TVF.elastic_bounding_boxes(
        bounding_boxes=bounding_boxes,
        format=_convert_ta_format_to_tv_format(format),
        canvas_size=canvas_size,
        displacement=displacement,
    )


def elastic_batch_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    canvas_size: Tuple[int, int],
    displacement: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(displacement, torch.Tensor):
        raise TypeError("Argument displacement should be a Tensor")
    else:
        batch_size = displacement.shape[0]
        expected_shape = (batch_size, canvas_size[0], canvas_size[1], 2)
        if displacement.shape != expected_shape:
            raise ValueError(f"Argument displacement shape should be {expected_shape}, but given {displacement.shape}")

    if bounding_boxes.numel() == 0:
        return bounding_boxes

    # TODO: add in docstring about approximation we are doing for grid inversion
    device = bounding_boxes.device
    dtype = bounding_boxes.dtype if torch.is_floating_point(bounding_boxes) else torch.float32

    if displacement.dtype != dtype or displacement.device != device:
        displacement = displacement.to(dtype=dtype, device=device)

    original_shape = bounding_boxes.shape
    # TODO: first cast to float if bbox is int64 before convert_bounding_box_format
    bounding_boxes = (
        convert_bounding_box_format(
            bounding_boxes,
            old_format=format,
            new_format=ta_tensors.BoundingBoxFormat.XYXY,
        )
    ).reshape(-1, 4)

    id_grid = _create_identity_grid_batch(batch_size, canvas_size, device=device, dtype=dtype)
    # We construct an approximation of inverse grid as inv_grid = id_grid - displacement
    # This is not an exact inverse of the grid
    inv_grid = id_grid.sub_(displacement)

    # Get points from bboxes
    points = bounding_boxes[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].reshape(batch_size, -1, 2)
    if points.is_floating_point():
        points = points.ceil_()
    index_xy = points.to(dtype=torch.long)
    index_x, index_y = index_xy[:, :, 0], index_xy[:, :, 1]

    # Transform points:
    out_bboxes = []
    # TODO: vectorize this loop
    for i in range(batch_size):
        t_size = torch.tensor(canvas_size[::-1], device=displacement.device, dtype=displacement.dtype)
        transformed_points_i = inv_grid[i, index_y[i], index_x[i], :].add_(1).mul_(0.5 * t_size).sub_(0.5)
        transformed_points_i = transformed_points_i.reshape(-1, 4, 2)

        out_bbox_mins, out_bbox_maxs = torch.aminmax(transformed_points_i, dim=1)
        out_bboxes_i = clamp_bounding_boxes(
            torch.cat([out_bbox_mins, out_bbox_maxs], dim=1).to(bounding_boxes.dtype),
            format=ta_tensors.BoundingBoxFormat.XYXY,
            canvas_size=canvas_size,
        )
        out_bboxes.append(out_bboxes_i)
    out_bboxes = torch.stack(out_bboxes, dim=0)

    return convert_bounding_box_format(
        out_bboxes,
        old_format=ta_tensors.BoundingBoxFormat.XYXY,
        new_format=format,
        inplace=True,
    ).reshape(original_shape)


@_register_kernel_internal(elastic, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(elastic, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def _elastic_bounding_boxes_dispatch(
    inpt: ta_tensors.BoundingBoxes, displacement: torch.Tensor, **kwargs
) -> ta_tensors.BoundingBoxes:
    output = elastic_bounding_boxes(
        inpt.as_subclass(torch.Tensor),
        format=inpt.format,
        canvas_size=inpt.canvas_size,
        displacement=displacement,
    )
    return ta_tensors.wrap(output, like=inpt)


@_register_kernel_internal(elastic_batch, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def _elastic_batch_bounding_boxes_dispatch(
    inpt: ta_tensors.BatchBoundingBoxes, displacement: torch.Tensor, **kwargs
) -> ta_tensors.BatchBoundingBoxes:
    if not isinstance(displacement, torch.Tensor):
        raise TypeError("Argument displacement should be a Tensor")

    displacement = displacement.repeat_interleave(
        torch.tensor([inpt.get_num_boxes_sample(i) for i in range(inpt.batch_size)], device=displacement.device),
        dim=0,
    )

    output = elastic_batch_bounding_boxes(
        inpt.as_subclass(torch.Tensor),
        format=inpt.format,
        canvas_size=inpt.canvas_size,
        displacement=displacement,
    )
    return ta_tensors.wrap(output, like=inpt)


def elastic_mask(
    mask: torch.Tensor,
    displacement: torch.Tensor,
    fill: _FillTypeJIT = None,
) -> torch.Tensor:
    if mask.ndim < 3:
        mask = mask.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False

    output = elastic_image(
        mask,
        displacement=displacement,
        interpolation=InterpolationMode.NEAREST,
        fill=fill,
    )

    if needs_squeeze:
        output = output.squeeze(0)

    return output


def elastic_batch_masks(
    masks: torch.Tensor,
    displacement: torch.Tensor,
    fill: _FillTypeJIT = None,
) -> torch.Tensor:
    if masks.ndim < 4:
        masks = masks.unsqueeze(1)
        needs_squeeze = True
    else:
        needs_squeeze = False

    output = elastic_batch_images(
        masks,
        displacement=displacement,
        interpolation=InterpolationMode.NEAREST,
        fill=fill,
    )

    if needs_squeeze:
        output = output.squeeze(1)

    return output


@_register_kernel_internal(elastic, ta_tensors.Mask, ta_tensor_wrapper=False)
@_register_kernel_internal(elastic, ta_tensors.BatchMasks, ta_tensor_wrapper=False)
def _elastic_mask_dispatch(
    inpt: ta_tensors.Mask,
    displacement: torch.Tensor,
    fill: _FillTypeJIT = None,
    **kwargs,
) -> ta_tensors.Mask:
    output = elastic_mask(inpt.as_subclass(torch.Tensor), displacement=displacement, fill=fill)
    return ta_tensors.wrap(output, like=inpt)


@_register_kernel_internal(elastic_batch, ta_tensors.BatchMasks, ta_tensor_wrapper=False)
def _elastic_batch_masks_dispatch(
    inpt: ta_tensors.BatchMasks,
    displacement: torch.Tensor,
    fill: _FillTypeJIT = None,
    **kwargs,
) -> ta_tensors.BatchMasks:
    if not isinstance(displacement, torch.Tensor):
        raise TypeError("Argument displacement should be a Tensor")

    displacement = displacement.repeat_interleave(
        torch.tensor([inpt.get_num_masks_sample(i) for i in range(inpt.batch_size)], device=displacement.device),
        dim=0,
    )

    output = elastic_batch_masks(inpt.as_subclass(torch.Tensor), displacement=displacement, fill=fill)
    return ta_tensors.wrap(output, like=inpt)


@_register_kernel_internal(elastic, ta_tensors.Video)
@_register_kernel_internal(elastic, ta_tensors.BatchVideos)
def elastic_video(
    video: torch.Tensor,
    displacement: torch.Tensor,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: _FillTypeJIT = None,
) -> torch.Tensor:
    return elastic_image(image=video, displacement=displacement, interpolation=interpolation, fill=fill)


@_register_kernel_internal(elastic_batch, ta_tensors.BatchVideos)
def elastic_batch_videos(
    videos: torch.Tensor,
    displacement: torch.Tensor,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    fill: _FillTypeJIT = None,
) -> torch.Tensor:
    return elastic_batch_images(images=videos, displacement=displacement, interpolation=interpolation, fill=fill)


def center_crop(inpt: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomCrop` for details."""
    if torch.jit.is_scripting():
        return center_crop_image(inpt, output_size=output_size)

    _log_api_usage_once(center_crop)

    kernel = _get_kernel(center_crop, type(inpt))
    return kernel(inpt, output_size=output_size)


@_register_kernel_internal(center_crop, torch.Tensor)
@_register_kernel_internal(center_crop, ta_tensors.Image)
@_register_kernel_internal(center_crop, ta_tensors.BatchImages)
def center_crop_image(image: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    return TVF.center_crop_image(image=image, output_size=output_size)


def center_crop_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    canvas_size: Tuple[int, int],
    output_size: List[int],
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    return TVF.center_crop_bounding_boxes(
        bounding_boxes=bounding_boxes,
        format=_convert_ta_format_to_tv_format(format),
        canvas_size=canvas_size,
        output_size=output_size,
    )


@_register_kernel_internal(center_crop, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(center_crop, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def _center_crop_bounding_boxes_dispatch(
    inpt: ta_tensors.BoundingBoxes, output_size: List[int]
) -> ta_tensors.BoundingBoxes:
    output, canvas_size = center_crop_bounding_boxes(
        inpt.as_subclass(torch.Tensor),
        format=inpt.format,
        canvas_size=inpt.canvas_size,
        output_size=output_size,
    )
    return ta_tensors.wrap(output, like=inpt, canvas_size=canvas_size)


@_register_kernel_internal(center_crop, ta_tensors.Mask)
@_register_kernel_internal(center_crop, ta_tensors.BatchMasks)
def center_crop_mask(mask: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    if mask.ndim < 3:
        mask = mask.unsqueeze(0)
        needs_squeeze = True
    else:
        needs_squeeze = False

    output = center_crop_image(image=mask, output_size=output_size)

    if needs_squeeze:
        output = output.squeeze(0)

    return output


@_register_kernel_internal(center_crop, ta_tensors.Video)
@_register_kernel_internal(center_crop, ta_tensors.BatchVideos)
def center_crop_video(video: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    return center_crop_image(image=video, output_size=output_size)


def resized_crop(
    inpt: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    antialias: bool = True,
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomResizedCrop` for details."""
    if torch.jit.is_scripting():
        return resized_crop_image(
            inpt,
            top=top,
            left=left,
            height=height,
            width=width,
            size=size,
            interpolation=interpolation,
            antialias=antialias,
        )

    _log_api_usage_once(resized_crop)

    kernel = _get_kernel(resized_crop, type(inpt))
    return kernel(
        inpt,
        top=top,
        left=left,
        height=height,
        width=width,
        size=size,
        interpolation=interpolation,
        antialias=antialias,
    )


@_register_kernel_internal(resized_crop, torch.Tensor)
@_register_kernel_internal(resized_crop, ta_tensors.Image)
@_register_kernel_internal(resized_crop, ta_tensors.BatchImages)
def resized_crop_image(
    image: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    antialias: bool = True,
) -> torch.Tensor:
    return TVF.resized_crop_image(
        image=image,
        top=top,
        left=left,
        height=height,
        width=width,
        size=size,
        interpolation=interpolation,
        antialias=antialias,
    )


def resized_crop_bounding_boxes(
    bounding_boxes: torch.Tensor,
    format: ta_tensors.BoundingBoxFormat,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    return TVF.resized_crop_bounding_boxes(
        bounding_boxes=bounding_boxes,
        format=_convert_ta_format_to_tv_format(format),
        top=top,
        left=left,
        height=height,
        width=width,
        size=size,
    )


@_register_kernel_internal(resized_crop, ta_tensors.BoundingBoxes, ta_tensor_wrapper=False)
@_register_kernel_internal(resized_crop, ta_tensors.BatchBoundingBoxes, ta_tensor_wrapper=False)
def _resized_crop_bounding_boxes_dispatch(
    inpt: ta_tensors.BoundingBoxes,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    **kwargs,
) -> ta_tensors.BoundingBoxes:
    output, canvas_size = resized_crop_bounding_boxes(
        inpt.as_subclass(torch.Tensor),
        format=inpt.format,
        top=top,
        left=left,
        height=height,
        width=width,
        size=size,
    )
    return ta_tensors.wrap(output, like=inpt, canvas_size=canvas_size)


def resized_crop_mask(
    mask: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
) -> torch.Tensor:
    return TVF.resized_crop_mask(
        mask=mask,
        top=top,
        left=left,
        height=height,
        width=width,
        size=size,
    )


@_register_kernel_internal(resized_crop, ta_tensors.Mask, ta_tensor_wrapper=False)
@_register_kernel_internal(resized_crop, ta_tensors.BatchMasks, ta_tensor_wrapper=False)
def _resized_crop_mask_dispatch(
    inpt: Union[ta_tensors.Mask, ta_tensors.BatchMasks],
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    **kwargs,
) -> Union[ta_tensors.Mask, ta_tensors.BatchMasks]:
    output = resized_crop_mask(
        inpt.as_subclass(torch.Tensor),
        top=top,
        left=left,
        height=height,
        width=width,
        size=size,
    )
    return ta_tensors.wrap(output, like=inpt)


@_register_kernel_internal(resized_crop, ta_tensors.Video)
@_register_kernel_internal(resized_crop, ta_tensors.BatchVideos)
def resized_crop_video(
    video: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    antialias: bool = True,
) -> torch.Tensor:
    return resized_crop_image(
        image=video,
        top=top,
        left=left,
        height=height,
        width=width,
        antialias=antialias,
        size=size,
        interpolation=interpolation,
    )


def five_crop(
    inpt: torch.Tensor, size: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """See :class:`~torchaug.transforms.FiveCrop` for details."""
    if torch.jit.is_scripting():
        return five_crop_image(inpt, size=size)

    _log_api_usage_once(five_crop)

    kernel = _get_kernel(five_crop, type(inpt))
    return kernel(inpt, size=size)


@_register_five_ten_crop_kernel_internal(five_crop, torch.Tensor)
@_register_five_ten_crop_kernel_internal(five_crop, ta_tensors.Image)
@_register_five_ten_crop_kernel_internal(five_crop, ta_tensors.BatchImages)
def five_crop_image(
    image: torch.Tensor, size: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return TVF.five_crop_image(image=image, size=size)


@_register_five_ten_crop_kernel_internal(five_crop, ta_tensors.Video)
@_register_five_ten_crop_kernel_internal(five_crop, ta_tensors.BatchVideos)
def five_crop_video(
    video: torch.Tensor, size: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return five_crop_image(image=video, size=size)


def ten_crop(
    inpt: torch.Tensor, size: List[int], vertical_flip: bool = False
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """See :class:`~torchaug.transforms.TenCrop` for details."""
    if torch.jit.is_scripting():
        return ten_crop_image(inpt, size=size, vertical_flip=vertical_flip)

    _log_api_usage_once(ten_crop)

    kernel = _get_kernel(ten_crop, type(inpt))
    return kernel(inpt, size=size, vertical_flip=vertical_flip)


@_register_five_ten_crop_kernel_internal(ten_crop, torch.Tensor)
@_register_five_ten_crop_kernel_internal(ten_crop, ta_tensors.Image)
@_register_five_ten_crop_kernel_internal(ten_crop, ta_tensors.BatchImages)
def ten_crop_image(
    image: torch.Tensor, size: List[int], vertical_flip: bool = False
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return TVF.ten_crop_image(image=image, size=size, vertical_flip=vertical_flip)


@_register_five_ten_crop_kernel_internal(ten_crop, ta_tensors.Video)
@_register_five_ten_crop_kernel_internal(ten_crop, ta_tensors.BatchVideos)
def ten_crop_video(
    video: torch.Tensor, size: List[int], vertical_flip: bool = False
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return ten_crop_image(image=video, size=size, vertical_flip=vertical_flip)
