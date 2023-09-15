from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torchvision.transforms._functional_tensor import _assert_image_tensor

from torchaug.transforms._utils import (_assert_module_or_list_of_modules,
                                        _assert_tensor, _assert_video_tensor)
from torchaug.utils import VideoFormat, _log_api_usage_once

from ._transform import VideoBase


class Wrapper(nn.Module):
    """Wrap transforms to handle tensor data.

    .. note::

            Iterates through transforms and their submodules.

            - If ``inplace`` attribute is found, it is set to ``True``,
              ``inplace`` is handled at the wrapper level.


    .. note::
        If a transform makes a copy, the resulting tensor will not share the same
        underlying storage even if ``inplace`` is set to ``True``.

    Args:
        transforms: A list of transform modules.
        inplace: Whether to perform the transforms inplace. If a transform makes a copy,
            the resulting tensor will not share the same underlying storage.
    """

    def __init__(
        self, transforms: list[nn.Module] | nn.Module, inplace: bool = False
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        _assert_module_or_list_of_modules(transforms)

        if isinstance(transforms, nn.Module):
            transforms = [transforms]

        self._prepare_transforms(transforms)
        self.transforms = nn.ModuleList(transforms)

        self.inplace = inplace

    @staticmethod
    def _prepare_transform(transform: nn.Module):
        if hasattr(transform, "inplace"):
            transform.inplace = True

    @staticmethod
    def _prepare_transforms(transforms: list[nn.Module]):
        for transform in transforms:
            Wrapper._prepare_transform(transform)
            Wrapper._prepare_transforms(list(transform.modules())[1:])

    def forward(self, tensor: torch.Tensor) -> Tensor:
        """Apply :attr:`~transforms` on the tensor.

        If :attr:`~inplace` is ``True``, clone the tensor.

        Args:
            tensor: The tensor to transform.

        Returns:
            The transformed tensor.
        """
        _assert_tensor(tensor)

        output = tensor if self.inplace else tensor.clone()

        for transform in self.transforms:
            output: Tensor = transform(output.contiguous())

        return output.contiguous()

    def __repr__(self):
        transforms_repr = str(self.transforms).replace("\n", "\n    ")
        return (
            f"{self.__class__.__name__}(\n"
            f"    inplace={self.inplace},\n"
            f"    transforms={transforms_repr}\n)"
        )


class ImageWrapper(Wrapper):
    """Wrap transforms to handle image data.

    .. note::

            Iterates through transforms and their submodules.

            - If ``inplace`` attribute is found, it is set to ``True``,
              ``inplace`` is handled at the wrapper level.


    .. note::
        If a transform makes a copy, the resulting tensor will not share the same
        underlying storage even if ``inplace`` is set to ``True``.

    Args:
        transforms: A list of transform modules.
        inplace: Whether to perform the transforms inplace.
    """

    def __init__(
        self, transforms: Sequence[nn.Module] | nn.Module, inplace: bool = False
    ) -> None:
        super().__init__(transforms=transforms, inplace=inplace)
        _log_api_usage_once(self)

    def forward(self, img: torch.Tensor) -> Tensor:
        """Apply :attr:`~transforms` on the image.

        Call :meth:`Wrapper.forward`.

        Args:
            image: The image to transform.

        Returns:
            The transformed image.
        """
        _assert_image_tensor(img)

        output = super().forward(img)

        return output


class VideoWrapper(Wrapper, VideoBase):
    """Wrap transforms to handle video data.

    If the frames should be augmented differently, the transform must
    handle the leading dimension differently. The video is expected to
    be in format [C, T, H, W] or [T, C, H, W].

    .. note::

            Iterates through transforms and their submodules:

            - If ``inplace`` attribute is found, it is set to ``True``,
              ``inplace`` is handled at the wrapper level.

            - If ``video_format`` attribute is found, it is set to ``TCHW``,
              ``video_format`` is handled at the wrapper level.


    .. note::
        If ``video_format`` is ``CTHW``, a copy might occur even if ``inplace`` is set to ``True``.

    .. note::
        If a transform makes a copy, the resulting tensor will not share the same
        underlying storage even if ``inplace`` is set to ``True``.


    Args:
        transforms: A list of transform modules.
        inplace: Whether to perform the transforms inplace.
        video_format: Format of the video. Either ``CTHW`` or ``TCHW``.
    """

    def __init__(
        self,
        transforms: Sequence[nn.Module] | nn.Module,
        inplace: bool = False,
        video_format: VideoFormat = VideoFormat.CTHW,
    ) -> None:
        Wrapper.__init__(self, transforms=transforms, inplace=inplace)
        VideoBase.__init__(self, video_format=video_format)

        _log_api_usage_once(self)

    @staticmethod
    def _prepare_transform(transform: nn.Module):
        Wrapper._prepare_transform(transform)

        if hasattr(transform, "video_format"):
            transform.video_format = VideoFormat.TCHW

    @staticmethod
    def _prepare_transforms(transforms: list[nn.Module]):
        for transform in transforms:
            VideoWrapper._prepare_transform(transform)
            VideoWrapper._prepare_transforms(list(transform.modules())[1:])

    def forward(self, video: Tensor) -> Tensor:
        """Apply :attr:`~transforms` on the video.

        Call :meth:`Wrapper.forward`.

        Args:
            video: The video to transform.

        Returns:
            The transformed video.
        """
        _assert_video_tensor(video)

        if not self.time_before_channel:
            video = video.permute(1, 0, 2, 3)

        output = Wrapper.forward(self, video)

        if not self.time_before_channel:
            output = output.permute(1, 0, 2, 3)

        return output

    def __repr__(self):
        transforms_repr = str(self.transforms).replace("\n", "\n    ")
        return (
            f"{self.__class__.__name__}(\n"
            f"    inplace={self.inplace},\n"
            f"    video_format={self.video_format.value},\n"
            f"    transforms={transforms_repr}\n)"
        )
