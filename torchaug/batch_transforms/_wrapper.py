from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from torchaug.batch_transforms._utils import (_assert_batch_images_tensor,
                                              _assert_batch_videos_tensor)
from torchaug.transforms._transform import VideoBase
from torchaug.transforms._wrapper import Wrapper
from torchaug.utils import VideoFormat, _log_api_usage_once


class BatchImageWrapper(Wrapper):
    """Wrap transforms to handle image data.

    .. note::
            Transforms and their submodules are iterated over:

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
        super().__init__(transforms, inplace=inplace)
        _log_api_usage_once(self)

    def forward(self, imgs: torch.Tensor) -> Tensor:
        """Apply :attr:`~transforms` on the batch of images.

        Call :meth:`torchaug.transforms.Wrapper.forward`.

        Args:
            imgs: The batch of images to transform.

        Returns:
            The transformed images.
        """
        _assert_batch_images_tensor(imgs)

        output = super().forward(imgs)

        return output


class BatchVideoWrapper(Wrapper, VideoBase):
    """Wrap transforms to handle batched video data.

    The transforms are expected to handle the leading dimension differently.

    The batch of videos is expected to be in format [B, C, T, H, W] or [B, T, C, H, W].

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
        same_on_frames: If True, apply the same transform on all the frames, else it
            flattens the batch and temporal dimensions to apply different transformations to each frame.
        video_format: Format of the video. Either ``CTHW`` or ``TCHW``.
    """

    def __init__(
        self,
        transforms: Sequence[nn.Module] | nn.Module,
        inplace: bool = False,
        same_on_frames: bool = True,
        video_format: VideoFormat = VideoFormat.CTHW,
    ) -> None:
        Wrapper.__init__(self, transforms=transforms, inplace=inplace)
        VideoBase.__init__(self, video_format=video_format)

        _log_api_usage_once(self)

        self.same_on_frames = same_on_frames

    @staticmethod
    def _prepare_transform(transform: nn.Module):
        Wrapper._prepare_transform(transform)

        if hasattr(transform, "video_format"):
            transform.video_format = VideoFormat.TCHW

    @staticmethod
    def _prepare_transforms(transforms: list[nn.Module]):
        for transform in transforms:
            BatchVideoWrapper._prepare_transform(transform)
            BatchVideoWrapper._prepare_transforms(list(transform.modules())[1:])

    def forward(self, videos: Tensor) -> Tensor:
        """Apply :attr:`~transforms` on the batch of videos.

        Call :meth:`torchaug.transforms.Wrapper.forward`.

        .. note:: If :attr:`~same_on_frames` is ``False``, the batch and frames
            dimensions are merged.

        Args:
            videos: The batch of videos to transform.

        Returns:
            The transformed videos.
        """
        _assert_batch_videos_tensor(videos)

        if self.time_before_channel:
            b, t, c, h, w = videos.shape
        else:
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)

        if not self.same_on_frames:
            videos = videos.reshape(b * t, c, h, w)

        output = Wrapper.forward(self, videos)

        if not self.same_on_frames:
            output = output.reshape(b, t, *videos.shape[-3:])

        if not self.time_before_channel:
            output = output.permute(0, 2, 1, 3, 4)

        return output

    def __repr__(self):
        transforms_repr = str(self.transforms).replace("\n", "\n    ")
        return (
            f"{self.__class__.__name__}(\n"
            f"    inplace={self.inplace},\n"
            f"    same_on_frames={self.same_on_frames},\n"
            f"    video_format={self.video_format.value},\n"
            f"    transforms={transforms_repr}\n)"
        )
