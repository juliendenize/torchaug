from __future__ import annotations

from math import ceil
from typing import Sequence

import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as F_tv
from torch import Tensor, nn
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import _interpolation_modes_from_int

from torchaug.batch_transforms._utils import _assert_batch_videos_tensor
from torchaug.transforms import VideoBase
from torchaug.utils import VideoFormat, _log_api_usage_once

from ._transform import BatchRandomTransform


class BatchRandomHorizontalFlip(BatchRandomTransform):
    """Horizontally flip the given batch of images randomly with a given probability.

    The batch of images is expected to be of shape [B, ..., C, H, W]
    where ... means an arbitrary number of dimensions.

    Args:
        p: probability of the images being flipped.
        inplace: If True, perform inplace operation to save memory and time.
    """

    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
    ):
        super().__init__(p=p, inplace=inplace)
        _log_api_usage_once(self)

    def apply_transform(self, imgs: Tensor) -> Tensor:
        """Flip the batch of images.

        Args:
            imgs: Batch of images to be horizontally fliped.

        Returns:
            Horizontally fliped batch of images.
        """
        imgs = F_tv.hflip(imgs)
        return imgs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"p={self.p}" f", inplace={self.inplace})"


class BatchRandomResizedCrop(tv_transforms.RandomResizedCrop):
    """Crop a random portion of a batch of images and resize it to a given size.

    The batch shape is expected to be [B, ..., H, W], where ... means an arbitrary number of dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size: Expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale: Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio: lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation: Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
        antialias: Whether to apply antialiasing.
            It only affects bilinear or bicubic modes and it is
            ignored otherwise.
        num_rand_calls: Number of random calls performed to apply augmentations at different orders on sub-batches.
            If -1, B calls are performed. The maximum is 24 = 4!.
    """

    def __init__(
        self,
        size: int | Sequence[int],
        scale: Sequence[float] = (0.08, 1.0),
        ratio: Sequence[float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        num_rand_calls: int = -1,
    ) -> None:
        if isinstance(size, int):
            self.size = [size, size]
        elif isinstance(size, Sequence) and isinstance(size[0], int) and len(size) == 1:
            self.size = [size[0], size[0]]
        elif not isinstance(size, Sequence) or not all(
            [isinstance(s, int) for s in size]
        ):
            raise TypeError(f"size should be a int or a sequence of int. Got {size}.")

        super().__init__(size, scale, ratio, interpolation, antialias)
        _log_api_usage_once(self)

        if not isinstance(num_rand_calls, int) or not num_rand_calls >= -1:
            raise ValueError(
                f"num_rand_calls attribute should be an int superior to -1, {num_rand_calls} given."
            )

        self.num_rand_calls = num_rand_calls

    def single_forward(self, imgs: Tensor) -> Tensor:
        """Perform a random resized crop of same scale and shape for the batch of images.

        Args:
            img: Batch of images to be cropped and resized.

        Returns:
            Cropped and resized batch of images.
        """
        i, j, h, w = self.get_params(imgs, self.scale, self.ratio)
        return F_tv.resized_crop(
            imgs, i, j, h, w, self.size, self.interpolation, antialias=self.antialias
        )

    def forward(self, imgs: Tensor) -> Tensor:
        """Resize and crop the batch of images.

        .. note:: Apply different scaling and cropping based on :attr:`~num_rand_calls`.

        Args:
            imgs: Batch of images to be cropped and resized.

        Returns:
            Randomly cropped and resized batch of images.
        """
        if self.num_rand_calls == -1:
            return torch.stack([self.single_forward(img) for img in imgs])
        elif self.num_rand_calls == 0:
            return imgs
        else:
            # Avoid inplace operation
            output = torch.empty(
                size=(*imgs.shape[:-2], *self.size),
                dtype=imgs.dtype,
                device=imgs.device,
            )

            batch_size = imgs.shape[0]
            num_combination = min(batch_size, self.num_rand_calls)
            num_apply_per_combination = ceil(batch_size / num_combination)

            indices_do_apply = torch.randperm(batch_size, device=imgs.device)

            for i in range(min(batch_size, self.num_rand_calls)):
                indices_combination = indices_do_apply[
                    i * num_apply_per_combination : (i + 1) * num_apply_per_combination
                ]

                imgs_combination = imgs[indices_combination]
                imgs_combination = self.single_forward(imgs_combination)

                output[indices_combination] = imgs_combination

        return output

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", antialias={self.antialias}"
        format_string += f", num_rand_calls={self.num_rand_calls})"
        return format_string


class BatchVideoResize(nn.Module, VideoBase):
    """Resize the input video to the given size. The video is expected to have [B, ..., H, W] shape, where ...
    means an arbitrary number of dimensions.

    Args:
        size: Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).

            .. note::
                In torchscript mode size as single int is not supported,
                use a sequence of length 1: ``[size, ]``.
        interpolation: Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are
            supported.
        max_size : The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e. the
            smaller edge may be shorter than ``size``. This is only supported
            if ``size`` is an int (or a sequence of length 1 in torchscript
            mode).
        antialias: Whether to apply antialiasing.  If ``True``, will apply antialiasing for bilinear or bicubic modes.
            Other mode aren't affected.
        video_format: Format of the video. Either ``CTHW`` or ``TCHW``.
    """

    def __init__(
        self,
        size: int | list[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: int | None = None,
        antialias: bool = True,
        video_format: VideoFormat = VideoFormat.CTHW,
    ):
        nn.Module.__init__(self)
        VideoBase.__init__(self, video_format=video_format)

        _log_api_usage_once(self)

        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}.")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values.")
        self.size = size
        self.max_size = max_size

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, videos: Tensor) -> Tensor:
        """Resize the batch of videos.

        Args:
            videos: The batch of videos to resize.

        Returns:
            The resized videos.
        """
        _assert_batch_videos_tensor(videos)

        if self.time_before_channel:
            b, t, c, h, w = videos.shape
        else:
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)

        videos = videos.reshape(b * t, c, h, w)

        videos = F_tv.resize(
            videos, self.size, self.interpolation, self.max_size, self.antialias
        )

        videos = videos.reshape(b, t, *videos.shape[-3:])

        if not self.time_before_channel:
            videos = videos.permute(0, 2, 1, 3, 4)

        return videos

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size={self.size}, "
            f"interpolation={self.interpolation.value}, "
            f"max_size={self.max_size}, "
            f"antialias={self.antialias}, "
            f"video_format={self.video_format.value})"
        )
