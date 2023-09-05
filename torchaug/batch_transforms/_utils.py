from typing import List

from torch import Tensor


def _is_tensor_batch_images(x: Tensor) -> bool:
    return x.ndim >= 4


def _is_tensor_batch_videos(x: Tensor) -> bool:
    return x.ndim == 5


def get_batch_channels_height_width(imgs: Tensor) -> List[int]:
    _assert_batch_images_tensor(imgs)
    batch = imgs.shape[0]
    channels, height, width = imgs.shape[-3:]
    return [batch, channels, height, width]


def _assert_batch_channels(img: Tensor, permitted: List[int]) -> None:
    c = get_batch_channels_height_width(img)[1]
    if c not in permitted:
        raise TypeError(
            f"Input image tensor permitted channel values are {permitted}, but found {c}."
        )


def _assert_batch_images_tensor(imgs: Tensor) -> None:
    if not isinstance(imgs, Tensor) or not _is_tensor_batch_images(imgs):
        raise TypeError("Tensor is not a torch batch of images.")


def _assert_batch_videos_tensor(videos: Tensor) -> None:
    if not isinstance(videos, Tensor) or not _is_tensor_batch_videos(videos):
        raise TypeError("Tensor is not a torch batch of videos.")


def _assert_video_or_batch_videos_tensor(videos: Tensor) -> None:
    if not isinstance(videos, Tensor) or videos.ndim not in [4, 5]:
        raise TypeError("Tensor is not a torch video or a batch of videos.")
