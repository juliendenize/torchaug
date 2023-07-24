from typing import List

import torch


def _is_tensor_a_torch_batch_images(x: torch.Tensor) -> bool:
    return x.ndim >= 4


def get_batched_dimensions(img: torch.Tensor) -> List[int]:
    _assert_batch_images_tensor(img)
    channels = 1 if img.ndim == 3 else img.shape[-3]
    batch = img.shape[0]
    height, width = img.shape[-2:]
    return [batch, channels, height, width]


def _assert_batch_channels(img: torch.Tensor, permitted: List[int]) -> None:
    c = get_batched_dimensions(img)[1]
    if c not in permitted:
        raise TypeError(
            f"Input image tensor permitted channel values are {permitted}, but found {c}."
        )


def _assert_batch_images_tensor(imgs: torch.Tensor) -> None:
    if not _is_tensor_a_torch_batch_images(imgs):
        raise TypeError("Tensor is not a torch batch of images.")
