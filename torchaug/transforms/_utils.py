from typing import Any

import torch

cpu_device = torch.device(type="cpu")


def _assert_tensor(obj: Any) -> None:
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Object should be a tensor. Got {type(obj)}.")


def _is_tensor_video(x: torch.Tensor) -> bool:
    return x.ndim == 4


def is_tensor_on_cpu(tensor: torch.Tensor) -> bool:
    return tensor.device == cpu_device


def _assert_video_tensor(video: torch.Tensor) -> None:
    if not isinstance(video, torch.Tensor) or not _is_tensor_video(video):
        raise TypeError("Tensor is not a torch batch of videos.")
