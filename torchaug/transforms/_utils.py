from typing import Any

import torch


def _assert_tensor(obj: Any) -> None:
    """Check an object is a tensor. Raise an error if not.

    Args:
        obj (Any): The object to check.
    """
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Object should be a tensor. Got {type(obj)}.")


def _is_tensor_video(x: torch.Tensor) -> bool:
    return x.ndim == 4


def is_tensor_on_cpu(tensor: torch.Tensor) -> bool:
    """Check if tensor is on CPU.

    Args:
        tensor (torch.Tensor): The tensor to check.

    Returns:
        bool: True if the tensor is on CPU.
    """
    return tensor.device.type == "cpu"


def transfer_on_device(
    tensor: torch.Tensor, device: torch.device, non_blocking: bool = False
) -> torch.Tensor:
    """Transfer a tensor to a device.

    Args:
        tensor (torch.Tensor): The tensor to transfer.
        device (torch.device): The device to transfer on.
        non_blocking (bool, optional): Whether to perform asynchronous transfer. Useful for cuda. Defaults to False.

    Returns:
        torch.Tensor: The tensor transfered on the device.
    """
    if non_blocking and not tensor.device == device and device.type == "cuda":
        tensor = tensor.pin_memory(device=device)

    tensor = tensor.to(device=device, non_blocking=non_blocking)

    return tensor


def _assert_video_tensor(video: torch.Tensor) -> None:
    if not isinstance(video, torch.Tensor) or not _is_tensor_video(video):
        raise TypeError("Tensor is not a torch batch of videos.")
