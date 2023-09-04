from __future__ import annotations

import numbers
from typing import Any, Sequence

import torch
from torch import Tensor, nn


def _assert_channels(img: Tensor, permitted: list[int]) -> None:
    c = get_channels_height_width(img)[0]
    if c not in permitted:
        raise TypeError(
            f"Input image tensor permitted channel values are {permitted}, but found {c}"
        )


def _assert_image_tensor(img: Tensor) -> None:
    if not _is_tensor_torch_image(img):
        raise TypeError("Tensor is not a torch image.")


def _assert_module_or_list_of_modules(collection: Any):
    if not isinstance(collection, nn.Module) and (
        not isinstance(collection, Sequence)
        or not all([isinstance(obj, nn.Module) for obj in collection])
    ):
        raise TypeError("collection should be a module or a list of modules.")


def _assert_tensor(obj: Any) -> None:
    """Check an object is a tensor. Raise an error if not.

    Args:
        obj (Any): The object to check.
    """
    if not isinstance(obj, Tensor):
        raise TypeError(f"Object should be a tensor. Got {type(obj)}.")


def _assert_video_tensor(video: Tensor) -> None:
    if not isinstance(video, Tensor) or not _is_tensor_video(video):
        raise TypeError("Tensor is not a torch video.")


def _check_input(
    value: numbers.Number,
    name: str,
    center: float = 1.0,
    bound: tuple[float] = (0, float("inf")),
    clip_first_on_zero: bool = True,
):
    # Adapted from Torchvision.
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError(f"If {name} is a single number, it must be non negative.")
        value = [center - float(value), center + float(value)]
        if clip_first_on_zero:
            value[0] = max(value[0], 0.0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        value = [float(value[0]), float(value[1])]
    else:
        raise TypeError(
            f"{name} should be a single number or a list/tuple with length 2."
        )

    if not bound[0] <= value[0] <= value[1] <= bound[1]:
        raise ValueError(f"{name} values should be between {bound}, but got {value}.")

    if value[0] == value[1] == center:
        return None
    else:
        return tuple(value)


def _is_tensor_torch_image(x: Tensor) -> bool:
    return x.ndim >= 3


def _is_tensor_video(x: Tensor) -> bool:
    return x.ndim == 4


def is_tensor_on_cpu(tensor: Tensor) -> bool:
    """Check if tensor is on CPU.

    Args:
        tensor (Tensor): The tensor to check.

    Returns:
        bool: True if the tensor is on CPU.
    """
    return tensor.device.type == "cpu"


def get_channels_height_width(img: Tensor) -> list[int]:
    _assert_image_tensor(img)
    return list(img.shape[-3:])


def transfer_tensor_on_device(
    tensor: Tensor, device: torch.device, non_blocking: bool = False
) -> Tensor:
    """Transfer a tensor to a device.

    Args:
        tensor (Tensor): The tensor to transfer.
        device (torch.device): The device to transfer on.
        non_blocking (bool, optional): Whether to perform asynchronous transfer. Useful for cuda. Defaults to False.

    Returns:
        Tensor: The tensor transferred on the device.
    """
    if non_blocking and not tensor.device == device and device.type == "cuda":
        tensor = tensor.pin_memory(device=device)

    tensor = tensor.to(device=device, non_blocking=non_blocking)

    return tensor
