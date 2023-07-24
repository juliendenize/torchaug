from typing import Any

import torch

cpu_device = torch.device(type="cpu")


def _assert_tensor(obj: Any) -> None:
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Object should be a tensor. Got {type(obj)}.")


def is_tensor_on_cpu(tensor: torch.Tensor) -> bool:
    return tensor.device == cpu_device
