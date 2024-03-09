from __future__ import annotations

from typing import Any, Optional, Union

import torch

from torchaug import ta_tensors


def _transfer_tensor_on_device(tensor: torch.Tensor, device: torch.device, non_blocking: bool = False) -> torch.Tensor:
    if non_blocking and not tensor.device == device and device.type == "cuda":
        tensor = tensor.pin_memory(device=device)

    tensor = tensor.to(device=device, non_blocking=non_blocking)

    return tensor


def _get_batch_factor(
    factor: Union[int, float, torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    value_check: bool = False,
    min_value: float = 0.0,
    max_value: float = torch.inf,
):
    if isinstance(factor, (int, float)):
        factor = float(factor)
        if factor < min_value or factor > max_value:
            raise ValueError(f"factor should be in the range [{min_value}, {max_value}].")
        factor = torch.tensor([factor], device=device, dtype=dtype).expand(batch_size)
    elif isinstance(factor, torch.Tensor):
        factor = _transfer_tensor_on_device(factor, device, True)
        if value_check and torch.any(torch.logical_or(torch.lt(factor, min_value), torch.gt(factor, max_value))):
            raise ValueError(f"factor should be in the range [{min_value}, {max_value}].")
        if factor.numel() == 1:
            factor = factor.expand(batch_size)
        elif factor.numel() != batch_size:
            raise ValueError("factor tensor should contain 1 or B elements.")
    else:
        raise TypeError("factor should be int or float or Tensor.")

    return factor


def _max_value(dtype: torch.dtype) -> int:
    if dtype == torch.uint8:
        return 255
    elif dtype == torch.int8:
        return 127
    elif dtype == torch.int16:
        return 32767
    elif dtype == torch.int32:
        return 2147483647
    elif dtype == torch.int64:
        return 9223372036854775807
    else:
        # This is only here for completeness. This value is implicitly assumed in a lot of places so changing it is not
        # easy.
        return 1


def is_pure_tensor(inpt: Any) -> bool:
    return isinstance(inpt, torch.Tensor) and not isinstance(inpt, ta_tensors.TATensor)
