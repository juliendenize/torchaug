from __future__ import annotations

import torch


def _transfer_tensor_on_device(
    tensor: torch.Tensor, device: torch.device, non_blocking: bool = False
) -> torch.Tensor:
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


def _get_positive_batch_factor(
    factor: int | float | torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
    value_check: bool = False,
):
    if isinstance(factor, (int, float)):
        if factor < 0.0:
            raise ValueError("factor is not non-negative.")
        factor = torch.tensor(factor, device=device, dtype=dtype).expand(batch_size)
    elif isinstance(factor, torch.Tensor):
        factor = _transfer_tensor_on_device(factor, device, True)
        if value_check and not torch.all(torch.ge(factor, 0)):
            raise ValueError("factor is not non-negative.")
        if factor.numel() == 1:
            factor = factor.expand(batch_size)
        elif factor.numel() != batch_size:
            raise ValueError("factor tensor should contain 1 or B elements.")
    else:
        raise TypeError("factor should be int or float or Tensor.")

    return factor


def _max_value(dtype: torch.dtype) -> int:
    """Get the maximum value of a torch dtype.

    From Torchvision.

    Args:
        dtype: the dtype to analyze.

    Returns:
        The maximum value for the dtype.
    """
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
