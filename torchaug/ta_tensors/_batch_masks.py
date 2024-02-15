from __future__ import annotations
import torch
from typing import Any
from torchvision.tv_tensors import TVTensor


class BatchMasks(TVTensor):
    """:class:`torch.Tensor` subclass for batch of segmentation and detection masks.

    Args:
        data (tensor-like, PIL.Image.Image): Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype (torch.dtype, optional): Desired data type. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the mask is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    def __new__(
        cls,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> BatchMasks:
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )

        if tensor.ndim < 3:
            raise ValueError

        return tensor.as_subclass(cls)

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr()
