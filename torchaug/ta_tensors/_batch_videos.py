from __future__ import annotations
from typing import Any
from torch._C import device, dtype
from torchvision.tv_tensors import TVTensor


class BatchVideos(TVTensor):
    """:class:`torch.Tensor` subclass for batch of videos.

    Args:
        data (tensor-like): Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype (torch.dtype, optional): Desired data type. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the video is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    def __new__(
        cls,
        data: Any,
        *,
        dtype: dtype | None = None,
        device: device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> BatchVideos:
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        if data.ndim < 5:
            raise ValueError
        return tensor.as_subclass(cls)

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr()
