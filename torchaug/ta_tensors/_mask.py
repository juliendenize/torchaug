from __future__ import annotations

from typing import Any, Optional, Union

import torch

from ._ta_tensor import TATensor


class Mask(TATensor):
    """:class:`torch.Tensor` subclass for segmentation and detection masks.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype: Desired data type. If omitted, will be inferred from
            ``data``.
        device: Desired device. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the mask is constructed on the CPU.
        requires_grad: Whether autograd should record operations. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> Mask:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor.as_subclass(cls)
