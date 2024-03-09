from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
from torch._C import device, dtype

from ._ta_tensor import TATensor


class BatchImages(TATensor):
    """:class:`torch.Tensor` subclass for batch of images.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype: Desired data type. If omitted, will be inferred from
            ``data``.
        device: Desired device. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the image is constructed on the CPU.
        requires_grad: Whether autograd should record operations. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[dtype] = None,
        device: Optional[Union[device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> BatchImages:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

        if tensor.ndim < 4:
            raise ValueError

        return tensor.as_subclass(cls)

    @classmethod
    def cat(cls, images_batches: Sequence[BatchImages]) -> BatchImages:
        """Concatenates the sequence of images batches into a single batch."""
        attrs = [
            "requires_grad",
            "device",
            "dtype",
        ]

        for batch_images in images_batches:
            if not isinstance(batch_images, BatchImages):
                raise ValueError("All batches must be of type BatchImages.")
            for attr in attrs:
                if getattr(batch_images, attr) != getattr(images_batches[0], attr):
                    raise ValueError(f"All batches of images must have the same {attr} attribute.")

        return cls(torch.cat([images.data for images in images_batches], 0))

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr()
