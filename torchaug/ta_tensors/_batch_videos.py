from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
from torch._C import device, dtype

from ._ta_tensor import TATensor


class BatchVideos(TATensor):
    """:class:`torch.Tensor` subclass for batch of videos.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype: Desired data type. If omitted, will be inferred from
            ``data``.
        device: Desired device. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the video is constructed on the CPU.
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
    ) -> BatchVideos:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        if data.ndim < 5:
            raise ValueError
        return tensor.as_subclass(cls)

    @classmethod
    def cat(cls, video_batches: Sequence[BatchVideos]) -> BatchVideos:
        """Concatenates the sequence of images batches into a single batch."""
        attrs = [
            "requires_grad",
            "device",
            "dtype",
        ]

        for batch_videos in video_batches:
            if not isinstance(batch_videos, BatchVideos):
                raise ValueError("All batches must be of type BatchVideos.")
            for attr in attrs:
                if getattr(batch_videos, attr) != getattr(video_batches[0], attr):
                    raise ValueError(f"All batches of images must have the same {attr} attribute.")

        return cls(torch.cat([images.data for images in video_batches], 0))

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr()
