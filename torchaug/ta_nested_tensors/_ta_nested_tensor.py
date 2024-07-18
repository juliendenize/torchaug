from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from torch import Size, Tensor, memory_format
from torch._prims_common import DeviceLikeType


def _assert_is_ta_nested_tensor(obj: object) -> None:
    if not isinstance(obj, TANestedTensor):
        raise TypeError(f"Expected a TANestedTensor, but got {type(obj)}")


def _assert_is_tensor(obj: object) -> None:
    if not isinstance(obj, Tensor):
        raise TypeError(f"Expected a Tensor, but got {type(obj)}")


class TANestedTensor:
    """Implement Nested Tensor for PyTorch."""

    def __init__(self, tensors: Sequence[Tensor]):
        for tensor in tensors:
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Expected a Tensor, but got {type(tensor)}")
            elif tensor.device != tensors[0].device:
                raise ValueError("All tensors must be on the same device")
            elif tensor.dtype != tensors[0].dtype:
                raise ValueError("All tensors must have the same dtype")

        self.tensors = list(tensors)

    @property
    def device(self) -> torch.device:
        return self.tensors[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.tensors[0].dtype

    def __deepcopy__(self, memo):
        return TANestedTensor([tensor.__deepcopy__(memo) for tensor in self.tensors])

    def __copy__(self):
        return TANestedTensor([tensor.clone() for tensor in self.tensors])

    def __contains__(self, tensor: Tensor) -> bool:
        return tensor in self.tensors

    def __repr__(self) -> str:
        return str(self.tensors)

    def __add__(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        return self.add(other)

    def __mul__(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        return self.mul(other)

    def __truediv__(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        return self.div(other)

    def __floordiv__(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        return self.floor_divide(other)

    def __mod__(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        return self.remainder(other)

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, index: int) -> Tensor:
        return self.tensors[index]

    def __setitem__(self, index: int, tensor: Tensor) -> None:
        _assert_is_tensor(tensor)
        if tensor.device != self.device:
            raise ValueError("tensor must be on the same device")
        elif tensor.dtype != self.dtype:
            raise ValueError("tensor must have the same dtype")
        elif index < 0 or index >= len(self.tensors):
            raise IndexError("index out of range")
        self.tensors[index] = tensor

    def __iter__(self):
        return iter(self.tensors)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TANestedTensor):
            return False
        return all(t1.equal(t2) for t1, t2 in zip(self.tensors, other.tensors))

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def count(self, tensor: Tensor) -> int:
        _assert_is_tensor(tensor)
        return self.tensors.count(tensor)

    def index(self, tensor: Tensor) -> int:
        _assert_is_tensor(tensor)
        return self.tensors.index(tensor)

    def clone(self, memory_format: Optional[memory_format] = None) -> "TANestedTensor":
        return TANestedTensor([tensor.clone(memory_format) for tensor in self.tensors])

    def add(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        if isinstance(other, (float, int)):
            return TANestedTensor([tensor + other for tensor in self.tensors])

        _assert_is_ta_nested_tensor(other)
        return TANestedTensor([t1 + t2 for t1, t2 in zip(self.tensors, other.tensors, strict=True)])

    def add_(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        if isinstance(other, (float, int)):
            for tensor in self.tensors:
                tensor.add_(other)
            return self

        _assert_is_ta_nested_tensor(other)
        for t1, t2 in zip(self.tensors, other.tensors, strict=True):
            t1.add_(t2)
        return self

    def mul(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        if isinstance(other, (float, int)):
            return TANestedTensor([tensor.mul(other) for tensor in self.tensors])

        _assert_is_ta_nested_tensor(other)
        return TANestedTensor([t1.mul(t2) for t1, t2 in zip(self.tensors, other.tensors, strict=True)])

    def mul_(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        if isinstance(other, (float, int)):
            for tensor in self.tensors:
                tensor.mul_(other)
            return self

        _assert_is_ta_nested_tensor(other)
        for t1, t2 in zip(self.tensors, other.tensors, strict=True):
            t1.mul_(t2)
        return self

    def div(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        if isinstance(other, (float, int)):
            return TANestedTensor([tensor.div(other) for tensor in self.tensors])
        _assert_is_ta_nested_tensor(other)
        return TANestedTensor([t1.div(t2) for t1, t2 in zip(self.tensors, other.tensors, strict=True)])

    def div_(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        if isinstance(other, (float, int)):
            for tensor in self.tensors:
                tensor.div_(other)
            return self

        _assert_is_ta_nested_tensor(other)
        for t1, t2 in zip(self.tensors, other.tensors, strict=True):
            t1.div_(t2)
        return self

    def floor_divide(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        if isinstance(other, (float, int)):
            return TANestedTensor([tensor.floor_divide(other) for tensor in self.tensors])

        _assert_is_ta_nested_tensor(other)
        return TANestedTensor([t1.floor_divide(t2) for t1, t2 in zip(self.tensors, other.tensors, strict=True)])

    def floor_divide_(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        if isinstance(other, (float, int)):
            for tensor in self.tensors:
                tensor.floor_divide_(other)
            return self

        _assert_is_ta_nested_tensor(other)
        for t1, t2 in zip(self.tensors, other.tensors, strict=True):
            t1.floor_divide_(t2)
        return self

    def remainder(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        if isinstance(other, (float, int)):
            return TANestedTensor([tensor.remainder(other) for tensor in self.tensors])

        _assert_is_ta_nested_tensor(other)
        return TANestedTensor([t1.remainder(t2) for t1, t2 in zip(self.tensors, other.tensors, strict=True)])

    def remainder_(self, other: Union["TANestedTensor", float, int]) -> "TANestedTensor":
        if isinstance(other, (float, int)):
            for tensor in self.tensors:
                tensor.remainder_(other)
            return self

        _assert_is_ta_nested_tensor(other)
        for t1, t2 in zip(self.tensors, other.tensors, strict=True):
            t1.remainder_(t2)
        return self

    def shape(self) -> List[Size]:
        return [tensor.shape for tensor in self.tensors]

    def size(self, dim: Optional[int] = None) -> List[Size]:
        return [tensor.size() for tensor in self.tensors]

    def to(
        self,
        other: Tensor,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: Optional[torch.memory_format] = None,
    ) -> "TANestedTensor":
        return TANestedTensor([tensor.to(other, non_blocking, copy, memory_format) for tensor in self.tensors])

    def to_tensor_list(self) -> Sequence[Tensor]:
        return self.tensors

    def tolist(self) -> Sequence:
        return [tensor.tolist() for tensor in self.tensors]

    def pin_memory(self, device: Optional[Optional[DeviceLikeType]] = None) -> "TANestedTensor":
        return TANestedTensor([tensor.pin_memory(device=device) for tensor in self.tensors])
