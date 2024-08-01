# ==================================
# Copyright: CEA-LIST/DIASI/SIALV/
# Author : Torchaug Developers
# License: CECILL-C
# ==================================

from __future__ import annotations

from abc import ABC
from numbers import Number
from typing import (
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import torch
from torch import Size, Tensor, memory_format
from torch._prims_common import DeviceLikeType
from torch.types import (
    _complex,
)
from typing_extensions import Self

from torchaug import ta_tensors
from torchaug.ta_tensors import (
    BatchBoundingBoxes,
    BatchImages,
    BatchLabels,
    BatchMasks,
    BatchVideos,
    BoundingBoxes,
    Image,
    Labels,
    Mask,
    TATensor,
    Video,
    set_return_type,
)


T = TypeVar("T", Image, Video, Labels, BoundingBoxes, Mask, torch.Tensor)
BATCH_T = TypeVar("BATCH_T", BatchImages, BatchBoundingBoxes, BatchLabels, BatchMasks, BatchVideos, torch.Tensor)


def _assert_is_ta_nested_tensor(obj: object) -> None:
    if not isinstance(obj, TANestedTensors):
        raise TypeError(f"Expected a TANestedTensor, but got {type(obj)}")


def _assert_is_tensor(obj: object) -> None:
    if not isinstance(obj, Tensor):
        raise TypeError(f"Expected a Tensor, but got {type(obj)}")


class TANestedTensors(ABC, Generic[T, BATCH_T]):
    """TANestedTensors is a generic class to handle nested tensors.

    A nested tensor is a list of tensors with the same device, dtype, and number of dimensions.

    Args:
        tensors: The sequence of tensors to nest.

    Returns:
        _description_
    """

    tensors_type: Type[T]
    batch_tensors_type: Type[BATCH_T]
    tensors: List[T]

    def __init__(self, tensors: Sequence[T]) -> None:
        self.tensors = list(tensors)
        self._check_tensors()

    def _check_tensors(self) -> None:
        first_tensor = self.tensors[0]
        for tensor in self.tensors:
            if tensor.device != first_tensor.device:
                raise ValueError("All tensors must be on the same device")
            elif tensor.dtype != first_tensor.dtype:
                raise ValueError("All tensors must have the same dtype")
            elif type(tensor) is not self.tensors_type:  # noqa: E721
                raise TypeError(f"All tensors must have the type {self.tensors_type}")
            elif tensor.ndim != first_tensor.ndim:
                raise ValueError("All tensors must have the same number of dimensions")

    @property
    def shapes(self) -> List[Size]:
        """Return the shapes of the tensors."""
        return [tensor.size() for tensor in self.tensors]

    @property
    def batch_size(self) -> int:
        """Return the batch size of the nested tensors."""
        return len(self.tensors)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the nested tensors."""
        return self.tensors[0].ndim + 1

    @property
    def device(self) -> torch.device:
        """Return the device of the nested tensors."""
        return self.tensors[0].device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the nested tensors."""
        return self.tensors[0].dtype

    @property
    def _return_type(self) -> str:
        return "TATensor" if issubclass(self.tensors_type, TATensor) else "Tensor"

    def __deepcopy__(self, memo) -> Self:
        return self.__class__([tensor.__deepcopy__(memo) for tensor in self.tensors])

    def __copy__(self) -> Self:
        return self.__class__([tensor.clone() for tensor in self.tensors])

    def __contains__(self, tensor: Tensor) -> bool:
        return tensor in self.tensors

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n\ttensors={self.tensors},\n\tbatch_size={self.batch_size}, "
            f"shapes={self.shapes}, device={self.device}, dtype={self.dtype}\n)"
        )

    def __add__(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        return self.add(other)

    def __sub__(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        return self.sub(other)

    def __mul__(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        return self.mul(other)

    def __truediv__(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        return self.div(other)

    def __floordiv__(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        return self.floor_divide(other)

    def __mod__(self, other: Union[Self, T, Tensor, Number, _complex]) -> Self:
        return self.remainder(other)

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, index) -> Self:
        if isinstance(index, Tensor):
            index = index.tolist()

        if isinstance(index, Sequence):
            list_tensors = [self.tensors[i] for i in index]
        else:
            tensors = self.tensors[index]
            if not isinstance(tensors, (List)):
                list_tensors = [tensors]
            else:
                list_tensors = tensors

        return self.__class__(list_tensors)

    def __setitem__(self, index, value) -> None:
        if isinstance(index, Tensor):
            index = index.tolist()
        if isinstance(index, Sequence):
            if len(value) != len(index):
                raise ValueError("The length of the value must be equal to the length of the index")
            for i, v in zip(index, value):
                self.tensors[i] = v
        else:
            self.tensors[index] = value
        self._check_tensors()

    def __iter__(self) -> Iterator[T]:
        return iter(self.tensors)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TANestedTensors):
            return False
        return all(t1.equal(t2) for t1, t2 in zip(self.tensors, other.tensors))

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def count(self, tensor: T) -> int:
        """Return the number of occurrences of a tensor in the nested tensors."""
        _assert_is_tensor(tensor)
        return self.tensors.count(tensor)

    def index(self, tensor: T) -> int:
        """Return the index of a tensor in the nested tensors."""
        _assert_is_tensor(tensor)
        return self.tensors.index(tensor)

    def clone(self, memory_format: Optional[memory_format] = None) -> Self:
        """Return a deep copy of the nested tensors."""
        with set_return_type(self._return_type):
            return self.__class__([tensor.clone(memory_format=memory_format) for tensor in self.tensors])

    def add(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        """Add a tensor or a scalar to the nested tensors."""
        with set_return_type(self._return_type):
            if isinstance(other, (self.tensors_type, Tensor, Number, _complex, torch.SymInt, torch.SymFloat)):
                return self.__class__([tensor + other for tensor in self.tensors])
            elif not isinstance(other, self.__class__):
                raise TypeError(
                    (
                        f"Expected a {self.__class__.__name__} or {self.tensors_type} or float or int, "
                        f"but got {type(other)}"
                    )
                )
            elif len(other.tensors) != len(self.tensors):
                raise ValueError("The number of tensors must be the same")

            return self.__class__([t1 + t2 for t1, t2 in zip(self.tensors, other.tensors)])

    def add_(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        """Add a tensor or a scalar to the nested tensors in-place."""
        with set_return_type(self._return_type):
            if isinstance(other, (self.tensors_type, Tensor, Number, _complex, torch.SymInt, torch.SymFloat)):
                for tensor in self.tensors:
                    tensor.add_(other)
                return self
            elif not isinstance(other, self.__class__):
                raise TypeError(
                    (
                        f"Expected a {self.__class__.__name__} or {self.tensors_type} or float or int, "
                        f"but got {type(other)}"
                    )
                )
            elif len(other.tensors) != len(self.tensors):
                raise ValueError("The number of tensors must be the same")

            for t1, t2 in zip(self.tensors, other.tensors):
                t1.add_(t2)
            return self

    def sub(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        """Subtract a tensor or a scalar to the nested tensors."""
        with set_return_type(self._return_type):
            if isinstance(other, (self.tensors_type, Tensor, Number, _complex, torch.SymInt, torch.SymFloat)):
                return self.__class__([tensor.sub(other) for tensor in self.tensors])
            elif not isinstance(other, self.__class__):
                raise TypeError(
                    (
                        f"Expected a {self.__class__.__name__} or {self.tensors_type} or float or int, "
                        f"but got {type(other)}"
                    )
                )
            elif len(other.tensors) != len(self.tensors):
                raise ValueError("The number of tensors must be the same")

            return self.__class__([t1.sub(t2) for t1, t2 in zip(self.tensors, other.tensors)])

    def sub_(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        """Subtract a tensor or a scalar to the nested tensors in-place."""
        with set_return_type(self._return_type):
            if isinstance(other, (self.tensors_type, Tensor, Number, _complex, torch.SymInt, torch.SymFloat)):
                for tensor in self.tensors:
                    tensor.sub_(other)
                return self
            elif not isinstance(other, self.__class__):
                raise TypeError(
                    (
                        f"Expected a {self.__class__.__name__} or {self.tensors_type} or float or int, "
                        f"but got {type(other)}"
                    )
                )
            elif len(other.tensors) != len(self.tensors):
                raise ValueError("The number of tensors must be the same")

            for t1, t2 in zip(self.tensors, other.tensors):
                t1.sub_(t2)
            return self

    def mul(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        """Multiply a tensor or a scalar to the nested tensors."""
        with set_return_type(self._return_type):
            if isinstance(other, (self.tensors_type, Tensor, Number, _complex, torch.SymInt, torch.SymFloat)):
                return self.__class__([tensor.mul(other) for tensor in self.tensors])
            elif not isinstance(other, self.__class__):
                raise TypeError(
                    (
                        f"Expected a {self.__class__.__name__} or {self.tensors_type} or float or int, "
                        f"but got {type(other)}"
                    )
                )
            elif len(other.tensors) != len(self.tensors):
                raise ValueError("The number of tensors must be the same")

            _assert_is_ta_nested_tensor(other)
            return self.__class__([t1.mul(t2) for t1, t2 in zip(self.tensors, other.tensors)])

    def mul_(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        """Multiply a tensor or a scalar to the nested tensors in-place."""
        with set_return_type(self._return_type):
            if isinstance(other, (self.tensors_type, Tensor, Number, _complex, torch.SymInt, torch.SymFloat)):
                for tensor in self.tensors:
                    tensor.mul_(other)
                return self
            elif not isinstance(other, self.__class__):
                raise TypeError(
                    (
                        f"Expected a {self.__class__.__name__} or {self.tensors_type} or float or int, "
                        f"but got {type(other)}"
                    )
                )
            elif len(other.tensors) != len(self.tensors):
                raise ValueError("The number of tensors must be the same")

            for t1, t2 in zip(self.tensors, other.tensors):
                t1.mul_(t2)
            return self

    def div(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        """Divide the nested tensors by a tensor or a scalar."""
        with set_return_type(self._return_type):
            if isinstance(other, (self.tensors_type, Tensor, Number, _complex, torch.SymInt, torch.SymFloat)):
                return self.__class__([tensor.div(other) for tensor in self.tensors])
            elif not isinstance(other, self.__class__):
                raise TypeError(
                    (
                        f"Expected a {self.__class__.__name__} or {self.tensors_type} or float or int, "
                        f"but got {type(other)}"
                    )
                )
            elif len(other.tensors) != len(self.tensors):
                raise ValueError("The number of tensors must be the same")

            return self.__class__([t1.div(t2) for t1, t2 in zip(self.tensors, other.tensors)])

    def div_(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        """Divide the nested tensors by a tensor or a scalar in-place."""
        with set_return_type(self._return_type):
            if isinstance(other, (self.tensors_type, Tensor, Number, _complex, torch.SymInt, torch.SymFloat)):
                for tensor in self.tensors:
                    tensor.div_(other)
                return self
            elif not isinstance(other, self.__class__):
                raise TypeError(
                    (
                        f"Expected a {self.__class__.__name__} or {self.tensors_type} or float or int, "
                        f"but got {type(other)}"
                    )
                )
            elif len(other.tensors) != len(self.tensors):
                raise ValueError("The number of tensors must be the same")

            for t1, t2 in zip(self.tensors, other.tensors):
                t1.div_(t2)
            return self

    def floor_divide(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        """Floor divide the nested tensors by a tensor or a scalar."""
        with set_return_type(self._return_type):
            if isinstance(other, (self.tensors_type, Tensor, Number, _complex, torch.SymInt, torch.SymFloat)):
                return self.__class__([tensor.floor_divide(other) for tensor in self.tensors])
            elif not isinstance(other, self.__class__):
                raise TypeError(
                    (
                        f"Expected a {self.__class__.__name__} or {self.tensors_type} or float or int, "
                        f"but got {type(other)}"
                    )
                )
            elif len(other.tensors) != len(self.tensors):
                raise ValueError("The number of tensors must be the same")

            return self.__class__([t1.floor_divide(t2) for t1, t2 in zip(self.tensors, other.tensors)])

    def floor_divide_(self, other: Union[Self, T, Tensor, Number, _complex, torch.SymInt, torch.SymFloat]) -> Self:
        """Floor divide the nested tensors by a tensor or a scalar in-place."""
        with set_return_type(self._return_type):
            if isinstance(other, (self.tensors_type, Tensor, Number, _complex, torch.SymInt, torch.SymFloat)):
                for tensor in self.tensors:
                    tensor.floor_divide_(other)
                return self
            elif not isinstance(other, self.__class__):
                raise TypeError(
                    (
                        f"Expected a {self.__class__.__name__} or {self.tensors_type} or float or int, "
                        f"but got {type(other)}"
                    )
                )
            elif len(other.tensors) != len(self.tensors):
                raise ValueError("The number of tensors must be the same")

            for t1, t2 in zip(self.tensors, other.tensors):
                t1.floor_divide_(t2)
            return self

    def remainder(self, other: Union[Self, T, Tensor, Number, _complex]) -> Self:
        """Return the remainder of the nested tensors by a tensor or a scalar."""
        with set_return_type(self._return_type):
            if isinstance(other, (Tensor, Number, _complex)):
                return self.__class__([tensor.remainder(other) for tensor in self.tensors])
            elif not isinstance(other, self.__class__):
                raise TypeError(
                    (
                        f"Expected a {self.__class__.__name__} or {self.tensors_type} or float or int, "
                        f"but got {type(other)}"
                    )
                )
            elif len(other.tensors) != len(self.tensors):
                raise ValueError("The number of tensors must be the same")

            return self.__class__([t1.remainder(t2) for t1, t2 in zip(self.tensors, other.tensors)])

    def remainder_(self, other: Union[Self, T, Tensor, Number, _complex]) -> Self:
        """Return the remainder of the nested tensors by a tensor or a scalar in-place."""
        with set_return_type(self._return_type):
            if isinstance(other, (Tensor, Number, _complex)):
                for tensor in self.tensors:
                    tensor.remainder_(other)
                return self
            elif not isinstance(other, self.__class__):
                raise TypeError(
                    (
                        f"Expected a {self.__class__.__name__} or {self.tensors_type} or float or int, "
                        f"but got {type(other)}"
                    )
                )
            elif len(other.tensors) != len(self.tensors):
                raise ValueError("The number of tensors must be the same")

            for t1, t2 in zip(self.tensors, other.tensors):
                t1.remainder_(t2)
            return self

    def size(self, dim: Optional[int] = None) -> List[Size]:
        """Return the size of the nested tensors."""
        return [tensor.size() for tensor in self.tensors]

    def to(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: Optional[torch.memory_format] = None,
    ) -> Self:
        """Move the nested tensors to a new device and/or dtype."""
        with set_return_type(self._return_type):
            return self.__class__(
                [
                    tensor.to(
                        device=device, dtype=dtype, non_blocking=non_blocking, copy=copy, memory_format=memory_format
                    )
                    for tensor in self.tensors
                ]
            )

    def pin_memory(self, device: Optional[DeviceLikeType] = None) -> Self:
        """Pin the memory of the nested tensors."""
        with set_return_type(self._return_type):
            return self.__class__([tensor.pin_memory(device=device) for tensor in self.tensors])

    def to_batch(self) -> BATCH_T:
        """Return the batched tensor and the shapes of the nested tensors."""
        if len(set(self.shapes)) != 1:
            raise ValueError("All tensors must have the same shape")
        elif issubclass(self.batch_tensors_type, ta_tensors._BatchConcatenatedTATensor):
            raise TypeError("BatchConcatenatedTATensor cannot be nested naïvely.")
        return self.batch_tensors_type(torch.stack(self.tensors, dim=0))

    def to_list(self) -> List[T]:
        """Return the nested tensors as a list."""
        return self.tensors
