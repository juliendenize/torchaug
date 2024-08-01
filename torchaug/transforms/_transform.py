# ==================================
# Copyright: CEA-LIST/DIASI/SIALV/
# Author : Torchaug Developers
# License: CECILL-C
# ==================================

# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

from __future__ import annotations

import enum
from math import ceil, floor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.transforms.v2._utils import check_type, has_any

from torchaug import ta_tensors
from torchaug._utils import _log_api_usage_once
from torchaug.ta_tensors import TANestedTensors, _BatchConcatenatedTATensor, set_return_type

from ._utils import is_pure_tensor
from .functional._utils._kernel import _get_kernel


class RandomApplyTransform(nn.Module):
    """Base class for all randomly applied transforms.

    For more details, please see :ref:`tutorial/transforms:Transforms Tutorial`.

    Args:
        p: The probability of applying the transform.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        num_chunks: number of chunks to split the batched input into.
        permute_chunks: whether to permute the chunks.
        batch_transform: whether to apply the transform in batch mode.
    """

    _transformed_types: Tuple[Union[Type, Callable[[Any], bool]], ...] = (torch.Tensor,)

    def __init__(
        self,
        p: float = 0.5,
        batch_inplace: bool = False,
        num_chunks: int = 1,
        permute_chunks: bool = False,
        batch_transform: bool = False,
    ) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")
        elif p > 0 and p < 1 and self._reshape_transform and batch_transform:
            raise ValueError("`p` should be 0 or 1 if `_reshape_transform` is True and `batch_transform` is True.")
        if batch_inplace and self._reshape_transform:
            raise ValueError("`inplace` should be False if `_reshape_transform` is True.")

        super().__init__()
        _log_api_usage_once(self)
        self.batch_inplace = batch_inplace
        self.permute_chunks = permute_chunks
        self.p = p
        self.batch_transform = batch_transform
        self.num_chunks = num_chunks

        self._receive_flatten_inputs = False

    @property
    def _reshape_transform(self) -> bool:
        return False

    @property
    def num_chunks(self) -> int:
        """Get the number of chunks to split the input into.

        Some subclasses can have a specific logic to determine the number of chunks.
        """
        return self._num_chunks

    @num_chunks.setter
    def num_chunks(self, num_chunks) -> None:
        """Get the number of chunks to split the input into.

        Some subclasses can have a specific logic to determine the number of chunks.
        """
        if (num_chunks == -1 or num_chunks > 1) and not self.batch_transform:
            raise ValueError("`num_chunks` should be 1 if `batch_transform` is False.")
        elif num_chunks < -1 or num_chunks == 0:
            raise ValueError("`num_chunks` should be greater than 0 or -1.")
        self._num_chunks = num_chunks

    @staticmethod
    def _get_input_batch_size(flat_inpts: list[Any]) -> int:
        """Get the batch size of the input.

        .. note::
            It returns the batch size of the first encountered tensor or TA Tensor.

        Args:
            flat_inpts: List of inputs.

        Returns:
            The batch size of the input.
        """
        for inpt in flat_inpts:
            if isinstance(inpt, (_BatchConcatenatedTATensor, TANestedTensors)):
                return inpt.batch_size
            elif isinstance(inpt, torch.Tensor):
                return inpt.shape[0]
        raise ValueError("Expected one of the inputs to be a tensor or a batched tensor.")

    @staticmethod
    def _get_input_device(flat_inpts: list[Any]) -> torch.device:
        """Get the device of the input.

        .. note::
            It returns the device of the first encountered tensor or TA Tensor.

        Args:
            flat_inpts: List of inputs.

        Returns:
            The device of the input.
        """
        for inpt in flat_inpts:
            if isinstance(inpt, (_BatchConcatenatedTATensor, torch.Tensor, TANestedTensors)):
                return inpt.device
        raise ValueError("Expected one of the inputs to be a tensor or a batched tensor.")

    def _get_chunks_indices(self, batch_size: int, num_chunks: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        if num_chunks <= 0:
            raise ValueError("`num_chunks` should be greater than 0.")
        elif num_chunks > batch_size:
            raise ValueError(
                f"`num_chunks` should be less than or equal to the batch size, but got {num_chunks} "
                f"and batch size {batch_size}."
            )
        elif num_chunks == 1:
            return (torch.arange(0, batch_size, device=device),)

        if self.permute_chunks:
            indices = torch.randperm(batch_size, device=device)
        else:
            indices = torch.arange(0, batch_size, device=device)
        return indices.chunk(num_chunks)

    def _needs_transform_list(self, flat_inputs: List[Any]) -> List[bool]:
        # Below is a heuristic on how to deal with pure tensor inputs:
        # Pure tensors, i.e. tensors that are not a ta_tensor, are passed through if there is an explicit image
        # (`ta_tensors.Image`, `ta_tensors.BatchImages`) or video (`ta_tensors.Video`, `ta_tensors.BatchVideos`)
        # in the sample.
        #
        # The heuristic should work well for most people in practice. The only case where it doesn't is if someone
        # tries to transform multiple pure tensors at the same time, expecting them all to be treated as images.
        # However, this case wasn't supported by transforms v1 either, so there is no BC concern.

        needs_transform_list = []

        if self.batch_transform:
            transform_pure_tensor = not has_any(
                flat_inputs,
                ta_tensors.BatchImages,
                ta_tensors.BatchVideos,
            )
        else:
            transform_pure_tensor = not has_any(
                flat_inputs,
                ta_tensors.Image,
                ta_tensors.BatchImages,
                ta_tensors.Video,
                ta_tensors.BatchVideos,
            )
        for inpt in flat_inputs:
            needs_transform = True

            if not check_type(inpt, self._transformed_types):
                needs_transform = False
            elif is_pure_tensor(inpt):
                if transform_pure_tensor:
                    transform_pure_tensor = False
                else:
                    needs_transform = False
            needs_transform_list.append(needs_transform)
        return needs_transform_list

    def _get_params(
        self,
        flat_inputs: List[Any],
        num_chunks: int,
        chunks_indices: Tuple[torch.Tensor, ...],
    ) -> List[Dict[str, Any]]:
        return [{} for _ in range(num_chunks)]

    def _get_indices_transform(self, batch_size: int, device: torch.device) -> torch.Tensor:
        p_mul_batch_size = self.p * batch_size
        floor_apply = floor(p_mul_batch_size)
        ceil_apply = ceil(p_mul_batch_size)

        # If 0 < p_mul_batch_size < 1, then only one element from input is augmented
        # with p probability.
        if floor_apply == 0 or ceil_apply == 0:
            num_transform = 1 if torch.rand(1).item() < self.p else 0
        elif floor_apply == ceil_apply:
            num_transform = floor_apply
        # If p_mul_batch_size is rational, then upper or lower integer p_mul_batch_size
        # elements from input are augmented randomly depending with the decimal.
        else:
            decimal = p_mul_batch_size % 1
            num_transform = floor_apply if decimal < torch.rand(1).item() else ceil_apply

        # If no augmentation return the output directly, keep consistency of inplace.
        if num_transform == 0:
            return torch.empty(0, device=device, dtype=torch.long)
        elif num_transform == 1:
            indices_transform = torch.randint(0, batch_size, (1,), device=device)
        elif num_transform > 1:
            indices_transform = torch.randperm(batch_size, device=device)[:num_transform]
        else:
            raise ValueError("The number of elements to transform should be greater than or equal to 0.")
        return indices_transform

    def _check_inputs(self, flat_inputs: List[Any]) -> None:
        pass

    def _call_kernel(self, functional: Callable, inpt: Any, *args: Any, **kwargs: Any) -> Any:
        kernel = _get_kernel(functional, type(inpt), allow_passthrough=True)
        return kernel(inpt, *args, **kwargs)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def forward_single(self, flat_inputs: List[Any]) -> List[Any]:
        if self.p == 1.0:
            pass
        elif self.p == 0.0 or torch.rand(1) >= self.p:
            return flat_inputs

        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform],
            num_chunks=1,
            chunks_indices=(
                torch.tensor(
                    [0],
                    device=torch.device("cpu"),
                ),
            ),
        )[0]

        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return flat_outputs

    def forward_batch(self, flat_inputs: List[Any]) -> List[Any]:
        if self.p == 0:  # if p is 0, return the input directly after checking the input
            return flat_inputs

        needs_transform_list = self._needs_transform_list(flat_inputs)
        batch_size = self._get_input_batch_size(flat_inputs)

        if self.p == 1:  # if p is 1, transform all inputs
            transform_all = True
            indices_transform = torch.arange(batch_size, device=torch.device("cpu"))
        else:
            indices_transform = self._get_indices_transform(
                batch_size,
                torch.device("cpu"),
            )
            transform_all = indices_transform.shape[0] == batch_size

        if not transform_all and indices_transform.shape[0] == 0:  # if no augmentation return the inputs directly.
            return flat_inputs
        elif transform_all:
            transform_inpts = flat_inputs
        else:
            transform_inpts = []  # store the input part to be augmented

            # Store the complete outputs before augmentation.
            # Part of the output that is augmented will be updated with the augmented part.
            flat_pre_outputs = []

            for inpt, needs_transform in zip(flat_inputs, needs_transform_list):
                if not needs_transform:
                    transform_inpts.append(None)
                    flat_pre_outputs.append(inpt)
                    continue

                is_ta_inpt = isinstance(inpt, ta_tensors.TATensor)
                is_contatenated_batch_ta_tensors = isinstance(inpt, _BatchConcatenatedTATensor)
                pre_output = (
                    inpt
                    if self.batch_inplace or (self._reshape_transform and not is_contatenated_batch_ta_tensors)
                    else inpt.clone()
                )
                flat_pre_outputs.append(pre_output)

                if is_contatenated_batch_ta_tensors:
                    pre_output = cast(_BatchConcatenatedTATensor, pre_output)
                    transform_inpt = pre_output.get_chunk(chunk_indices=indices_transform)
                else:
                    with set_return_type("TATensor" if is_ta_inpt else "Tensor"):
                        transform_inpt = pre_output[indices_transform]
                transform_inpts.append(transform_inpt)

        transform_batch_size = indices_transform.shape[0] if not transform_all else batch_size
        if self._num_chunks == -1:
            num_chunks = transform_batch_size
        else:
            num_chunks = min(transform_batch_size, self._num_chunks)

        chunks_indices = self._get_chunks_indices(transform_batch_size, num_chunks, torch.device("cpu"))
        if self._reshape_transform and self.permute_chunks:
            cat_chunks_indices = torch.cat(chunks_indices)

        params = self._get_params(
            [
                transform_inpt
                for (transform_inpt, needs_transform) in zip(transform_inpts, needs_transform_list)
                if needs_transform
            ],
            len(chunks_indices),
            chunks_indices,
        )

        transform_outputs = []

        for transform_inpt, needs_transform in zip(transform_inpts, needs_transform_list):
            if not needs_transform:
                transform_outputs.append(transform_inpt)
                continue
            is_ta_inpt = isinstance(transform_inpt, ta_tensors.TATensor)
            is_contatenated_batch_ta_tensors = isinstance(transform_inpt, _BatchConcatenatedTATensor)

            if num_chunks == 1:
                output = self._transform(transform_inpt, params[0])
            else:
                if self._reshape_transform:
                    output = []
                for i, chunk_indices in enumerate(chunks_indices):
                    if is_contatenated_batch_ta_tensors:
                        chunk_inpt = transform_inpt.get_chunk(chunk_indices=chunk_indices)
                        chunk_output = self._transform(chunk_inpt, params[i])
                        if self._reshape_transform:
                            output.append(chunk_output)
                        else:
                            transform_inpt.update_chunk_(chunk_output, chunk_indices=chunk_indices)
                            output = transform_inpt
                    else:
                        with set_return_type("TATensor" if is_ta_inpt else "Tensor"):
                            chunk_inpt = transform_inpt[chunk_indices]

                        chunk_output = self._transform(chunk_inpt, params[i])

                        if self._reshape_transform:
                            output.append(chunk_output)
                        else:
                            with set_return_type("TATensor" if is_ta_inpt else "Tensor"):
                                transform_inpt[chunk_indices] = chunk_output
                            output = transform_inpt
                if self._reshape_transform:
                    if is_contatenated_batch_ta_tensors:
                        output = type(transform_inpt).cat(output)
                        if self.permute_chunks:
                            data_indices = transform_inpt._get_data_indices_from_chunk_indices(cat_chunks_indices)
                            order = torch.argsort(data_indices)  # reorder output to match the original order
                            output = ta_tensors.wrap(
                                output.data[order],
                                like=output,
                                samples_ranges=transform_inpt.samples_ranges,
                            )
                    else:
                        with set_return_type("TATensor" if is_ta_inpt else "Tensor"):
                            output = torch.cat(output, dim=0)
                            if self.permute_chunks:
                                order = torch.argsort(cat_chunks_indices)  # reorder output to match the original order
                                output = output[order]
            transform_outputs.append(output)

        if not transform_all:
            flat_outputs = []
            for flat_pre_output, transform_output, needs_transform in zip(
                flat_pre_outputs, transform_outputs, needs_transform_list
            ):
                if not needs_transform:
                    flat_outputs.append(flat_pre_output)
                    continue

                is_ta_output = isinstance(flat_pre_output, ta_tensors.TATensor)
                is_contatenated_batch_ta_tensors = isinstance(
                    flat_pre_output,
                    _BatchConcatenatedTATensor,
                )

                if is_contatenated_batch_ta_tensors:
                    flat_pre_output.update_chunk_(transform_output, chunk_indices=indices_transform)
                else:
                    with set_return_type("TATensor" if is_ta_output else "Tensor"):
                        flat_pre_output[indices_transform] = transform_output
                with set_return_type("TATensor" if is_ta_output else "Tensor"):
                    flat_pre_output = flat_pre_output.contiguous()
                flat_outputs.append(flat_pre_output)
        else:
            flat_outputs = transform_outputs

        return flat_outputs

    def forward_nested(self, flat_inputs: List[Any]):
        if self.p == 0:  # if p is 0, return the input directly after checking the input
            return flat_inputs

        flattened_flat_inputs = []
        nested_types: List[Optional[Type]] = []
        batch_size = None
        for inpt in flat_inputs:
            nested_types.append(type(inpt))
            if isinstance(inpt, TANestedTensors):
                flattened_flat_inputs.append(list(inpt.tensors))
                if batch_size is None:
                    batch_size = inpt.batch_size
                elif inpt.batch_size != batch_size:
                    raise ValueError("All nested tensors should have the same batch size.")
            elif isinstance(inpt, torch.Tensor):
                raise ValueError("Expected a nested tensor, but got a single tensor.")
            else:
                flattened_flat_inputs.append(inpt)
                nested_types[-1] = None

        if batch_size is None:
            raise ValueError("Expected at least one nested tensor.")

        sample_outputs = []
        for i in range(batch_size):
            sample_input = [
                flattened_inpt[i] if nested_types[j] is not None else flattened_inpt
                for j, flattened_inpt in enumerate(flattened_flat_inputs)
            ]

            self._check_inputs(sample_input)

            sample_outputs.append(self.forward_single(sample_input))

        flat_outputs = []
        for i in range(len(sample_outputs[0])):
            flat_outputs.append(
                [
                    sample_output[i] if nested_types[i] is not None else sample_output
                    for sample_output in sample_outputs
                ]
            )
            nested_type = nested_types[i]
            if nested_type is not None:
                flat_outputs[-1] = nested_type(flat_outputs[-1])

        return flat_outputs

    def forward(self, *inputs: Any) -> Any:
        """Performs forward pass of the transform.

        Args:
            inputs: Inputs to the transform.

        Returns:
            Transformed inputs.
        """
        if not self._receive_flatten_inputs:
            inputs = inputs if len(inputs) > 1 else inputs[0]
            flat_inputs, spec = tree_flatten(inputs)
        else:
            flat_inputs = list(inputs)

        if any(isinstance(inpt, TANestedTensors) for inpt in flat_inputs):
            batch_transform = self.batch_transform
            num_chunks = self.num_chunks
            permute_chunks = self.permute_chunks

            self.batch_transform = False
            self.num_chunks = 1
            self.permute_chunks = False

            flat_outputs = self.forward_nested(flat_inputs)

            self.batch_transform = batch_transform
            self.num_chunks = num_chunks
            self.permute_chunks = permute_chunks
        elif not self.batch_transform:
            self._check_inputs(flat_inputs)
            flat_outputs = self.forward_single(flat_inputs)
        else:
            self._check_inputs(flat_inputs)
            flat_outputs = self.forward_batch(flat_inputs)
        if not self._receive_flatten_inputs:
            return tree_unflatten(flat_outputs, spec)

        return flat_outputs

    def extra_repr(self, exclude_names: List[str] = []) -> str:
        """Set the extra representation of the transform."""
        if not self.batch_transform:
            exclude_names.extend(["batch_inplace", "num_chunks", "permute_chunks", "batch_transform"])

        last_extra: Dict[str, Any] = {
            "p": None,
            "batch_inplace": None,
            "num_chunks": None,
            "permute_chunks": None,
            "batch_transform": None,
        }
        transform_extra = []
        parameters_dict = dict(self.__dict__, num_chunks=self.num_chunks)
        for name, value in parameters_dict.items():
            if name.startswith("_") or name == "training" or name in exclude_names:
                continue

            if not isinstance(value, (bool, int, float, str, tuple, list, enum.Enum)) and value is not None:
                continue

            if name in last_extra:
                last_extra[name] = value
            else:
                transform_extra.append(f"{name}={value}")

        extra = transform_extra + [f"{name}={value}" for name, value in last_extra.items() if value is not None]

        return ", ".join(extra)


class Transform(RandomApplyTransform):
    """Base class for all transforms.

    For more details, please see :ref:`tutorial/transforms:Transforms Tutorial`.

    Args:
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        num_chunks: number of chunks to split the batched input into.
        permute_chunks: whether to permute the chunks.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(
        self,
        batch_inplace: bool = False,
        num_chunks: int = 1,
        permute_chunks: bool = False,
        batch_transform: bool = False,
    ) -> None:
        super().__init__(
            p=1.0,
            batch_inplace=batch_inplace,
            num_chunks=num_chunks,
            permute_chunks=permute_chunks,
            batch_transform=batch_transform,
        )

    def extra_repr(self, exclude_names: List[str] = []) -> str:
        exclude_names.append("p")
        return super().extra_repr(exclude_names)
