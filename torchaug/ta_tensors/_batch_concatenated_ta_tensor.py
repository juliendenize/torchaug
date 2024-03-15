# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ._ta_tensor import TATensor


class _BatchConcatenatedTATensor(TATensor):
    """:class:`torch.Tensor` subclass for batch of tensors that are contatenated.

    Args:
        range_samples: Each element is the range of the indices of the tensors for each sample.
    """

    range_samples: List[Tuple[int, int]]

    @classmethod
    def _check_range_samples(cls, range_samples: List[Tuple[int, int]], data: torch.Tensor) -> None:
        for i, idx in enumerate(range_samples):
            if not isinstance(idx, tuple) or len(idx) != 2:
                raise ValueError(f"Expected a tuple of two integers, got {idx}")
            if not isinstance(idx[0], int) or not isinstance(idx[1], int):
                raise ValueError(f"Expected a tuple of two integers, got {idx}")
            if i == 0:
                if idx[0] != 0:
                    raise ValueError(f"Expected the start index to be 0, got {idx[0]}")
            else:
                if idx[0] != range_samples[i - 1][1]:
                    raise ValueError("Expected the start index to be stop index of the previous sample.")
            if i == (len(range_samples) - 1):
                if idx[1] != data.shape[0]:
                    raise ValueError(f"Expected the stop index to be {data.shape[0]}, got {idx[1]}")
            else:
                if idx[1] != range_samples[i + 1][0]:
                    raise ValueError("Expected the stop index to be the start index of the next sample.")
            if idx[1] < idx[0]:
                raise ValueError(f"Expected the stop index to be greater than the start index, got {idx}")

    @property
    def batch_size(self) -> int:
        return len(self.range_samples)

    @property
    def num_data(self) -> int:
        return self.data.shape[0]

    def get_num_data_sample(self, idx: int) -> int:
        return self.range_samples[idx][1] - self.range_samples[idx][0]

    @classmethod
    def cat(cls, ta_tensors: Sequence[TATensor]) -> _BatchConcatenatedTATensor:
        """Concatenates a sequence of :class:`~torchaug.ta_tensors.TATensor` along the first dimension.

        Must be defined by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def _wrap(
        cls,
        tensor: Tensor,
        *,
        range_samples: List[Tuple[int, int]],
        check_dims: bool = True,
    ) -> _BatchConcatenatedTATensor:  # type: ignore[override]
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> _BatchConcatenatedTATensor:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_sample(self, idx: int) -> TATensor:
        """Get the data for a sample in the batch."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _get_data_indices_from_chunk_indices(self, chunk_indices: torch.Tensor) -> torch.Tensor:
        """Get the data indices from the chunk indices."""
        return [idx for indices in chunk_indices for idx in range(*self.range_samples[indices])]

    def _get_chunk_range_samples_from_chunk_indices(self, chunk_indices: torch.Tensor) -> List[Tuple[int, int]]:
        """Get the chunk idx sample from the chunk indices."""
        chunk_range_samples = []
        sum_boxes = 0
        for chunk_indice in chunk_indices:
            num_data_sample = self.get_num_data_sample(chunk_indice)
            chunk_range_samples.append((sum_boxes, sum_boxes + num_data_sample))
            sum_boxes += num_data_sample
        return chunk_range_samples

    def get_chunk(self, chunk_indices: torch.Tensor) -> _BatchConcatenatedTATensor:
        """Get a chunk of the batch."""
        raise NotImplementedError("Subclasses must implement this method.")

    def update_chunk_(
        self, chunk: _BatchConcatenatedTATensor, chunk_indices: torch.Tensor
    ) -> _BatchConcatenatedTATensor:
        """Update a chunk of the batch.

        Args:
            chunk: The chunk update.
            chunk_indices: The indices of the chunk to update.

        Returns:
            The updated batch.
        """
        data_indices = self._get_data_indices_from_chunk_indices(chunk_indices)
        self[data_indices] = chunk
        return self

    def to_samples(self) -> list[TATensor]:
        """Get the tensors."""
        return [self.get_sample(i).clone() for i in range(self.batch_size)]

    @classmethod
    def masked_select(cls, ta_tensors: _BatchConcatenatedTATensor, mask: torch.Tensor) -> _BatchConcatenatedTATensor:
        """Remove ta_tensors from the batch of ta_tensors."""
        raise NotImplementedError("Subclasses must implement this method.")
