# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import torch
from torch import Tensor

from ._ta_tensor import TATensor


class _BatchConcatenatedTATensor(TATensor):
    """:class:`torch.Tensor` subclass for batch of tensors that are contatenated.

    Args:
        idx_sample: Each element is the index of the first mask of the corresponding sample in the batch of
         N samples. Contains N+1 elements whose last value is the number of masks.
    """

    idx_sample: List[int]

    @property
    def batch_size(self) -> int:
        return len(self.idx_sample) - 1

    @property
    def num_data(self) -> int:
        return self.data.shape[0]

    def get_num_data_sample(self, idx: int) -> int:
        return self.idx_sample[idx + 1] - self.idx_sample[idx]

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
        idx_sample: List[int],
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
        return [
            idx for indices in chunk_indices for idx in range(self.idx_sample[indices], self.idx_sample[indices + 1])
        ]

    def _get_chunk_idx_sample_from_chunk_indices(self, chunk_indices: torch.Tensor) -> List[int]:
        """Get the chunk idx sample from the chunk indices."""
        chunk_idx_sample = torch.tensor(
            [0] + [self.idx_sample[chunk_indice + 1] - self.idx_sample[chunk_indice] for chunk_indice in chunk_indices]
        )

        chunk_idx_sample = chunk_idx_sample.cumsum(0).tolist()
        return chunk_idx_sample

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
    def masked_remove(cls, ta_tensors: _BatchConcatenatedTATensor, mask: torch.Tensor) -> _BatchConcatenatedTATensor:
        """Remove ta_tensors from the batch of ta_tensors."""
        raise NotImplementedError("Subclasses must implement this method.")
