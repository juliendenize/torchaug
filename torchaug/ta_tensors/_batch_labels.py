# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils._pytree import tree_flatten

from ._batch_concatenated_ta_tensor import _BatchConcatenatedTATensor
from ._labels import Labels


_CHECK_ATTRS = [
    "requires_grad",
    "device",
    "dtype",
]


def convert_labels_to_batch_labels(
    labels: Sequence[Labels],
) -> BatchLabels:
    """Convert a sequence of :class:`~torchaug.ta_tensors.Labels` to a
    :class:`~torchaug.ta_tensors.BatchLabels` object.

    Assumes all tensors are valid.
    """
    for attr in _CHECK_ATTRS:
        for label in labels:
            if getattr(label, attr) != getattr(labels[0], attr):
                raise ValueError(f"All labels must have the same {attr} attribute.")

    labels_data = torch.cat(labels)

    idx_sample = []
    sum_labels = 0
    for label in labels:
        idx_sample.append((sum_labels, sum_labels + label.shape[0]))
        sum_labels += label.shape[0]

    batch_labels = BatchLabels(
        labels_data,
        idx_sample=idx_sample,
    )

    return batch_labels


def convert_batch_labels_to_labels(
    batch_labels: BatchLabels,
) -> List[Labels]:
    """Convert :class:`~torchaug.ta_tensors.BatchLabels` object to a list of
    :class:`~torchaug.ta_tensors.Labels`.
    """
    idx_sample = batch_labels.idx_sample

    list_labels = [Labels(batch_labels[idx_start:idx_stop]) for idx_start, idx_stop in idx_sample]
    return list_labels


class BatchLabels(_BatchConcatenatedTATensor):
    """:class:`BatchLabels` subclass for concatenated labels.

    Useful for labels of bounding boxes or masks, where each sample can have a different number of labels.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype: Desired data type. If omitted, will be inferred from
            ``data``.
        idx_sample: Each element is a tuple of the index of the first tensor and the last label of the corresponding
            sample in the batch of N samples. If None, the corresponding sample has no label.
        device: Desired device. If omitted and ``data`` is a :class:`torch.Tensor`, the device is taken from
            it. Otherwise, the batch of tensor is constructed on the CPU.
        requires_grad: Whether autograd should record operations. If omitted and
            ``data`` is a :class:`Labels`, the value is taken from it. Otherwise, defaults to ``False``.

    """

    @classmethod
    def cat(cls, labels_batches: Sequence[BatchLabels]):
        """Concatenates a sequence of :class:`~torchaug.ta_tensors.BatchLabels` along the first dimension.

        Args:
            labels_batches: A sequence of :class:`~torchaug.ta_tensors.BatchLabels` to concatenate.

        Returns:
            The concatenated :class:`~torchaug.ta_tensors.BatchLabels`.
        """
        for batch_label in labels_batches:
            if not isinstance(batch_label, BatchLabels):
                raise ValueError("All batches must be of type BatchLabels.")
            if batch_label.ndim > 1 and not batch_label.shape[1:] == labels_batches[0].shape[1:]:
                raise ValueError("All batches of tensors must have the same size.")
            for attr in _CHECK_ATTRS:
                if getattr(batch_label, attr) != getattr(labels_batches[0], attr):
                    raise ValueError(f"All batches of tensors must have the same {attr} attribute.")

        idx_sample = []
        sum_labels = 0
        for batch_labels in labels_batches:
            for idx_start, idx_stop in batch_labels.idx_sample:
                idx_sample.append((idx_start + sum_labels, idx_stop + sum_labels))
            sum_labels += batch_labels.num_data

        data = torch.cat([batch_label.data for batch_label in labels_batches], 0)

        return cls(
            data,
            idx_sample=idx_sample,
        )

    @classmethod
    def _wrap(  # type: ignore[override]
        cls,
        tensor: torch.Tensor,
        *,
        idx_sample: List[Tuple[int, int]],
    ) -> BatchLabels:
        batch_labels = tensor.as_subclass(cls)
        batch_labels.idx_sample = idx_sample
        return batch_labels

    def __new__(
        cls,
        data: Any,
        *,
        idx_sample: List[Tuple[int, int]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> BatchLabels:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

        cls._check_idx_sample(idx_sample, tensor)

        return cls._wrap(tensor, idx_sample=idx_sample)

    @classmethod
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> BatchLabels:
        # If there are BatchLabels instances in the output, their metadata got lost when we called
        # super().__torch_function__. We need to restore the metadata somehow, so we choose to take
        # the metadata from the first batch of tensors in the parameters.
        # This should be what we want in most cases. When it's not, it's probably a mis-use anyway, e.g.
        # something like batch_cat_tensor_dims1 + batch_cat_tensor_dims1; we don't guard against those cases.
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))  # type: ignore[operator]
        first_batch_labels_from_args = next(x for x in flat_params if isinstance(x, BatchLabels))

        idx_sample = first_batch_labels_from_args.idx_sample.copy()  # clone the list.

        if isinstance(output, torch.Tensor) and not isinstance(output, BatchLabels):
            output = BatchLabels._wrap(
                output,
                idx_sample=idx_sample,
            )
        elif isinstance(output, (tuple, list)):
            output = type(output)(
                BatchLabels._wrap(
                    part,
                    idx_sample=idx_sample,
                )
                for part in output
            )
        return output

    def get_sample(self, idx: int) -> Labels:
        """Get the tensors for a sample in the batch.

        Args:
            idx: The index of the sample to get.

        Returns:
            The tensors for the sample.
        """
        labels = self[self.idx_sample[idx][0] : self.idx_sample[idx][1]]
        return Labels(labels)

    def get_chunk(self, chunk_indices: torch.Tensor) -> BatchLabels:
        """Get a chunk of the batch of tensors.

        Args:
            chunk_indices: The indices of the chunk to get.

        Returns:
            The chunk of the batch of tensors.
        """
        chunk_idx_sample = self._get_chunk_idx_sample_from_chunk_indices(chunk_indices)
        data_indices = self._get_data_indices_from_chunk_indices(chunk_indices)
        return BatchLabels(
            self[data_indices],
            idx_sample=chunk_idx_sample,
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def update_chunk_(self, chunk: BatchLabels, chunk_indices: torch.Tensor) -> BatchLabels:
        """Update a chunk of the batch of labels.

        Args:
            chunk: The chunk update.
            chunk_indices: The indices of the chunk to update.

        Returns:
            The updated batch of labels.
        """
        return super().update_chunk_(chunk, chunk_indices)

    def to_samples(self) -> list[Labels]:
        """Get the tensors."""
        return [self.get_sample(i).clone() for i in range(self.batch_size)]

    @classmethod
    def masked_select(cls, labels: BatchLabels, mask: torch.Tensor) -> BatchLabels:
        """Remove labels from the batch of labels.

        Args:
            labels: The batch of labels to remove labels from.
            mask: A boolean mask to keep labels.

        Returns:
            The updated batch of labels.
        """
        old_idx_sample = labels.idx_sample
        data = labels.data[mask]

        neg_mask = (~mask).cpu()

        num_delete_per_sample = [neg_mask[idx_start:idx_stop].sum().item() for idx_start, idx_stop in old_idx_sample]

        new_idx_sample = [
            (
                old_idx_sample[i][0] - sum(num_delete_per_sample[:i]),
                old_idx_sample[i][1] - sum(num_delete_per_sample[: i + 1]),
            )
            for i in range(len(old_idx_sample))
        ]

        return cls._wrap(
            data,
            idx_sample=new_idx_sample,
        )
