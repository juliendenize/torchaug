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
    """
    for attr in _CHECK_ATTRS:
        for label in labels:
            if getattr(label, attr) != getattr(labels[0], attr):
                raise ValueError(f"All labels must have the same {attr} attribute.")

    labels_data = torch.cat(labels)

    samples_ranges = []
    sum_labels = 0
    for label in labels:
        samples_ranges.append((sum_labels, sum_labels + label.shape[0]))
        sum_labels += label.shape[0]

    batch_labels = BatchLabels(
        labels_data,
        samples_ranges=samples_ranges,
    )

    return batch_labels


def convert_batch_labels_to_labels(
    batch_labels: BatchLabels,
) -> List[Labels]:
    """Convert :class:`~torchaug.ta_tensors.BatchLabels` object to a list of
    :class:`~torchaug.ta_tensors.Labels`.
    """
    samples_ranges = batch_labels.samples_ranges

    list_labels = [Labels(batch_labels[idx_start:idx_stop]) for idx_start, idx_stop in samples_ranges]
    return list_labels


class BatchLabels(_BatchConcatenatedTATensor):
    """:class:`BatchLabels` subclass for concatenated labels.

    Useful for labels of bounding boxes or masks, where each sample can have a different number of labels.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype: Desired data type. If omitted, will be inferred from
            ``data``.
        samples_ranges: Each element is the range of the indices of the labels for each sample.
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

        samples_ranges = []
        sum_labels = 0
        for batch_labels in labels_batches:
            for idx_start, idx_stop in batch_labels.samples_ranges:
                samples_ranges.append((idx_start + sum_labels, idx_stop + sum_labels))
            sum_labels += batch_labels.num_data

        data = torch.cat([batch_label.data for batch_label in labels_batches], 0)

        return cls(
            data,
            samples_ranges=samples_ranges,
        )

    @classmethod
    def _wrap(  # type: ignore[override]
        cls,
        tensor: torch.Tensor,
        *,
        samples_ranges: List[Tuple[int, int]],
    ) -> BatchLabels:
        batch_labels = tensor.as_subclass(cls)
        batch_labels.samples_ranges = samples_ranges
        return batch_labels

    def __new__(
        cls,
        data: Any,
        *,
        samples_ranges: List[Tuple[int, int]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> BatchLabels:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

        cls._check_samples_ranges(samples_ranges, tensor)

        return cls._wrap(tensor, samples_ranges=samples_ranges)

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

        samples_ranges = first_batch_labels_from_args.samples_ranges.copy()  # clone the list.

        if isinstance(output, torch.Tensor) and not isinstance(output, BatchLabels):
            output = BatchLabels._wrap(
                output,
                samples_ranges=samples_ranges,
            )
        elif isinstance(output, (tuple, list)):
            output = type(output)(
                BatchLabels._wrap(
                    part,
                    samples_ranges=samples_ranges,
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
        labels = self[self.samples_ranges[idx][0] : self.samples_ranges[idx][1]]
        return Labels(labels)

    def get_chunk(self, chunk_indices: torch.Tensor) -> BatchLabels:
        """Get a chunk of the batch of tensors.

        Args:
            chunk_indices: The indices of the chunk to get.

        Returns:
            The chunk of the batch of tensors.
        """
        chunk_samples_ranges = self._get_chunk_samples_ranges_from_chunk_indices(chunk_indices)
        data_indices = self._get_data_indices_from_chunk_indices(chunk_indices)
        return BatchLabels(
            self[data_indices],
            samples_ranges=chunk_samples_ranges,
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
        old_samples_ranges = labels.samples_ranges
        data = labels.data[mask]

        neg_mask = (~mask).cpu()

        num_delete_per_sample = [
            neg_mask[idx_start:idx_stop].sum().item() for idx_start, idx_stop in old_samples_ranges
        ]

        new_samples_ranges = [
            (
                old_samples_ranges[i][0] - sum(num_delete_per_sample[:i]),
                old_samples_ranges[i][1] - sum(num_delete_per_sample[: i + 1]),
            )
            for i in range(len(old_samples_ranges))
        ]

        return cls._wrap(
            data,
            samples_ranges=new_samples_ranges,
        )
