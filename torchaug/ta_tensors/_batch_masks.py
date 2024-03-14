# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten

from ._batch_concatenated_ta_tensor import _BatchConcatenatedTATensor
from ._mask import Mask


_CHECK_ATTRS = [
    "requires_grad",
    "device",
    "dtype",
]


def convert_masks_to_batch_masks(
    masks: Sequence[Mask],
) -> BatchMasks:
    """Convert a sequence of :class:`~torchaug.ta_tensors.Mask` objects to a
    :class:`~torchaug.ta_tensors.BatchMasks` object.

    Assumes all masks are valid.
    """
    for mask in masks:
        if not mask.shape[-2:] == masks[0].shape[-2:]:
            raise ValueError("All masks must have the same size.")
        for attr in _CHECK_ATTRS:
            if getattr(mask, attr) != getattr(masks[0], attr):
                raise ValueError(f"All masks must have the same {attr} attribute.")

    masks = [mask if mask.ndim > 2 else mask.unsqueeze(0) for mask in masks]
    masks_data = torch.cat(masks)
    idx_sample = []
    sum_masks = 0
    for mask in masks:
        idx_sample.append((sum_masks, sum_masks + mask.shape[0]))
        sum_masks += mask.shape[0]

    batch_masks = BatchMasks(
        masks_data,
        idx_sample=idx_sample,
    )

    return batch_masks


def convert_batch_masks_to_masks(
    batch_masks: BatchMasks,
) -> List[Mask]:
    """Convert :class:`~torchaug.ta_tensors.BatchMasks` object to a list of
    :class:`~torchaug.ta_tensors.Mask` objects.
    """
    idx_sample = batch_masks.idx_sample

    list_masks = [
        Mask(
            batch_masks[idx_start:idx_stop],
        )
        for (idx_start, idx_stop) in idx_sample
    ]
    return list_masks


class BatchMasks(_BatchConcatenatedTATensor):
    """:class:`torch.Tensor` subclass for batch of segmentation and detection masks.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype: Desired data type. If omitted, will be inferred from
            ``data``.
        idx_sample: Each element is the range of the indexes of the masks for each sample.
        device: Desired device. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the mask is constructed on the CPU.
        requires_grad: Whether autograd should record operations. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.

    """

    @classmethod
    def cat(cls, masks_batches: Sequence[BatchMasks]):
        """Concatenates a sequence of :class:`~torchaug.ta_tensors.BatchMasks` along the first dimension.

        Args:
            masks_batches: A sequence of :class:`~torchaug.ta_tensors.BatchMasks` to concatenate.

        Returns:
            The concatenated :class:`~torchaug.ta_tensors.BatchMasks`.
        """
        for batch_mask in masks_batches:
            if not isinstance(batch_mask, BatchMasks):
                raise ValueError("All batches must be of type BatchMasks.")
            if not batch_mask.shape[-2:] == masks_batches[0].shape[-2:]:
                raise ValueError("All batches of masks must have the same size.")
            for attr in _CHECK_ATTRS:
                if getattr(batch_mask, attr) != getattr(masks_batches[0], attr):
                    raise ValueError(f"All batches of masks must have the same {attr} attribute.")

        idx_sample = []
        sum_masks = 0
        for batch_masks in masks_batches:
            for idx_start, idx_stop in batch_masks.idx_sample:
                idx_sample.append((idx_start + sum_masks, idx_stop + sum_masks))
            sum_masks += batch_masks.num_data

        data = torch.cat([batch_masks.data for batch_masks in masks_batches], 0)

        return cls(
            data,
            idx_sample=idx_sample,
        )

    @classmethod
    def _wrap(
        cls,
        tensor: Tensor,
        *,
        idx_sample: List[Tuple[int, int]],
        check_dims: bool = True,
    ) -> BatchMasks:  # type: ignore[override]
        if check_dims and tensor.ndim < 2:
            raise ValueError(f"Expected at least a 2D tensor, got {tensor.ndim}D tensor")
        batch_masks = tensor.as_subclass(cls)
        batch_masks.idx_sample = idx_sample
        return batch_masks

    def __new__(
        cls,
        data: Any,
        *,
        idx_sample: List[Tuple[int, int]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> BatchMasks:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

        if tensor.ndim < 3:
            raise ValueError

        cls._check_idx_sample(idx_sample, tensor)

        return cls._wrap(tensor, idx_sample=idx_sample)

    @classmethod
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> BatchMasks:
        # If there are BatchMasks instances in the output, their metadata got lost when we called
        # super().__torch_function__. We need to restore the metadata somehow, so we choose to take
        # the metadata from the first mask in the parameters.
        # This should be what we want in most cases. When it's not, it's probably a mis-use anyway, e.g.
        # something like mask_N1HW + mask_N2HW; we don't guard against those cases.
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))  # type: ignore[operator]
        first_batch_masks_from_args = next(x for x in flat_params if isinstance(x, BatchMasks))

        idx_sample = first_batch_masks_from_args.idx_sample.copy()  # clone the list.

        if isinstance(output, torch.Tensor) and not isinstance(output, BatchMasks):
            output = BatchMasks._wrap(
                output,
                idx_sample=idx_sample,
                check_dims=False,
            )
        elif isinstance(output, (tuple, list)):
            output = type(output)(
                BatchMasks._wrap(
                    part,
                    idx_sample=idx_sample,
                    check_dims=False,
                )
                for part in output
            )
        return output

    def get_sample(self, idx: int) -> Mask:
        """Get the masks for a sample in the batch.

        Args:
            idx: The index of the sample to get.

        Returns:
            The masks for the sample.
        """
        masks = self[self.idx_sample[idx][0] : self.idx_sample[idx][1]]
        return Mask(
            masks,
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def get_chunk(self, chunk_indices: torch.Tensor) -> BatchMasks:
        """Get a chunk of the batch of masks.

        Args:
            chunk_indices: The indices of the chunk to get.

        Returns:
            The chunk of the batch of masks.
        """
        chunk_idx_sample = self._get_chunk_idx_sample_from_chunk_indices(chunk_indices)
        data_indices = self._get_data_indices_from_chunk_indices(chunk_indices)
        return BatchMasks(
            self[data_indices],
            idx_sample=chunk_idx_sample,
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def update_chunk_(self, chunk: BatchMasks, chunk_indices: torch.Tensor) -> BatchMasks:
        """Update a chunk of the batch of masks.

        Args:
            chunk: The chunk update.
            chunk_indices: The indices of the chunk to update.

        Returns:
            The updated batch of masks.
        """
        return super().update_chunk_(chunk, chunk_indices)

    def to_samples(self) -> list[Mask]:
        """Get the tensors."""
        return [self.get_sample(i).clone() for i in range(self.batch_size)]

    @classmethod
    def masked_select(cls, masks: BatchMasks, mask: torch.Tensor) -> BatchMasks:
        """Remove masks from the batch of masks.

        Args:
            masks: The batch of masks to remove masks from.
            mask: A boolean mask to keep masks.

        Returns:
            The updated batch of masks.
        """
        old_idx_sample = masks.idx_sample
        data = masks.data[mask]

        neg_mask = (~mask).cpu()

        num_delete_per_sample = [
            neg_mask[old_idx_sample[i][0] : old_idx_sample[i][1]].sum().item() for i in range(len(old_idx_sample))
        ]

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
