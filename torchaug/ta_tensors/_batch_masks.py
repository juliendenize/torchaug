# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten

from ._mask import Mask
from ._ta_tensor import TATensor


def convert_masks_to_batch_masks(
    masks: Sequence[Mask],
) -> BatchMasks:
    """Convert a sequence of :class:`~torchaug.ta_tensors.Mask` objects to a
    :class:`~torchaug.ta_tensors.BatchMasks` object.

    Assumes all masks are valid.
    """
    attrs = [
        "requires_grad",
        "device",
        "dtype",
    ]

    for mask in masks:
        if not mask.shape[-2:] == masks[0].shape[-2:]:
            raise ValueError("All masks must have the same size.")
        for attr in attrs:
            if getattr(mask, attr) != getattr(masks[0], attr):
                raise ValueError(f"All masks must have the same {attr} attribute.")

    masks_data = torch.cat([mask.data for mask in masks])
    idx_sample = torch.tensor([0] + [mask.shape[0] for mask in masks], dtype=torch.long).cumsum(0).tolist()

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
            batch_masks[idx_sample[i] : idx_sample[i + 1]],
        )
        for i in range(len(idx_sample) - 1)
    ]

    return list_masks


class BatchMasks(TATensor):
    """:class:`torch.Tensor` subclass for batch of segmentation and detection masks.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype: Desired data type. If omitted, will be inferred from
            ``data``.
        idx_sample: Each element is the index of the first mask of the corresponding sample in the batch of
         N samples. Contains N+1 elements whose last value is the number of masks.
        device: Desired device. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the mask is constructed on the CPU.
        requires_grad: Whether autograd should record operations. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.

    """

    idx_sample: List[int]

    @property
    def batch_size(self) -> int:
        return len(self.idx_sample) - 1

    @property
    def num_masks(self) -> int:
        return self.data.shape[0]

    def get_num_masks_sample(self, idx: int) -> int:
        return self.idx_sample[idx + 1] - self.idx_sample[idx]

    @classmethod
    def cat(cls, masks_batches: Sequence[BatchMasks]):
        """Concatenates a sequence of :class:`~torchaug.ta_tensors.BatchMasks` along the first dimension.

        Args:
            masks_batches: A sequence of :class:`~torchaug.ta_tensors.BatchMasks` to concatenate.

        Returns:
            The concatenated :class:`~torchaug.ta_tensors.BatchMasks`.
        """
        attrs = [
            "requires_grad",
            "device",
            "dtype",
        ]

        for batch_mask in masks_batches:
            if not isinstance(batch_mask, BatchMasks):
                raise ValueError("All batches must be of type BatchMasks.")
            if not batch_mask.shape[-2:] == masks_batches[0].shape[-2:]:
                raise ValueError("All batches of masks must have the same size.")
            for attr in attrs:
                if getattr(batch_mask, attr) != getattr(masks_batches[0], attr):
                    raise ValueError(f"All batches of masks must have the same {attr} attribute.")

        idx_sample = (
            torch.tensor(
                [0]
                + [
                    batch_mask.get_num_masks_sample(i)
                    for batch_mask in masks_batches
                    for i in range(batch_mask.batch_size)
                ]
            )
            .cumsum(0)
            .tolist()
        )

        data = torch.cat([mask.data for mask in masks_batches], 0)

        return cls(
            data,
            idx_sample=idx_sample,
        )

    @classmethod
    def _wrap(
        cls,
        tensor: Tensor,
        *,
        idx_sample: List[int],
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
        idx_sample: List[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> BatchMasks:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

        if tensor.ndim < 2:
            raise ValueError

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
        masks = self[self.idx_sample[idx] : self.idx_sample[idx + 1]]
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
        chunk_idx_sample = torch.tensor(
            [0] + [self.idx_sample[chunk_indice + 1] - self.idx_sample[chunk_indice] for chunk_indice in chunk_indices]
        )

        chunk_idx_sample = chunk_idx_sample.cumsum(0).tolist()

        return BatchMasks(
            self[chunk_indices],
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
        self[chunk_indices] = chunk

        return self

    @classmethod
    def masked_remove(cls, masks: BatchMasks, mask: torch.Tensor) -> BatchMasks:
        """Remove masks from the batch of masks.

        Args:
            masks: The batch of masks to remove masks from.
            mask: A boolean mask to keep masks.

        Returns:
            The updated batch of masks.
        """
        old_idx_sample = masks.idx_sample
        data = masks.data[~mask]

        cpu_mask = mask.cpu()

        num_delete_per_sample = [
            cpu_mask[old_idx_sample[i] : old_idx_sample[i + 1]].sum().item() for i in range(len(old_idx_sample) - 1)
        ]

        new_idx_sample = [old_idx_sample[i] - sum(num_delete_per_sample[: i + 1]) for i in range(len(old_idx_sample))]

        return cls._wrap(
            data,
            idx_sample=new_idx_sample,
        )
