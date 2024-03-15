# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten

from ._batch_concatenated_ta_tensor import _BatchConcatenatedTATensor
from ._bounding_boxes import BoundingBoxes, BoundingBoxFormat


_CHECK_ATTRS = [
    "canvas_size",
    "format",
    "requires_grad",
    "device",
    "dtype",
]


def convert_bboxes_to_batch_bboxes(
    bboxes: List[BoundingBoxes],
) -> BatchBoundingBoxes:
    """Convert a list of :class:`~torchaug.ta_tensors.BoundingBoxes` objects to a
    :class:`~torchaug.ta_tensors.BatchBoundingBoxes` object.

    Assumes all bboxes are valid.
    """
    if not all(
        getattr(bbox, attr) == getattr(bboxes[0], attr) for bbox in bboxes if bbox is not None for attr in _CHECK_ATTRS
    ):
        raise ValueError("All bounding boxes must have the same attributes.")

    bboxes_data = torch.cat(bboxes)
    range_samples = []
    sum_bboxes = 0
    for bbox in bboxes:
        range_samples.append((sum_bboxes, sum_bboxes + bbox.shape[0]))
        sum_bboxes += bbox.shape[0]

    batch_bboxes = BatchBoundingBoxes(
        bboxes_data,
        canvas_size=bboxes[0].canvas_size,
        format=bboxes[0].format,
        range_samples=range_samples,
    )

    return batch_bboxes


def convert_batch_bboxes_to_bboxes(
    bboxes: BatchBoundingBoxes,
) -> List[BoundingBoxes]:
    """Convert :class:`~torchaug.ta_tensors.BatchBoundingBoxes` object to a list of
    :class:`~torchaug.ta_tensors.BoundingBoxes` objects.
    """
    canvas_size, format, range_samples = (
        bboxes.canvas_size,
        bboxes.format,
        bboxes.range_samples,
    )

    list_bboxes = [
        BoundingBoxes(
            bboxes[idx_start:idx_stop],
            canvas_size=canvas_size,
            format=format,
        )
        for idx_start, idx_stop in range_samples
    ]

    return list_bboxes


class BatchBoundingBoxes(_BatchConcatenatedTATensor):
    """:class:`torch.Tensor` subclass for batch of bounding boxes.

    .. note::
        There should be only one :class:`~torchaug.ta_tensors.BatchBoundingBoxes`
        instance per sample e.g. ``{"img": img, "bbox": BatchBoundingBoxes(...)}``.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        format: Format of the bounding box.
        canvas_size: Height and width of the corresponding batch of images or videos.
        range_samples: Each element is the range of the indices of the bounding boxes for each sample.
        dtype: Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device: Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad: Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    format: BoundingBoxFormat
    canvas_size: Tuple[int, int]

    @classmethod
    def cat(cls, bounding_boxes_batches: Sequence[BatchBoundingBoxes]) -> BatchBoundingBoxes:
        """Concatenates the given sequence of :class:`~torchaug.ta_tensors._batch_bounding_boxes.BatchBoundingBoxes`
        along the first dimension.

        Args:
            bounding_boxes_batches: The sequence of
                :class:`~torchaug.ta_tensors._batch_bounding_boxes.BatchBoundingBoxes` to concatenate.

        Returns:
            BatchBoundingBoxes: The concatenated batch of bounding boxes.
        """
        for batch_bounding_boxes in bounding_boxes_batches:
            if not isinstance(batch_bounding_boxes, BatchBoundingBoxes):
                raise ValueError("All elements in the sequence must be instances of BatchBoundingBoxes.")
            for attr in _CHECK_ATTRS:
                if getattr(batch_bounding_boxes, attr) != getattr(bounding_boxes_batches[0], attr):
                    raise ValueError(f"All batches of masks must have the same {attr} attribute.")

        range_samples = []
        sum_boxes = 0
        for batch_bounding_boxes in bounding_boxes_batches:
            for idx_start, idx_stop in batch_bounding_boxes.range_samples:
                range_samples.append(
                    (
                        idx_start + sum_boxes,
                        idx_stop + sum_boxes,
                    )
                )
            sum_boxes += batch_bounding_boxes.num_data

        data = torch.cat([bounding_box.data for bounding_box in bounding_boxes_batches], 0)

        return cls(
            data,
            range_samples=range_samples,
            format=bounding_boxes_batches[0].format,
            canvas_size=bounding_boxes_batches[0].canvas_size,
        )

    @classmethod
    def _wrap(  # type: ignore[override]
        cls,
        tensor: Tensor,
        *,
        format: Union[BoundingBoxFormat, str],
        canvas_size: Tuple[int, int],
        range_samples: List[Tuple[int, int]],
        check_dims: bool = True,
    ) -> BatchBoundingBoxes:
        if check_dims and tensor.ndim != 2:
            raise ValueError(f"Expected a 2D tensor, got {tensor.ndim}D.")
        if isinstance(format, str):
            format = BoundingBoxFormat[format.upper()]  # type: ignore[misc]
        batch_bounding_boxes = tensor.as_subclass(cls)
        batch_bounding_boxes.format = format
        batch_bounding_boxes.canvas_size = canvas_size
        batch_bounding_boxes.range_samples = range_samples
        return batch_bounding_boxes

    def __new__(
        cls,
        data: Any,
        *,
        format: Union[BoundingBoxFormat, str],
        canvas_size: Tensor,
        range_samples: List[Tuple[int, int]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> BatchBoundingBoxes:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        cls._check_range_samples(range_samples, tensor)
        return cls._wrap(tensor, format=format, canvas_size=canvas_size, range_samples=range_samples)

    @classmethod
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> BatchBoundingBoxes:
        # If there are BatchBoundingBoxes instances in the output, their metadata got lost when we called
        # super().__torch_function__. We need to restore the metadata somehow, so we choose to take
        # the metadata from the first bbox in the parameters.
        # This should be what we want in most cases. When it's not, it's probably a mis-use anyway, e.g.
        # something like some_xyxy_bbox + some_xywh_bbox; we don't guard against those cases.
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))  # type: ignore[operator]
        first_batch_bboxes_from_args = next(x for x in flat_params if isinstance(x, BatchBoundingBoxes))
        format, canvas_size, range_samples = (
            first_batch_bboxes_from_args.format,
            first_batch_bboxes_from_args.canvas_size,
            first_batch_bboxes_from_args.range_samples,
        )
        range_samples = range_samples.copy()  # clone the list.

        if isinstance(output, torch.Tensor) and not isinstance(output, BatchBoundingBoxes):
            output = BatchBoundingBoxes._wrap(
                output,
                format=format,
                canvas_size=canvas_size,
                range_samples=range_samples,
                check_dims=False,
            )
        elif isinstance(output, (tuple, list)):
            output = type(output)(
                BatchBoundingBoxes._wrap(
                    part,
                    format=format,
                    canvas_size=canvas_size,
                    range_samples=range_samples,
                    check_dims=False,
                )
                for part in output
            )
        return output

    def get_sample(self, idx: int) -> BoundingBoxes:
        """Get the bounding boxes for a sample in the batch.

        Args:
            idx: The index of the sample to get.

        Returns:
            The bounding boxes for the sample.
        """
        boxes = self[self.range_samples[idx][0] : self.range_samples[idx][1]]
        return BoundingBoxes(
            boxes,
            format=self.format,
            canvas_size=self.canvas_size,
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def get_chunk(self, chunk_indices: torch.Tensor) -> BatchBoundingBoxes:
        """Get a chunk of the batch of  bounding boxes.

        Args:
            chunk_indices (torch.Tensor): The indices of the chunk to get.

        Returns:
            BatchBoundingBoxes: The chunk of the batch bounding boxes.
        """
        chunk_range_samples = self._get_chunk_range_samples_from_chunk_indices(chunk_indices)
        data_indices = self._get_data_indices_from_chunk_indices(chunk_indices)

        return BatchBoundingBoxes(
            self[data_indices],
            format=self.format,
            canvas_size=self.canvas_size,
            range_samples=chunk_range_samples,
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def update_chunk_(self, chunk: BatchBoundingBoxes, chunk_indices: torch.Tensor) -> BatchBoundingBoxes:
        """Update a chunk of the batch of bounding boxes.

        Args:
            chunk (BatchBoundingBoxes): The chunk update.
            chunk_indices (torch.Tensor): The indices of the chunk to update.

        Returns:
            BatchBoundingBoxes: The updated batch of bounding boxes.
        """
        if chunk.format != self.format:
            raise ValueError("The format of the chunk must be the same as the format of the batch of bounding boxes.")

        if chunk.canvas_size != self.canvas_size:
            raise ValueError(
                "The canvas size of the chunk must be the same as the canvas size of the batch of bounding boxes."
            )
        self = super().update_chunk_(chunk, chunk_indices)
        return self

    def to_samples(self) -> list[BoundingBoxes]:
        """Get the tensors."""
        return [self.get_sample(i).clone() for i in range(self.batch_size)]

    @classmethod
    def masked_select(cls, bboxes: BatchBoundingBoxes, mask: torch.Tensor) -> BatchBoundingBoxes:
        """Remove boxes from the batch of bounding boxes.

        Args:
            bboxes (BatchBoundingBoxes): The batch of bounding boxes to remove boxes from.
            mask (torch.Tensor): A boolean mask to keep boxes.

        Returns:
            BatchBoundingBoxes: The updated batch of bounding boxes.
        """
        # Remove boxes
        old_range_samples = bboxes.range_samples
        data = bboxes.data[mask]

        neg_mask = (~mask).cpu()

        num_delete_per_sample = [neg_mask[idx_start:idx_stop].sum().item() for idx_start, idx_stop in old_range_samples]

        new_range_samples = [
            (
                old_range_samples[i][0] - sum(num_delete_per_sample[:i]),
                old_range_samples[i][1] - sum(num_delete_per_sample[: i + 1]),
            )
            for i in range(len(old_range_samples))
        ]

        return cls._wrap(
            data,
            format=bboxes.format,
            canvas_size=bboxes.canvas_size,
            range_samples=new_range_samples,
        )

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(
            format=self.format,
            canvas_size=self.canvas_size,
            range_samples=self.range_samples,
        )
