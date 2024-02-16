from __future__ import annotations
import torch
from torch import Tensor
from typing import Any, Mapping, Sequence, Tuple
from torchvision.tv_tensors import BoundingBoxFormat, TVTensor
from torch.utils._pytree import tree_flatten
from torchaug.ta_tensors._bounding_boxes import BoundingBoxes


def convert_bboxes_to_batch_bboxes(
    bboxes: Sequence[BoundingBoxes],
) -> BatchBoundingBoxes:
    """Convert a sequence of :class:`~torchvision.tv_tensors.BoundingBoxes` objects to a
    :class:`~torchaug.torchaug_tensors.BatchBoundingBoxes` object."""

    if not all(
        bbox.canvas_size == bboxes[0].canvas_size and bbox.format == bboxes[0].format
        for bbox in bboxes
    ):
        raise ValueError

    canvas_size, format, device = (
        bboxes[0].canvas_size,
        bboxes[0].format,
        bboxes[0].device,
    )

    bboxes_data = torch.cat([bbox.as_subclass(Tensor) for bbox in bboxes])
    idx_sample = torch.tensor(
        [0] + [bbox.shape[0] for bbox in bboxes], device=device, dtype=torch.int32
    ).cumsum(0)

    batch_bboxes = BatchBoundingBoxes(
        bboxes_data, canvas_size=canvas_size, format=format, idx_sample=idx_sample
    )

    return batch_bboxes


def convert_batch_bboxes_to_bboxes(
    bboxes: BatchBoundingBoxes,
) -> list[BoundingBoxes]:
    """Convert :class:`~torchaug.torchaug_tensors.BatchBoundingBoxes` object to a list of
    :class:`~torchvision.tv_tensors.BoundingBoxes` objects."""

    canvas_size, format, device, idx_sample = (
        bboxes.canvas_size,
        bboxes.format,
        bboxes.device,
        bboxes.idx_sample,
    )

    list_bboxes = [
        BoundingBoxes(
            bboxes[idx_sample[i] : idx_sample[i + 1]],
            canvas_size=canvas_size,
            format=format,
            device=device,
        )
        for i in range(idx_sample.shape[0] - 1)
    ]

    return list_bboxes


class BatchBoundingBoxes(TVTensor):
    """:class:`torch.Tensor` subclass for bounding boxes.

    .. note::
        There should be only one :class:`~torchaug.ta_tensors.BatchBoundingBoxes`
        instance per sample e.g. ``{"img": img, "bbox": BatchBoundingBoxes(...)}``.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        format (BoundingBoxFormat, str): Format of the bounding box.
        canvas_size (two-tuple of ints): Height and width of the corresponding batch of images or videos.
        idx_sample (torch.Tensor): Each element is the index of the first bounding box of the corresponding sample in the batch of N samples. Contains N+1 elements whose last value is the number of bounding boxes.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    format: BoundingBoxFormat
    canvas_size: Tuple[int, int]
    idx_sample: Tensor

    @classmethod
    def _wrap(
        cls,
        tensor: Tensor,
        *,
        format: BoundingBoxFormat | str,
        canvas_size: Tuple[int, int],
        idx_sample: Tensor,
        check_dims: bool = True,
    ) -> BatchBoundingBoxes:  # type: ignore[override]
        if check_dims and tensor.ndim != 2:
            raise ValueError(f"Expected a 2D tensor, got {tensor.ndim}D.")
        if isinstance(format, str):
            format = BoundingBoxFormat[format.upper()]
        batch_bounding_boxes = tensor.as_subclass(cls)
        batch_bounding_boxes.format = format
        batch_bounding_boxes.canvas_size = canvas_size
        batch_bounding_boxes.idx_sample = idx_sample
        return batch_bounding_boxes

    def __new__(
        cls,
        data: Any,
        *,
        format: BoundingBoxFormat | str,
        canvas_size: Tensor,
        idx_sample: Tensor,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> BatchBoundingBoxes:
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        idx_sample = cls._to_tensor(
            idx_sample, device=device, dtype=torch.int32, requires_grad=False
        )
        return cls._wrap(
            tensor, format=format, canvas_size=canvas_size, idx_sample=idx_sample
        )

    @classmethod
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> BatchBoundingBoxes:
        # If there are BatchBoundingBoxes instances in the output, their metadata got lost when we called
        # super().__torch_function__. We need to restore the metadata somehow, so we choose to take
        # the metadata from the first bbox in the parameters.
        # This should be what we want in most cases. When it's not, it's probably a mis-use anyway, e.g.
        # something like some_xyxy_bbox + some_xywh_bbox; we don't guard against those cases.
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))  # type: ignore[operator]
        first_batch_bboxes_from_args = next(
            x for x in flat_params if isinstance(x, BatchBoundingBoxes)
        )
        format, canvas_size, idx_sample = (
            first_batch_bboxes_from_args.format,
            first_batch_bboxes_from_args.canvas_size,
            first_batch_bboxes_from_args.idx_sample,
        )

        if isinstance(output, torch.Tensor) and not isinstance(
            output, BatchBoundingBoxes
        ):
            output = BatchBoundingBoxes._wrap(
                output,
                format=format,
                canvas_size=canvas_size,
                idx_sample=idx_sample,
                check_dims=False,
            )
        elif isinstance(output, (tuple, list)):
            output = type(output)(
                BatchBoundingBoxes._wrap(
                    part,
                    format=format,
                    canvas_size=canvas_size,
                    idx_sample=idx_sample,
                    check_dims=False,
                )
                for part in output
            )
        return output

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(
            format=self.format,
            canvas_size=self.canvas_size,
            idx_sample=self.idx_sample.tolist(),
        )
