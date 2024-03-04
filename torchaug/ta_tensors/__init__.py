import torch

from ._batch_bounding_boxes import (
    BatchBoundingBoxes,
    convert_batch_bboxes_to_bboxes,
    convert_bboxes_to_batch_bboxes,
)
from ._batch_images import BatchImages
from ._batch_masks import BatchMasks
from ._batch_videos import BatchVideos
from ._bounding_boxes import BoundingBoxes, BoundingBoxFormat
from ._image import Image
from ._mask import Mask
from ._ta_tensor import TATensor
from ._torch_function_helpers import set_return_type
from ._video import Video


@torch.compiler.disable
def wrap(wrappee, *, like, **kwargs) -> TATensor:
    """Convert a :class:`torch.Tensor` (``wrappee``) into the same :class:`~torchvision.tv_tensors.TVTensor` subclass as ``like``.

    If ``like`` is a :class:`~torchvision.tv_tensors.BoundingBoxes`, the ``format`` and ``canvas_size`` of
    ``like`` are assigned to ``wrappee``, unless they are passed as ``kwargs``.

    Args:
        wrappee (Tensor): The tensor to convert.
        like (:class:`~torchvision.tv_tensors.TVTensor`): The reference.
            ``wrappee`` will be converted into the same subclass as ``like``.
        kwargs: Can contain "format" and "canvas_size" if ``like`` is a :class:`~torchvision.tv_tensor.BoundingBoxes`.
            Ignored otherwise.
    """
    if isinstance(like, BoundingBoxes):
        return BoundingBoxes._wrap(
            wrappee,
            format=kwargs.get("format", like.format),
            canvas_size=kwargs.get("canvas_size", like.canvas_size),
        )
    elif isinstance(like, BatchBoundingBoxes):
        return BatchBoundingBoxes._wrap(
            wrappee,
            format=kwargs.get("format", like.format),
            canvas_size=kwargs.get("canvas_size", like.canvas_size),
            idx_sample=kwargs.get("idx_sample", like.idx_sample),
        )
    elif isinstance(like, BatchMasks):
        return BatchMasks._wrap(
            wrappee,
            idx_sample=kwargs.get("idx_sample", like.idx_sample),
        )
    else:
        return wrappee.as_subclass(type(like))
