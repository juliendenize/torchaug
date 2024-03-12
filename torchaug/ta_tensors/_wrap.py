# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C
#
# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

import torch

from ._batch_bounding_boxes import BatchBoundingBoxes
from ._batch_masks import BatchMasks
from ._bounding_boxes import BoundingBoxes
from ._ta_tensor import TATensor


@torch.compiler.disable
def wrap(wrappee, *, like, **kwargs) -> TATensor:
    """Convert a :class:`torch.Tensor` (``wrappee``) into the same :class:`~torchaug.ta_tensors.TATensor`
    subclass as ``like``.

    If ``like`` is a :class:`torchaug.ta_tensors.BoundingBoxes`, the ``format`` and ``canvas_size`` of
    ``like`` are assigned to ``wrappee``, unless they are passed as ``kwargs``.

    Args:
        wrappee (Tensor): The tensor to convert.
        like (:class:`~torchaug.ta_tensors.TATensor`): The reference.
            ``wrappee`` will be converted into the same subclass as ``like``.
        kwargs: Can contain "format" and "canvas_size" if ``like`` is a :class:`torchaug.ta_tensors.BoundingBoxes`.
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
