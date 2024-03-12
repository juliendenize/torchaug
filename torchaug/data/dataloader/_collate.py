from __future__ import annotations

import contextlib
from typing import Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch.utils.data._utils.collate import (
    collate,
    collate_float_fn,
    collate_int_fn,
    collate_numpy_array_fn,
    collate_numpy_scalar_fn,
    collate_str_fn,
    collate_tensor_fn,
    default_collate_err_msg_format,
)

from ...ta_tensors._batch_bounding_boxes import BatchBoundingBoxes, convert_bboxes_to_batch_bboxes
from ...ta_tensors._batch_images import BatchImages
from ...ta_tensors._batch_masks import BatchMasks, convert_masks_to_batch_masks
from ...ta_tensors._batch_videos import BatchVideos
from ...ta_tensors._bounding_boxes import BoundingBoxes
from ...ta_tensors._image import Image
from ...ta_tensors._mask import Mask
from ...ta_tensors._video import Video


def collate_ta_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    elem = batch[0]
    if isinstance(elem, Image):
        return BatchImages(torch.stack(batch, 0))
    elif isinstance(elem, Video):
        return BatchVideos(torch.stack(batch, 0))
    elif isinstance(elem, BoundingBoxes):
        return convert_bboxes_to_batch_bboxes(batch)
    elif isinstance(elem, Mask):
        return convert_masks_to_batch_masks(batch)
    elif isinstance(elem, BatchImages):
        return BatchImages.cat(batch)
    elif isinstance(elem, BatchVideos):
        return BatchVideos.cat(batch)
    elif isinstance(elem, BatchBoundingBoxes):
        return BatchBoundingBoxes.cat(batch)
    elif isinstance(elem, BatchMasks):
        return BatchMasks.cat(batch)
    else:
        raise TypeError(default_collate_err_msg_format.format(type(batch)))


default_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {torch.Tensor: collate_tensor_fn}
with contextlib.suppress(ImportError):
    import numpy as np

    # For both ndarray and memmap (subclass of ndarray)
    default_collate_fn_map[np.ndarray] = collate_numpy_array_fn
    # See scalars hierarchy: https://numpy.org/doc/stable/reference/arrays.scalars.html
    # Skip string scalars
    default_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn
default_collate_fn_map[float] = collate_float_fn
default_collate_fn_map[int] = collate_int_fn
default_collate_fn_map[str] = collate_str_fn
default_collate_fn_map[bytes] = collate_str_fn

for ta_type in [
    Image,
    Video,
    BoundingBoxes,
    Mask,
    BatchBoundingBoxes,
    BatchImages,
    BatchVideos,
    BatchMasks,
]:
    default_collate_fn_map[ta_type] = collate_ta_tensor_fn


def default_collate(batch):
    r"""Take in a batch of data and put the elements within the batch into a
    tensor or ta_tensor with an additional outer dimension - batch size if relevant.

    The exact output type can be a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, :class:`~torchaug.ta_tensors.TATensor`, a `Sequence` of
    :class:`~torchaug.ta_tensors.TATensor`, a Collection of :class:`~torchaug.ta_tensors.TATensor`,
    or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

    Here is the general input type (based on the type of the element within the batch) to output type mapping:

        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * :class:`~torchaug.ta_tensors.Image` -> :class:`~torchaug.ta_tensors.BatchImages`
        * :class:`~torchaug.ta_tensors.Video` -> :class:`~torchaug.ta_tensors.BatchVideos`
        * :class:`~torchaug.ta_tensors.BoundingBoxes` ->
          :class:`~torchaug.ta_tensors._batch_bounding_boxes.BatchBoundingBoxes`
        * :class:`~torchaug.ta_tensors.Mask` -> :class:`~torchaug.ta_tensors.BatchMasks`
        * :class:`~torchaug.ta_tensors.BatchImages` ->
          :class:`~torchaug.ta_tensors.BatchImages`
        * :class:`~torchaug.ta_tensors.BatchVideos` ->
          :class:`~torchaug.ta_tensors.BatchVideos`
        * :class:`~torchaug.ta_tensors._batch_bounding_boxes.BatchBoundingBoxes` ->
          :class:`~torchaug.ta_tensors._batch_bounding_boxes.BatchBoundingBoxes`
        * :class:`~torchaug.ta_tensors.BatchMasks` ->
          :class:`~torchaug.ta_tensors.BatchMasks`
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`

    Args:
        batch: a single batch to be collated

    """
    return collate(batch, collate_fn_map=default_collate_fn_map)
