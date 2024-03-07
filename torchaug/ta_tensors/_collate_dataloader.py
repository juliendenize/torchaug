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

from ._batch_bounding_boxes import BatchBoundingBoxes, convert_bboxes_to_batch_bboxes
from ._batch_images import BatchImages
from ._batch_masks import BatchMasks, convert_masks_to_batch_masks
from ._batch_videos import BatchVideos
from ._bounding_boxes import BoundingBoxes
from ._image import Image
from ._mask import Mask
from ._video import Video


def collate_ta_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None
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
        print("yoohoho")
        return BatchMasks.cat(batch)
    else:
        raise TypeError(default_collate_err_msg_format.format(type(batch)))


torchaug_default_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {
    torch.Tensor: collate_tensor_fn
}
with contextlib.suppress(ImportError):
    import numpy as np

    # For both ndarray and memmap (subclass of ndarray)
    torchaug_default_collate_fn_map[np.ndarray] = collate_numpy_array_fn
    # See scalars hierarchy: https://numpy.org/doc/stable/reference/arrays.scalars.html
    # Skip string scalars
    torchaug_default_collate_fn_map[
        (np.bool_, np.number, np.object_)
    ] = collate_numpy_scalar_fn
torchaug_default_collate_fn_map[float] = collate_float_fn
torchaug_default_collate_fn_map[int] = collate_int_fn
torchaug_default_collate_fn_map[str] = collate_str_fn
torchaug_default_collate_fn_map[bytes] = collate_str_fn

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
    torchaug_default_collate_fn_map[ta_type] = collate_ta_tensor_fn


def default_collate(batch):
    r"""
    Take in a batch of data and put the elements within the batch into a tensor with an additional outer dimension - batch size.

    The exact output type can be a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

    Here is the general input type (based on the type of the element within the batch) to output type mapping:

        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, torchaug_default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[torchaug_default_collate([V1_1, V1_2, ...]),
          torchaug_default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[torchaug_default_collate([V1_1, V1_2, ...]),
          torchaug_default_collate([V2_1, V2_2, ...]), ...]`

    Args:
        batch: a single batch to be collated

    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example with a batch of `int`s:
        >>> torchaug_default_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> torchaug_default_collate(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> torchaug_default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> torchaug_default_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> torchaug_default_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Example with `List` inside the batch:
        >>> torchaug_default_collate([[0, 1], [2, 3]])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Two options to extend `torchaug_default_collate` to handle specific type
        >>> # Option 1: Write custom collate function and invoke `torchaug_default_collate`
        >>> def custom_collate(batch):
        ...     elem = batch[0]
        ...     if isinstance(elem, CustomType):  # Some custom condition
        ...         return ...
        ...     else:  # Fall back to `torchaug_default_collate`
        ...         return torchaug_default_collate(batch)
        >>> # Option 2: In-place modify `torchaug_default_collate_fn_map`
        >>> def collate_customtype_fn(batch, *, collate_fn_map=None):
        ...     return ...
        >>> torchaug_default_collate_fn_map.update(CustoType, collate_customtype_fn)
        >>> torchaug_default_collate(batch)  # Handle `CustomType` automatically
    """
    return collate(batch, collate_fn_map=torchaug_default_collate_fn_map)
