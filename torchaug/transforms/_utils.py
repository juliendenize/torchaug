from __future__ import annotations

import collections
from contextlib import suppress
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torchvision._utils import sequence_to_str
from torchvision.transforms.v2._utils import check_type

from torchaug import ta_tensors
from torchaug.transforms.functional import get_dimensions, get_size
from torchaug.transforms.functional._utils import is_pure_tensor


def _assert_module_or_list_of_modules(collection: Any):
    if not isinstance(collection, nn.Module) and (
        not isinstance(collection, Sequence)
        or not all(isinstance(obj, nn.Module) for obj in collection)
        or len(collection) == 0
    ):
        raise TypeError("collection should be a module or a list of modules.")


def _assert_list_of_modules(collection: Any):
    if (
        not isinstance(collection, Sequence)
        or not all(isinstance(obj, nn.Module) for obj in collection)
        or len(collection) == 0
    ):
        raise TypeError("Collection should be a list of modules.")


def _find_labels_default_heuristic(inputs: Any) -> torch.Tensor:
    """Find the labels tensor in the inputs.

    This heuristic covers three cases:

    1. The input is tuple or list whose second item is a labels tensor. This happens for already batched
       classification inputs for MixUp and CutMix (typically after the Dataloder).
    2. The input is a tuple or list whose second item is a dictionary that contains the labels tensor
       under a label-like (see below) key. This happens for the inputs of detection models.
    3. The input is a dictionary that is structured as the one from 2.

    What is "label-like" key? We first search for an case-insensitive match of 'labels' inside the keys of the
    dictionary. This is the name our detection models expect. If we can't find that, we look for a case-insensitive
    match of the term 'label' anywhere inside the key, i.e. 'FooLaBeLBar'. If we can't find that either, the dictionary
    contains no "label-like" key.
    """
    if isinstance(inputs, (tuple, list)):
        inputs = inputs[1]

    # MixUp, CutMix
    if is_pure_tensor(inputs):
        return inputs

    if not isinstance(inputs, collections.abc.Mapping):
        raise ValueError(
            f"When using the default labels_getter, the input passed to forward must be a dictionary or a two-tuple "
            f"whose second item is a dictionary or a tensor, but got {inputs} instead."
        )

    candidate_key = None
    with suppress(StopIteration):
        candidate_key = next(key for key in inputs.keys() if key.lower() == "labels")
    if candidate_key is None:
        with suppress(StopIteration):
            candidate_key = next(key for key in inputs.keys() if "label" in key.lower())
    if candidate_key is None:
        raise ValueError(
            "Could not infer where the labels are in the sample. Try passing a callable as the "
            "labels_getter parameter?"
            "If there are no labels in the sample by design, pass labels_getter=None."
        )

    return inputs[candidate_key]


def _parse_labels_getter(
    labels_getter: Optional[Union[str, Callable[[Any], Optional[torch.Tensor]]]],
) -> Callable[[Any], Optional[torch.Tensor]]:
    if labels_getter == "default":
        return _find_labels_default_heuristic
    elif callable(labels_getter):
        return labels_getter
    elif labels_getter is None:
        return lambda _: None
    else:
        raise ValueError(f"labels_getter should either be 'default', a callable, or None, but got {labels_getter}.")


def get_bounding_boxes(flat_inputs: List[Any]) -> ta_tensors.BoundingBoxes:
    """Get the bounding boxes from a list of inputs.

    Args:
        flat_inputs: The list of inputs.

    Returns:
        The bounding boxes.
    """
    # This assumes there is only one bbox per sample as per the general convention
    try:
        return next(inpt for inpt in flat_inputs if isinstance(inpt, (ta_tensors.BoundingBoxes)))
    except StopIteration:
        raise ValueError("No bounding boxes were found in the sample")


def get_batch_bounding_boxes(flat_inputs: List[Any]) -> ta_tensors.BatchBoundingBoxes:
    """Get the batch of bounding boxes from a list of inputs.

    Args:
        flat_inputs: The list of inputs.

    Returns:
        The batch of bounding boxes.
    """
    # This assumes there is only one bbox per sample as per the general convention
    try:
        return next(inpt for inpt in flat_inputs if isinstance(inpt, ta_tensors.BatchBoundingBoxes))
    except StopIteration:
        raise ValueError("No batch of bounding boxes were found in the sample")


def get_sample_or_batch_bounding_boxes(
    flat_inputs: List[Any],
) -> Union[ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes]:
    """Get the bounding boxes from a list of inputs.

    Args:
        flat_inputs: The list of inputs.

    Returns:
        The bounding boxes.
    """
    # This assumes there is only one bbox per sample as per the general convention
    try:
        return next(
            inpt for inpt in flat_inputs if isinstance(inpt, (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes))
        )
    except StopIteration:
        raise ValueError("No bounding boxes were found in the sample")


def query_size(flat_inputs: List[Any]) -> Tuple[int, int]:
    sizes = {
        tuple(get_size(inpt))
        for inpt in flat_inputs
        if check_type(
            inpt,
            (
                is_pure_tensor,
                ta_tensors.Image,
                ta_tensors.BatchImages,
                ta_tensors.Video,
                ta_tensors.BatchVideos,
                ta_tensors.Mask,
                ta_tensors.BatchMasks,
                ta_tensors.BoundingBoxes,
                ta_tensors.BatchBoundingBoxes,
            ),
        )
    }
    if not sizes:
        raise TypeError("No image, video, mask or bounding box was found in the sample")
    elif len(sizes) > 1:
        raise ValueError(f"Found multiple HxW dimensions in the sample: {sequence_to_str(sorted(sizes))}")
    h, w = sizes.pop()
    return h, w


def query_chw(flat_inputs: List[Any]) -> Tuple[int, int, int]:
    chws = {
        tuple(get_dimensions(inpt))
        for inpt in flat_inputs
        if check_type(
            inpt,
            (
                is_pure_tensor,
                ta_tensors.Image,
                ta_tensors.Video,
                ta_tensors.BatchImages,
                ta_tensors.BatchVideos,
            ),
        )
    }
    if not chws:
        raise TypeError("No image or video was found in the sample")
    elif len(chws) > 1:
        raise ValueError(f"Found multiple CxHxW dimensions in the sample: {sequence_to_str(sorted(chws))}")
    c, h, w = chws.pop()
    return c, h, w
