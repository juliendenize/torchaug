from __future__ import annotations

from typing import Any, List

from torchaug import ta_tensors


def get_bounding_boxes(flat_inputs: List[Any]) -> ta_tensors.BoundingBoxes:
    """Get the bounding boxes from a list of inputs.

    Args:
        flat_inputs (List[Any]): The list of inputs.

    Returns:
        ta_tensors.BoundingBoxes: The bounding boxes.
    """
    # This assumes there is only one bbox per sample as per the general convention
    try:
        return next(
            inpt for inpt in flat_inputs if isinstance(inpt, (ta_tensors.BoundingBoxes))
        )
    except StopIteration:
        raise ValueError("No bounding boxes were found in the sample")


def get_batch_bounding_boxes(flat_inputs: List[Any]) -> ta_tensors.BatchBoundingBoxes:
    """Get the batch of bounding boxes from a list of inputs.

    Args:
        flat_inputs (List[Any]): The list of inputs.

    Returns:
        ta_tensors.BatchBoundingBoxes: The batch of bounding boxes.
    """
    # This assumes there is only one bbox per sample as per the general convention
    try:
        return next(
            inpt
            for inpt in flat_inputs
            if isinstance(inpt, ta_tensors.BatchBoundingBoxes)
        )
    except StopIteration:
        raise ValueError("No batch of bounding boxes were found in the sample")
