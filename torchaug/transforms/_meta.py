# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C
#
# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

from __future__ import annotations

from typing import Any, Dict, Union

from torchaug import ta_tensors

from . import functional as F
from ._transform import Transform


class ConvertBoundingBoxFormat(Transform):
    """Convert bounding box coordinates to the given ``format``, eg from "CXCYWH" to "XYXY".

    Args:
        format: output bounding box format.
            Possible values are defined by :class:`~torchaug.ta_tensors.BoundingBoxFormat` and
            string values match the enums, e.g. "XYXY" or "XYWH" etc.
    """

    _transformed_types = (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes)

    def __init__(self, format: Union[str, ta_tensors.BoundingBoxFormat]) -> None:
        super().__init__()
        self.format = format

    def _transform(
        self,
        inpt: Union[ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes],
        params: Dict[str, Any],
    ) -> Union[ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes]:
        return F.convert_bounding_box_format(inpt, new_format=self.format)  # type: ignore[return-value, arg-type]


class ClampBoundingBoxes(Transform):
    """Clamp bounding boxes to their corresponding image dimensions.

    The clamping is done according to the bounding boxes' ``canvas_size`` meta-data.
    """

    _transformed_types = (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes)

    def __init__(self) -> None:
        super().__init__()

    def _transform(
        self,
        inpt: Union[ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes],
        params: Dict[str, Any],
    ) -> Union[ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes]:
        return F.clamp_bounding_boxes(inpt)  # type: ignore[return-value]
