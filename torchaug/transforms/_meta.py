from typing import Any, Dict, Union

from torchaug import ta_tensors
from . import functional as F, Transform


class ConvertBoundingBoxFormat(Transform):
    """Convert bounding box coordinates to the given ``format``, eg from "CXCYWH" to "XYXY".

    Args:
        format (str or tv_tensors.BoundingBoxFormat): output bounding box format.
            Possible values are defined by :class:`~torchvision.tv_tensors.BoundingBoxFormat` and
            string values match the enums, e.g. "XYXY" or "XYWH" etc.
    """

    _transformed_types = (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes)

    def __init__(self, format: Union[str, ta_tensors.BoundingBoxFormat]) -> None:
        super().__init__()
        self.format = format

    def _transform(
        self,
        inpt: ta_tensors.BoundingBoxes | ta_tensors.BatchBoundingBoxes,
        params: Dict[str, Any],
    ) -> ta_tensors.BoundingBoxes | ta_tensors.BatchBoundingBoxes:
        return F.convert_bounding_box_format(inpt, new_format=self.format)  # type: ignore[return-value, arg-type]


class ClampBoundingBoxes(Transform):
    """Clamp bounding boxes to their corresponding image dimensions.

    The clamping is done according to the bounding boxes' ``canvas_size`` meta-data.

    """

    _transformed_types = (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes)

    def _transform(
        self,
        inpt: ta_tensors.BoundingBoxes | ta_tensors.BatchBoundingBoxes,
        params: Dict[str, Any],
    ) -> ta_tensors.BoundingBoxes | ta_tensors.BatchBoundingBoxes:
        return F.clamp_bounding_boxes(inpt)  # type: ignore[return-value]
