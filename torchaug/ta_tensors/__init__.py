# =======================================================
# Copyright: CEA-LIST/DIASI/SIALV/
# Author : CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# License: CECILL-C
# =======================================================

# ruff: noqa: F401
# ruff: noqa: D104

from ._batch_bounding_boxes import (
    BatchBoundingBoxes,
    convert_batch_bboxes_to_bboxes,
    convert_bboxes_to_batch_bboxes,
)
from ._batch_concatenated_ta_tensor import _BatchConcatenatedTATensor
from ._batch_images import BatchImages
from ._batch_labels import (
    BatchLabels,
    convert_batch_labels_to_labels,
    convert_labels_to_batch_labels,
)
from ._batch_masks import (
    BatchMasks,
    convert_batch_masks_to_masks,
    convert_masks_to_batch_masks,
)
from ._batch_videos import BatchVideos
from ._bounding_boxes import BoundingBoxes, BoundingBoxFormat
from ._image import Image
from ._labels import Labels
from ._mask import Mask
from ._ta_tensor import TATensor
from ._torch_function_helpers import set_return_type
from ._video import Video
from ._wrap import wrap


from .nested import (  # isort:skip
    BoundingBoxesNestedTensors,
    ImageNestedTensors,
    LabelsNestedTensors,
    MaskNestedTensors,
    NestedTensors,
    TANestedTensors,
    VideoNestedTensors,
)

from ._registry import (  # isort:skip
    _BATCH_TA_TENSORS,
    _NESTED_TA_TENSORS,
    _SAMPLE_TA_TENSORS,
)
