# ==================================
# Copyright: CEA-LIST/DIASI/SIALV/
# Author : Torchaug Developers
# License: CECILL-C
# ==================================

from ._batch_bounding_boxes import BatchBoundingBoxes
from ._batch_images import BatchImages
from ._batch_labels import BatchLabels
from ._batch_masks import BatchMasks
from ._batch_videos import BatchVideos
from ._bounding_boxes import BoundingBoxes
from ._image import Image
from ._labels import Labels
from ._mask import Mask
from ._video import Video
from .nested import (
    BoundingBoxesNestedTensors,
    ImageNestedTensors,
    LabelsNestedTensors,
    MaskNestedTensors,
    NestedTensors,
    VideoNestedTensors,
)


_SAMPLE_TA_TENSORS = [Image, Mask, BoundingBoxes, Labels, Video]
_BATCH_TA_TENSORS = [BatchImages, BatchMasks, BatchBoundingBoxes, BatchLabels, BatchVideos]
_NESTED_TA_TENSORS = [
    NestedTensors,
    MaskNestedTensors,
    ImageNestedTensors,
    BoundingBoxesNestedTensors,
    LabelsNestedTensors,
    VideoNestedTensors,
]
