# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C

from __future__ import annotations

from typing import (
    List,
)

from torchaug.ta_tensors import (
    BatchBoundingBoxes,
    BoundingBoxes,
)
from torchaug.ta_tensors._batch_bounding_boxes import convert_bboxes_to_batch_bboxes

from ._ta_nested_tensors import TANestedTensors


class BoundingBoxesNestedTensors(TANestedTensors[BoundingBoxes, BatchBoundingBoxes]):
    """Implement BoundingBoxes Nested Tensor for PyTorch."""

    tensors_type = BoundingBoxes
    batch_tensors_type = BatchBoundingBoxes
    tensors: List[BoundingBoxes]

    def to_batch(self) -> BatchBoundingBoxes:
        """Return the batched tensor and the shapes of the nested tensors."""
        return convert_bboxes_to_batch_bboxes(self.tensors)
