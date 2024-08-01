# =======================================================
# Copyright: CEA-LIST/DIASI/SIALV/
# Author : CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# License: CECILL-C
# =======================================================

from __future__ import annotations

from typing import (
    List,
)

from torchaug.ta_tensors import (
    BatchLabels,
    Labels,
)
from torchaug.ta_tensors._batch_labels import convert_labels_to_batch_labels

from ._ta_nested_tensors import TANestedTensors


class LabelsNestedTensors(TANestedTensors[Labels, BatchLabels]):
    """Implement Labels Nested Tensor for PyTorch."""

    tensors_type = Labels
    batch_tensors_type = BatchLabels
    tensors: List[Labels]

    def to_batch(self) -> BatchLabels:
        """Return the batched labels and the shapes of the nested labels."""
        return convert_labels_to_batch_labels(self.tensors)
