# ==================================
# Copyright: CEA-LIST/DIASI/SIALV/
# Author : Torchaug Developers
# License: CECILL-C
# ==================================

from __future__ import annotations

from typing import (
    List,
)

from torchaug.ta_tensors import (
    BatchMasks,
    Mask,
)
from torchaug.ta_tensors._batch_masks import convert_masks_to_batch_masks

from ._ta_nested_tensors import TANestedTensors


class MaskNestedTensors(TANestedTensors[Mask, BatchMasks]):
    """Implement Masks Nested Tensor for PyTorch."""

    tensors_type = Mask
    batch_tensors_type = BatchMasks
    tensors: List[Mask]

    def to_batch(self) -> BatchMasks:
        """Return the batched mask of the nested masks."""
        return convert_masks_to_batch_masks(self.tensors)
