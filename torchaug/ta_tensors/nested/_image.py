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
    BatchImages,
    Image,
)

from ._ta_nested_tensors import TANestedTensors


class ImageNestedTensors(TANestedTensors[Image, BatchImages]):
    """Implement Image Nested Tensor for PyTorch."""

    tensors_type = Image
    batch_tensors_type = BatchImages
    tensors: List[Image]
