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
    BatchVideos,
    Video,
)

from ._ta_nested_tensors import TANestedTensors


class VideoNestedTensors(TANestedTensors[Video, BatchVideos]):
    """Implement Video Nested Tensor for PyTorch."""

    tensors_type = Video
    batch_tensors_type = BatchVideos
    tensors: List[Video]
