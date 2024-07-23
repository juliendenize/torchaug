# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C

from __future__ import annotations

from typing import (
    List,
)

import torch
from torch import Tensor

from torchaug.ta_tensors import (
    BatchImages,
)

from ._ta_nested_tensors import TANestedTensors


class NestedTensors(TANestedTensors[torch.Tensor, torch.Tensor]):
    """Implement Nested Tensor for PyTorch."""

    tensors_type = Tensor
    batch_tensors_type = BatchImages
    tensors: List[Tensor]
