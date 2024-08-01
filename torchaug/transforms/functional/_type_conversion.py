# =======================================================
# Copyright: CEA-LIST/DIASI/SIALV/
# Author : CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# License: CECILL-C
# =======================================================

# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

from __future__ import annotations

from typing import Any, List, Union

import numpy as np
import torch
from PIL.Image import Image as PILImage
from torchvision.transforms.v2.functional import pil_to_tensor

from torchaug import ta_tensors


@torch.jit.unused
def to_image(inpt: Union[torch.Tensor, np.ndarray, PILImage]) -> ta_tensors.Image:
    """See :class:`~torchaug.transforms.ToImage` for details."""
    if isinstance(inpt, np.ndarray):
        output = torch.from_numpy(np.atleast_3d(inpt)).permute((2, 0, 1)).contiguous()
    elif isinstance(inpt, PILImage):
        output = pil_to_tensor(inpt)
    elif isinstance(inpt, torch.Tensor):
        output = inpt
    else:
        raise TypeError(
            f"Input can either be a pure Tensor, a numpy array, a PIL image, but got {type(inpt)} instead."
        )
    return ta_tensors.Image(output)


@torch.jit.unused
def to_batch_images(inpt: torch.Tensor) -> ta_tensors.BatchImages:
    """See :class:`~torchaug.transforms.ToBatchImages` for details."""
    if isinstance(inpt, torch.Tensor):
        output = inpt
    else:
        raise TypeError("Input should be a Tensor.")
    return ta_tensors.BatchImages(output)


@torch.jit.unused
def nested_to_list(inpt: ta_tensors.TANestedTensors) -> List:
    """Convert a nested structure of Tensors to a list."""
    if isinstance(inpt, ta_tensors.TANestedTensors):
        return inpt.to_list()
    else:
        raise TypeError(f"Inpt should be a TANestedTensors, but got {type(inpt)} instead.")


@torch.jit.unused
def nested_to_batch(inpt: ta_tensors.TANestedTensors) -> Any:
    """Convert a nested structure of Tensors to a batch of Tensors."""
    if isinstance(inpt, ta_tensors.TANestedTensors):
        return inpt.to_batch()
    else:
        raise TypeError(f"Inpt should be a TANestedTensors, but got {type(inpt)} instead.")
