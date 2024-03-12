# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C
#
# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from torchaug import ta_tensors


@torch.jit.unused
def to_image(inpt: Union[torch.Tensor, np.ndarray]) -> ta_tensors.Image:
    """See :class:`~torchaug.transforms.ToImage` for details."""
    if isinstance(inpt, np.ndarray):
        output = torch.from_numpy(np.atleast_3d(inpt)).permute((2, 0, 1)).contiguous()
    elif isinstance(inpt, torch.Tensor):
        output = inpt
    else:
        raise TypeError(f"Input can either be a pure Tensor, a numpy array, but got {type(inpt)} instead.")
    return ta_tensors.Image(output)


@torch.jit.unused
def to_batch_images(inpt: torch.Tensor) -> ta_tensors.BatchImages:
    """See :class:`~torchaug.transforms.ToBatchImages` for details."""
    if isinstance(inpt, torch.Tensor):
        output = inpt
    else:
        raise TypeError("Input should be a Tensor.")
    return ta_tensors.BatchImages(output)
