from __future__ import annotations

import numpy as np
import torch

from torchaug import ta_tensors


@torch.jit.unused
def to_image(inpt: torch.Tensor | np.ndarray) -> ta_tensors.Image:
    """See :class:`~torchvision.transforms.v2.ToImage` for details."""
    if isinstance(inpt, np.ndarray):
        output = torch.from_numpy(inpt).permute((2, 0, 1)).contiguous()
    elif isinstance(inpt, torch.Tensor):
        output = inpt
    else:
        raise TypeError(
            f"Input can either be a pure Tensor, a numpy array, but got {type(inpt)} instead."
        )
    return ta_tensors.Image(output)


@torch.jit.unused
def to_batch_images(inpt: torch.Tensor | np.ndarray) -> ta_tensors.BatchImages:
    """See :class:`~torchvision.transforms.v2.ToImage` for details."""
    if isinstance(inpt, np.ndarray):
        output = torch.from_numpy(inpt).permute((3, 1, 0)).contiguous()
    elif isinstance(inpt, torch.Tensor):
        output = inpt
    else:
        raise TypeError(
            f"Input can either be a pure Tensor, a numpy array, but got {type(inpt)} instead."
        )
    return ta_tensors.BatchImages(output)
