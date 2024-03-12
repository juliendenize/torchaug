# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C
#
# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

from __future__ import annotations

import torch
import torchvision.transforms.v2.functional as TVF

from torchaug import ta_tensors
from torchaug._utils import _log_api_usage_once

from ._utils._kernel import _get_kernel, _register_kernel_internal


def erase(
    inpt: torch.Tensor,
    i: int,
    j: int,
    h: int,
    w: int,
    v: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """See :class:`~torchaug.transforms.RandomErasing` for details."""
    if torch.jit.is_scripting():
        return erase_image(inpt, i=i, j=j, h=h, w=w, v=v, inplace=inplace)

    _log_api_usage_once(erase)

    kernel = _get_kernel(erase, type(inpt))
    return kernel(inpt, i=i, j=j, h=h, w=w, v=v, inplace=inplace)


@_register_kernel_internal(erase, torch.Tensor)
@_register_kernel_internal(erase, ta_tensors.Image)
@_register_kernel_internal(erase, ta_tensors.BatchImages)
def erase_image(
    image: torch.Tensor,
    i: int,
    j: int,
    h: int,
    w: int,
    v: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    return TVF.erase_image(image=image, i=i, j=j, h=h, w=w, v=v, inplace=inplace)


@_register_kernel_internal(erase, ta_tensors.Video)
@_register_kernel_internal(erase, ta_tensors.BatchVideos)
def erase_video(
    video: torch.Tensor,
    i: int,
    j: int,
    h: int,
    w: int,
    v: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    return erase_image(image=video, i=i, j=j, h=h, w=w, v=v, inplace=inplace)
