from __future__ import annotations

import torch
import torchvision.transforms.v2.functional as TVF

from torchaug import ta_tensors
from torchaug._utils import _log_api_usage_once

from ._utils._kernel import _get_kernel, _register_kernel_internal


def uniform_temporal_subsample(inpt: torch.Tensor, num_samples: int) -> torch.Tensor:
    """See :class:`~torchaug.transforms.UniformTemporalSubsample` for details."""
    if torch.jit.is_scripting():
        return uniform_temporal_subsample_video(inpt, num_samples=num_samples)

    _log_api_usage_once(uniform_temporal_subsample)

    kernel = _get_kernel(uniform_temporal_subsample, type(inpt))
    return kernel(inpt, num_samples=num_samples)


@_register_kernel_internal(uniform_temporal_subsample, torch.Tensor)
@_register_kernel_internal(uniform_temporal_subsample, ta_tensors.Video)
@_register_kernel_internal(uniform_temporal_subsample, ta_tensors.BatchVideos)
def uniform_temporal_subsample_video(video: torch.Tensor, num_samples: int) -> torch.Tensor:
    return TVF.uniform_temporal_subsample_video(video=video, num_samples=num_samples)
