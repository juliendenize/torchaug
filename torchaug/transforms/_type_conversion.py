from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import torch

from torchaug import ta_tensors

from . import functional as F
from ._transform import Transform
from ._utils import is_pure_tensor


class ToImage(Transform):
    """Convert a tensor, ndarray, to :class:`~torchaug.ta_tensors.Image`
    ; this does not scale values.

    This transform does not support torchscript.
    """

    _transformed_types = (is_pure_tensor, np.ndarray, ta_tensors.BatchImages)

    def _transform(self, inpt: Union[torch.Tensor, np.ndarray], params: Dict[str, Any]) -> ta_tensors.Image:
        return F.to_image(inpt)


class ToBatchImages(Transform):
    """Convert a tensor, ndarray to :class:`~torchaug.ta_tensors.Image`
    ; this does not scale values.

    This transform does not support torchscript.
    """

    _transformed_types = (is_pure_tensor, ta_tensors.Image)

    def _transform(
        self,
        inpt: Union[torch.Tensor, ta_tensors.Image, np.ndarray],
        params: Dict[str, Any],
    ) -> ta_tensors.BatchImages:
        return F.to_batch_images(inpt)


class ToPureTensor(Transform):
    """Convert all TATensors to pure tensors, removing associated metadata (if any).

    This doesn't scale or change the values, only the type.
    """

    _transformed_types = (ta_tensors.TATensor,)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> torch.Tensor:
        return inpt.as_subclass(torch.Tensor)
