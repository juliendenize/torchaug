# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C

from __future__ import annotations

from typing import Any, Optional, Union

import torch

from ._ta_tensor import TATensor


class Labels(TATensor):
    """Labels is a subclass of TATensor for a tensor that is supposed to
    be concatenated with other tensors to form :class:`torchaug.ta_tensors.BatchLabels`.
    """

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> Labels:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor.as_subclass(cls)
