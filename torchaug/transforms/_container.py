# @Copyright: CEA-LIST/DIASI/SIALV/ (2023-    )
# @Author: CEA-LIST/DIASI/SIALV/ <julien.denize@cea.fr>
# @License: CECILL-C
#
# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from torchaug._utils import _log_api_usage_once
from torchaug.transforms._utils import _assert_list_of_modules

from ._transform import RandomApplyTransform, Transform


class Compose(Transform):
    """Composes several transforms together.

    This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms: list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.SequentialTransform`` as below.

        >>> transforms = torch.nn.SequentialTransform(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``.

    """

    def __init__(self, transforms: Sequence[Callable]) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        elif not transforms:
            raise ValueError("Pass at least one transform")
        self.transforms = transforms

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1
        for transform in self.transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)
        return outputs

    def extra_repr(self) -> str:  # type: ignore[override]
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return "\n".join(format_string)


class RandomApply(Transform):
    """Apply randomly a list of transformations with a given probability.

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms: list of transformations
        p: probability of applying the list of transforms.
    """

    def __init__(self, transforms: Sequence[Union[Callable, nn.ModuleList]], p: float = 0.5) -> None:
        super().__init__()

        if not isinstance(transforms, (Sequence, nn.ModuleList)):
            raise TypeError("Argument transforms should be a sequence of callables or a `nn.ModuleList`")
        self.transforms = transforms

        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        if torch.rand(1) >= self.p:
            return inputs if needs_unpacking else inputs[0]

        for transform in self.transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)
        return outputs

    def extra_repr(self) -> str:  # type: ignore[override]
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return "\n".join(format_string)


class RandomChoice(Transform):
    """Apply single transformation randomly picked from a list.

    This transform does not support torchscript.

    Args:
        transforms: list of transformations
        p: probability of each transform being picked.
            If ``p`` doesn't sum to 1, it is automatically normalized. If ``None``
            (default), all transforms have the same probability.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: Optional[List[float]] = None,
    ) -> None:
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")

        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}")

        super().__init__()

        self.transforms = transforms
        total = sum(p)
        self.p_choices = [prob / total for prob in p]

    def forward(self, *inputs: Any) -> Any:
        idx = int(torch.multinomial(torch.tensor(self.p_choices), 1))
        transform = self.transforms[idx]
        return transform(*inputs)


class RandomOrder(Transform):
    """Apply a list of transformations in a random order.

    This transform does not support torchscript.

    Args:
        transforms: list of transformations
    """

    def __init__(self, transforms: Sequence[Callable]) -> None:
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        super().__init__()
        self.transforms = transforms

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1
        for idx in torch.randperm(len(self.transforms)):
            transform = self.transforms[idx]
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)
        return outputs


class SequentialTransform(Transform):
    """Sequentially apply a list of transforms.

    .. note::
        By default this SequentialTransform makes all its transforms to be inplace, batch_inplace, and
        batch_transform with unlimited chunks.

    Args:
        transforms: A list of transforms.
        transforms_attributes_override: A dictionary of parameters to override the default parameters
            of the transforms if they exist. Useful to make transforms for batches.
    """

    _receive_flatten_inputs = False

    def __init__(
        self,
        transforms: List[RandomApplyTransform],
        transforms_attributes_override: Optional[Dict[str, Any]] = {
            "inplace": True,
            "batch_inplace": True,
            "batch_transform": True,
            "num_chunks": -1,
            "permute_chunks": False,
        },
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        _assert_list_of_modules(transforms)

        self.transforms_attributes_override = transforms_attributes_override

        self._prepare_transforms(transforms)

        self.transforms = nn.ModuleList(transforms)

    def _prepare_transform(self, transform: nn.Module):
        import inspect

        if isinstance(transform, RandomApplyTransform):
            if self.transforms_attributes_override:
                init_signature = inspect.signature(transform.__init__)  # type: ignore[misc]
                parameters = init_signature.parameters
                for key, value in self.transforms_attributes_override.items():
                    has_key = key in parameters
                    if has_key:
                        setattr(transform, key, value)

            transform._receive_flatten_inputs = True

    def _prepare_transforms(self, transforms: List[nn.Module]):
        for transform in transforms:
            self._prepare_transform(transform)
            self._prepare_transforms(list(transform.modules())[1:])

    def forward(self, *inputs: Any) -> Any:
        if not self._receive_flatten_inputs:
            inputs = inputs if len(inputs) > 1 else inputs[0]
            flat_inputs, spec = tree_flatten(inputs)
        else:
            flat_inputs = list(inputs)
        for transform in self.transforms:
            flat_inputs = transform(*flat_inputs)

        if not self._receive_flatten_inputs:
            return tree_unflatten(flat_inputs, spec)

        return flat_inputs

    def extra_repr(self) -> str:  # type: ignore[override]
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return f"transforms_attributes_override={self.transforms_attributes_override},\ntransforms=\n" + "\n".join(
            format_string
        )
