from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil, floor

import torch
from torch import Tensor, nn


class BatchRandomTransform(nn.Module, ABC):
    """Abstract class to make a base class for all batch random transforms.

    args:
        p: Probability to apply the transform.
        inplace: If True, perform inplace operation to save memory and time.
    """

    def __init__(self, p: float, inplace: bool) -> None:
        super().__init__()

        if not 0 <= p <= 1:
            raise ValueError(
                f"p should be superior or equal to 0 and inferior or equal to 1. Got {p}."
            )

        self.p = p
        self.inplace = inplace

    @abstractmethod
    def apply_transform(self, imgs: Tensor) -> Tensor:
        """Function to perform transformation on the batch of images.

        .. note::
            Should be overridden by subclasses.

        Args:
            imgs: Batch of images to transform.

        Return:
            The transformed batch of images.
        """
        ...

    def forward(self, imgs: Tensor) -> Tensor:
        """Call :meth:`apply_transform` on a randomly selected sub samples of passed images.

        Args:
            imgs: Images to randomly transform based on :attr:`~p`.

        Return:
            The randomly transformed images.
        """
        if self.p == 0:
            return imgs

        elif self.p == 1:
            output = self.apply_transform(imgs if self.inplace else imgs.clone())
            return output

        batch_size = imgs.shape[0]
        p_mul_batch_size = self.p * batch_size
        floor_apply = floor(p_mul_batch_size)
        ceil_apply = ceil(p_mul_batch_size)

        output: Tensor = imgs if self.inplace else imgs.clone()

        # If 0 < p_mul_batch_size < 1, then only one element from input is augmented
        # with p probability.
        if floor_apply == 0 or ceil_apply == 0:
            num_apply = 1 if torch.rand(1).item() < self.p else 0
        elif floor_apply == ceil_apply:
            num_apply = floor_apply
        # If p_mul_batch_size is rational, then upper or lower integer p_mul_batch_size
        # elements from input are augmented randomly depending with the decimal.
        else:
            decimal = p_mul_batch_size % 1
            num_apply = floor_apply if decimal < torch.rand(1).item() else ceil_apply

        # If no augmentation return the output directly, keep consistency of inplace.
        if num_apply == 0:
            return output
        elif num_apply == 1:
            indices_do_apply = torch.randint(0, batch_size, (1,), device=imgs.device)
        elif num_apply > 1:
            indices_do_apply = torch.randperm(batch_size, device=imgs.device)[
                :num_apply
            ]

        output[indices_do_apply] = self.apply_transform(output[indices_do_apply])

        return output
