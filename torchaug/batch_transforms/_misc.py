from __future__ import annotations

import numbers
from typing import Sequence

import torch
from torch import Tensor, nn
from torchvision.transforms.transforms import _setup_size

import torchaug.batch_transforms.functional as F_b
from torchaug.transforms._utils import (_assert_module_or_list_of_modules,
                                        transfer_tensor_on_device)
from torchaug.utils import _log_api_usage_once

from ._transform import BatchRandomTransform


class BatchRandomGaussianBlur(BatchRandomTransform):
    """Blurs batch of images with randomly chosen Gaussian blur.

    The batch of images is expected to be of shape [B, ..., C, H, W]
    where ... means an arbitrary number of dimensions.

    Args:
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation to be used for creating kernel to perform blurring.
            If float, sigma is fixed. If it is tuple of float (min, max), sigma
            is chosen uniformly at random to lie in the given range.
        p: Probability to apply gaussian blur.
        inplace: If True, perform inplace operation to save memory and time.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization or large batch size.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        sigma: float | tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
        inplace: bool = False,
        value_check: bool = False,
    ):
        super().__init__(
            p=p,
            inplace=inplace,
        )
        _log_api_usage_once(self)

        self.kernel_size = _setup_size(
            kernel_size, "Kernel size should be a tuple/list of two integers."
        )
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError(
                    "Kernel size value should be an odd and positive number."
                )

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError(
                    "sigma values should be positive and of the form (min, max)."
                )
        else:
            raise ValueError(
                "sigma should be a single number or a list/tuple with length 2."
            )

        self.register_buffer("sigma", torch.as_tensor(sigma))
        self.value_check = value_check

    @staticmethod
    def get_params(sigma_min: Tensor, sigma_max: Tensor, batch_size: int) -> Tensor:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min: Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max: Maximum standard deviation that can be chosen for blurring kernel.
            batch_size: The number of samples to draw.

        Returns:
            Standard deviation to calculate kernel for gaussian blurring.
        """
        dtype = sigma_min.dtype
        device = sigma_min.device
        return (
            torch.rand(batch_size, 1, dtype=dtype, device=device).expand(batch_size, 2)
            * (sigma_max - sigma_min)
            + sigma_min
        )

    def apply_transform(self, imgs: Tensor) -> Tensor:
        """Blur the batch of images.

        Args:
            imgs: Batch of images to be blurred.
        Returns:
            Gaussian blurred images
        """
        sigma: Tensor = self.get_params(self.sigma[0], self.sigma[1], imgs.shape[0])
        imgs = F_b.batch_gaussian_blur(imgs, self.kernel_size, sigma, self.value_check)
        return imgs

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"kernel_size={self.kernel_size}, "
            f"sigma={self.sigma.tolist()}, "
            f"p={self.p}, "
            f"inplace={self.inplace}, "
            f"value_check={self.value_check})"
        )
        return s


class BatchMixUp(nn.Module):
    """Mix input tensor with linear interpolation drawn according a Beta law.

    The shape of the tensors is expected to be [B, ...] with ... any number of dimensions.
    The tensor should be float.

    .. note::
        The tensor is rolled according its first dimension and mixed with one
        drawn interpolation parameter per element in first dimension.

    Args:
        alpha: Parameter for the Beta law.
        inplace: Whether to perform the operation inplace.
    """

    def __init__(self, alpha: float, inplace: bool = False) -> None:
        super().__init__()
        _log_api_usage_once(self)

        self.alpha = alpha
        self.inplace = inplace
        self.mix_sampler = torch.distributions.Beta(
            torch.tensor([alpha]), torch.tensor([alpha])
        )

    def _get_params(self, batch_size: int, device: torch.device) -> Tensor:
        """Draw the mixing coefficients.

        Returns:
            The mixing coefficients.
        """
        return transfer_tensor_on_device(self.mix_sampler.sample((batch_size,)), device)

    def forward(
        self, tensor: Tensor, labels: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Mix the input tensor and labels.

        Args:
            tensor: The tensor to mix.
            labels: If not None, the labels to mix.

        Returns:
            Tuple:
            - mixed tensor.
            - mixed labels or None.
            - mixing coefficients.
        """
        lam = self._get_params(tensor.shape[0], tensor.device)

        tensor = tensor if self.inplace else tensor.clone()

        if labels is None:
            return F_b.batch_mixup(tensor, tensor.roll(1, 0), lam, True), None, lam

        labels = labels if self.inplace else labels.clone()

        return (
            F_b.batch_mixup(tensor, tensor.roll(1, 0), lam, True),
            F_b.batch_mixup(labels, labels.roll(1, 0), lam, True),
            lam,
        )

    def __repr__(self):
        return f"{__class__.__name__}(alpha={self.alpha}, inplace={self.inplace})"


class BatchRandomApply(BatchRandomTransform):
    """Apply randomly a list of transformations to a batch of images with a given probability.

    Args:
        transforms: List of transformations.
        p: Probability to apply the transform.
        inplace: If ``True``, perform inplace operation to save memory.
    """

    def __init__(
        self,
        transforms: list[nn.Module] | nn.Module,
        p: float = 0.5,
        inplace: bool = False,
    ) -> None:
        super().__init__(p=p, inplace=inplace)
        _log_api_usage_once(self)

        _assert_module_or_list_of_modules(transforms)

        if isinstance(transforms, nn.Module):
            transforms = [transforms]

        self.transforms = nn.ModuleList(transforms)

        self.inplace = inplace

    def apply_transform(self, imgs: Tensor) -> Tensor:
        """Apply the :attr:`~transforms` on the images.

        Args:
            imgs: Batch of images to transform.

        Returns:
            Transformed batch of images.
        """
        for t in self.transforms:
            imgs[:] = t(imgs)

        return imgs

    def __repr__(self) -> str:
        transforms_repr = str(self.transforms).replace("\n", "\n    ")
        return (
            f"{self.__class__.__name__}("
            f"\n    p={self.p},"
            f"\n    inplace={self.inplace},"
            f"\n    transforms={transforms_repr}"
            f"\n)"
        )
