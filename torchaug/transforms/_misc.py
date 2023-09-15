from __future__ import annotations

import numbers
from typing import Sequence

import torch
from torch import Tensor, nn
from torchvision.transforms.transforms import _setup_size

import torchaug.transforms.functional as F
from torchaug.batch_transforms._utils import \
    _assert_video_or_batch_videos_tensor
from torchaug.transforms._utils import _assert_module_or_list_of_modules
from torchaug.utils import VideoFormat, _log_api_usage_once

from ._transform import RandomTransform, VideoBase


class Div255(nn.Module):
    """Divide a tensor by 255.

    Args:
        inplace: Bool to make this operation in-place.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        _log_api_usage_once(self)

        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """Divide tensor by 255.

        Args:
            tensor: The tensor to divide.

        Returns:
            Divided tensor.
        """
        return F.div_255(tensor, inplace=self.inplace)

    def __repr__(self):
        return f"{__class__.__name__}(inplace={self.inplace})"


class MixUp(nn.Module):
    """Mix input tensor with linear interpolation drawn according a Beta law.

    The shape of the tensors is expected to be [B, ...] with ... any number of dimensions.
    The tensor should be float.

    .. note::
        The tensor is rolled according its first dimension and mixed with one
        drawn interpolation parameter for the whole tensor.

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

    def _get_params(self) -> float:
        """Draw the mixing coefficient.

        Returns:
            The mixing coefficient.
        """
        return float(self.mix_sampler.sample(()))

    def forward(
        self, tensor: Tensor, labels: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, float]:
        """Mix the input tensor and labels.

        Args:
            tensor: The tensor to mix.
            labels: If not None, the labels to mix.

        Returns:
            Tuple:
            - mixed tensor.
            - mixed labels or None.
            - mixing coefficient.
        """
        lam = self._get_params()

        tensor = tensor if self.inplace else tensor.clone()

        if labels is None:
            return F.mixup(tensor, tensor.roll(1, 0), lam, True), None, lam

        labels = labels if self.inplace else labels.clone()

        return (
            F.mixup(tensor, tensor.roll(1, 0), lam, True),
            F.mixup(labels, labels.roll(1, 0), lam, True),
            lam,
        )

    def __repr__(self):
        return f"{__class__.__name__}(alpha={self.alpha}, inplace={self.inplace})"


class Mul255(nn.Module):
    """Multiply a tensor by 255.

    Args:
        inplace: Bool to make this operation in-place.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        _log_api_usage_once(self)

        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """Multiply tensor by 255.

        Args:
            tensor: The tensor to multiply.

        Returns:
            Multiplied tensor.
        """
        return F.mul_255(tensor, inplace=self.inplace)

    def __repr__(self):
        return f"{__class__.__name__}(inplace={self.inplace})"


class Normalize(nn.Module):
    """Normalize a tensor image with mean and standard deviation. Given mean: ``(mean[1],...,mean[n])`` and
    std: ``(std[1],..,std[n])`` for ``n`` channels, this transform will normalize each channel of the input
    ``torch.Tensor`` i.e.,

    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.
        cast_dtype: If not None, scale and cast input to dtype. Expected to be a float dtype.
        inplace: Bool to make this operation in-place.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization.
    """

    def __init__(
        self,
        mean: Sequence[float] | float,
        std: Sequence[float] | float,
        cast_dtype: torch.dtype | None = None,
        inplace: bool = False,
        value_check: bool = False,
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)

        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)

        if mean.ndim in [0, 1]:
            mean = mean.view(-1, 1, 1)
        if std.ndim in [0, 1]:
            std = std.view(-1, 1, 1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.inplace = inplace
        self.value_check = value_check
        self.cast_dtype = cast_dtype

    def forward(self, tensor: Tensor) -> Tensor:
        """Normalize tensor.

        Args:
            tensor: The tensor to normalize.

        Returns:
            Normalized tensor.
        """
        return F.normalize(
            tensor,
            mean=self.mean,
            std=self.std,
            cast_dtype=self.cast_dtype,
            inplace=self.inplace,
            value_check=self.value_check,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mean={self.mean.tolist()},"
            f" std={self.std.tolist()},"
            f" cast_dtype={self.cast_dtype},"
            f" inplace={self.inplace},"
            f" value_check={self.value_check})"
        )


class RandomApply(RandomTransform):
    """Apply randomly a list of transformations with a given probability.

    Args:
        transforms: list of transformations
        p: probability
    """

    def __init__(
        self, transforms: Sequence[nn.Module] | nn.Module, p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        _log_api_usage_once(self)

        _assert_module_or_list_of_modules(transforms)

        if isinstance(transforms, nn.Module):
            transforms = [transforms]

        self.transforms = nn.ModuleList(transforms)

    def apply_transform(self, img: Tensor) -> Tensor:
        """
        Args:
            img: Image to transform.

        Returns:
            Transformed image.
        """
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        transforms_repr = str(self.transforms).replace("\n", "\n    ")
        return (
            f"{self.__class__.__name__}("
            f"\n    p={self.p},"
            f"\n    transforms={transforms_repr}"
            f"\n)"
        )


class RandomGaussianBlur(RandomTransform):
    """Blurs image with randomly chosen Gaussian blur.

    The image is expected to have the shape [..., C, H, W], where ...
    means an arbitrary number of leading dimensions.

    Args:
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation to be used for creating kernel to perform blurring.
            If float, sigma is fixed. If it is tuple of float (min, max), sigma
            is chosen uniformly at random to lie in the given range.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        sigma: float | tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
        value_check: bool = False,
    ):
        super().__init__(p=p)
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
    def get_params(sigma_min: Tensor, sigma_max: Tensor) -> Tensor:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min: Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max: Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        dtype = sigma_min.dtype
        device = sigma_min.device
        return (
            torch.rand([], dtype=dtype, device=device) * (sigma_max - sigma_min)
            + sigma_min
        )

    def apply_transform(self, img: Tensor) -> Tensor:
        """Blur the image.

        Args:
            img: Image to be blurred.

        Returns:
            Gaussian blurred image.
        """
        sigma: Tensor = self.get_params(self.sigma[0], self.sigma[1])
        return F.gaussian_blur(img, self.kernel_size, [sigma, sigma], self.value_check)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma.tolist()}, p={self.p}, value_check={self.value_check})"
        return s


class VideoNormalize(Normalize, VideoBase):
    """Normalize a tensor video with mean and standard deviation. Given mean: ``(mean[1],...,mean[n])`` and std:
    ``(std[1],..,std[n])`` for ``n`` channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,

    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Videos should be in format [..., T, C, H, W] or [..., C, T, H, W] with ... 0 or 1 leading dimension.

    Args:
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.
        video_format: Dimension order of the video. Can be ``TCHW`` or ``CTHW``.
        cast_dtype: If not None, scale and cast input to the dtype. Expected to be a float dtype.
        inplace: Bool to make this operation in-place.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization.
    """

    def __init__(
        self,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        cast_dtype: torch.dtype | None = None,
        inplace: bool = False,
        value_check: bool = False,
        video_format: VideoFormat = VideoFormat.CTHW,
    ) -> None:
        Normalize.__init__(
            self,
            mean=mean,
            std=std,
            cast_dtype=cast_dtype,
            inplace=inplace,
            value_check=value_check,
        )
        VideoBase.__init__(self, video_format=video_format)

        _log_api_usage_once(self)

    def forward(self, video: Tensor) -> Tensor:
        """Normalize a video.

        Args:
            video: The video to normalize.

        Returns:
            Normalized video.
        """
        _assert_video_or_batch_videos_tensor(video)

        if not self.time_before_channel:
            dims = [0, 2, 1, 3, 4] if video.ndim == 5 else [1, 0, 2, 3]
            video = video.permute(dims)

        video = Normalize.forward(self, video)

        if not self.time_before_channel:
            video = video.permute(dims)

        return video

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mean={self.mean.tolist()},"
            f" std={self.std.tolist()},"
            f" cast_dtype={self.cast_dtype},"
            f" inplace={self.inplace},"
            f" value_check={self.value_check},"
            f" video_format={self.video_format.value})"
        )
