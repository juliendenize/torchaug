from __future__ import annotations

import numbers
from typing import Sequence

import torch
import torchvision.transforms as tv_transforms
from torch import Tensor, nn
from torchvision.transforms.transforms import _setup_size

from torchaug.batch_transforms._utils import \
    _assert_video_or_batch_videos_tensor
from torchaug.transforms._utils import _assert_video_tensor
from torchaug.transforms.functional import gaussian_blur, normalize, solarize
from torchaug.utils import _log_api_usage_once


class Normalize(nn.Module):
    def __init__(
        self,
        mean: Sequence[float] | float,
        std: Sequence[float] | float,
        cast_dtype: torch.dtype | None = None,
        inplace: bool = False,
        value_check: bool = False,
    ) -> None:
        """Normalize a tensor image with mean and standard deviation. Given mean: ``(mean[1],...,mean[n])`` and
        std: ``(std[1],..,std[n])`` for ``n`` channels, this transform will normalize each channel of the input
        ``torch.Tensor`` i.e.,

        ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            cast_dtype (dtype, optional): If not None, scale and cast input to dtype. Expected to be a float dtype.
                Default, None.
            inplace (bool): Bool to make this operation in-place.
            value_check (bool, optional): Bool to perform tensor value check.
                Might cause slow down on some devices because of synchronization. Default, False.
        """
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
        return normalize(
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


class RandomApply(tv_transforms.RandomApply):
    """Apply randomly a list of transformations with a given probability.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(
        self, transforms: Sequence[nn.Module] | nn.Module, p: float = 0.5
    ) -> None:
        if not isinstance(transforms, nn.Module) and not isinstance(
            transforms, nn.ModuleList
        ):
            transforms = nn.ModuleList(transforms)
        elif not isinstance(transforms, nn.ModuleList):
            transforms = nn.ModuleList([transforms])

        super().__init__(transforms, p)
        _log_api_usage_once(self)

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): Images to transform.

        Returns:
            Tensor: Randomly transformed image.
        """
        if self.p < torch.rand(1):
            return img

        for t in self.transforms:
            img = t(img)
        return img


class RandomColorJitter(tv_transforms.ColorJitter):
    def __init__(
        self,
        brightness: float | tuple[float, float] | None = 0,
        contrast: float | tuple[float, float] | None = 0,
        saturation: float | tuple[float, float] | None = 0,
        hue: float | tuple[float, float] | None = 0,
        p: float = 0.0,
    ):
        """Randomly change the brightness, contrast, saturation and hue to images. The images is expected to have.

        [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

        Args:
            brightness (float or tuple of float (min, max)): How much to jitter brightness.
                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                or the given [min, max]. Should be non negative numbers.
            contrast (float or tuple of float (min, max)): How much to jitter contrast.
                contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                or the given [min, max]. Should be non-negative numbers.
            saturation (float or tuple of float (min, max)): How much to jitter saturation.
                saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                or the given [min, max]. Should be non negative numbers.
            hue (float or tuple of float (min, max)): How much to jitter hue.
                hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
                Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
                To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
                thus it does not work if you normalize your image to an interval with negative values,
                or use an interpolation that generates negative values before using this function.
            p (float): Probability to apply color jitter.
        """
        super().__init__(brightness, contrast, saturation, hue)
        _log_api_usage_once(self)

        self.p = p

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): Image to be jittered.

        Returns:
            Tensor: Randomly jittered image.
        """

        if self.p < torch.rand(1):
            return img
        return super().forward(img)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
            f", p={self.p})"
        )
        return s


class RandomGaussianBlur(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.

    The image is expected to have the shape [..., C, H, W], where ... means an arbitrary number of leading dimensions.
    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
        value_check (bool, optional): Bool to perform tensor value check.
             Might cause slow down on some devices because of synchronization. Default, False.
    Returns:
        Tensor: Gaussian blurred version of the input image.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        sigma: float | tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
        value_check: bool = False,
    ):
        super().__init__()
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
        self.p = p
        self.value_check = value_check

    @staticmethod
    def get_params(sigma_min: Tensor, sigma_max: Tensor) -> Tensor:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.
        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        dtype = sigma_min.dtype
        device = sigma_min.device
        return (
            torch.rand([], dtype=dtype, device=device) * (sigma_max - sigma_min)
            + sigma_min
        )

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): Image to be blurred.
        Returns:
            Tensor: Randomly gaussian blurred image.
        """

        if self.p < torch.rand(1):
            return img
        sigma: Tensor = self.get_params(self.sigma[0], self.sigma[1])
        return gaussian_blur(img, self.kernel_size, [sigma, sigma], self.value_check)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma.tolist()}, p={self.p}, value_check={self.value_check})"
        return s


class RandomSolarize(nn.Module):
    def __init__(
        self,
        threshold: float,
        p: float = 0.5,
        value_check: bool = False,
    ):
        """Solarize the image randomly with a given probability by inverting all pixel values above a threshold.
        The img is expected to be in [..., 1 or 3, H, W] format, where ... means it can have an arbitrary number of
        leading dimensions.

        Args:
            threshold (float): all pixels equal or above this value are inverted.
            p (float): probability of the image being solarized. Default value is 0.5.
            value_check (bool, optional): Bool to perform tensor value check.
                Might cause slow down on some devices because of synchronization. Default, False.
        """
        super().__init__()
        _log_api_usage_once(self)

        self.register_buffer("threshold", torch.as_tensor(threshold))
        self.p = p
        self.value_check = value_check

    def forward(self, img: Tensor):
        """
        Args:
            img (Tensor): Image to be solarized.
        Returns:
            Tensor: Randomly solarized image.
        """

        if torch.rand(1).item() < self.p:
            return solarize(img, self.threshold, self.value_check)
        return img

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(threshold={self.threshold.item()}"
            f", p={self.p}"
            f", value_check={self.value_check})"
        )


class VideoNormalize(Normalize):
    """Normalize a tensor video with mean and standard deviation. Given mean: ``(mean[1],...,mean[n])`` and std:
    ``(std[1],..,std[n])`` for ``n`` channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,

    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Videos should be in format [..., T, C, H, W] or [..., C, T, H, W] with ... 0 or 1 leading dimension.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        video_format(str): Dimension order of the video. Can be ``TCHW`` or ``CTHW``.
        cast_dtype (dtype, optional): If not None, scale and cast input to the dtype. Expected to be a float dtype.
            Default, None.
        inplace (bool): Bool to make this operation in-place.
        value_check (bool, optional): Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization. Default, False.
    """

    def __init__(
        self,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        cast_dtype: torch.dtype | None = None,
        inplace: bool = False,
        value_check: bool = False,
        video_format: str = "CTHW",
    ) -> None:
        super().__init__(
            mean=mean,
            std=std,
            cast_dtype=cast_dtype,
            inplace=inplace,
            value_check=value_check,
        )
        _log_api_usage_once(self)

        self.video_format = video_format

        if self.video_format == "CTHW":
            self.time_before_channel = False
        elif self.video_format == "TCHW":
            self.time_before_channel = True
        else:
            raise ValueError(
                f"video_format should be either 'CTHW' or 'TCHW'. Got {self.video_format}."
            )

    def forward(self, video: Tensor):
        _assert_video_or_batch_videos_tensor(video)

        if not self.time_before_channel:
            dims = [0, 2, 1, 3, 4] if video.ndim == 5 else [1, 0, 2, 3]
            video = video.permute(dims)

        video = super().forward(video)

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
            f" video_format={self.video_format})"
        )


class VideoWrapper(nn.Module):
    """Wrap a transform to handle video data. If the frames should be augmented differently, the transform must
    handle the leading dimension differently. The video is expected to be in format [C, T, H, W] or [T, C, H, W].

    Args:
        transform (nn.Module): The transform to wrap.
        video_format (str, optional): Format of the video. Either ``CTHW`` or ``TCHW``. Defaults to "CTHW".
    """

    def __init__(self, transform: nn.Module, video_format: str = "CTHW") -> None:
        super().__init__()
        _log_api_usage_once(self)

        self.transform = transform
        self.video_format = video_format

        if self.video_format == "CTHW":
            self.time_before_channel = False
        elif self.video_format == "TCHW":
            self.time_before_channel = True
        else:
            raise ValueError(
                f"video_format should be either 'CTHW' or 'TCHW'. Got {self.video_format}."
            )

    def forward(self, video: Tensor):
        _assert_video_tensor(video)

        if not self.time_before_channel:
            video = video.permute(1, 0, 2, 3)

        video = self.transform(video)

        if not self.time_before_channel:
            video = video.permute(1, 0, 2, 3)

        return video

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  transform={self.transform},\n"
            f"  video_format={self.video_format})"
        )
