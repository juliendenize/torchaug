from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torchvision.transforms.functional as F_tv
from torch import Tensor, nn
from torchvision.transforms.transforms import _setup_size

import torchaug.transforms.functional as F
from torchaug.batch_transforms._utils import \
    _assert_video_or_batch_videos_tensor
from torchaug.transforms._utils import _assert_video_tensor, _check_input
from torchaug.utils import _log_api_usage_once


class RandomTransform(nn.Module, ABC):
    """Abstract class to make a base class for all random transforms.

    args:
        p: Probability to apply the transform.
    """

    def __init__(self, p: float) -> None:
        super().__init__()

        if not 0 <= p <= 1:
            raise ValueError(
                f"p should be superior to 0 (included) and inferior to 1 (included). Got {p}."
            )

        self.p = p

    @abstractmethod
    def apply_transform(self, img: Tensor):
        """Function to perform transformation on the image.

        .. note::
            Should be overriden by subclasses.

        Args:
            img: Image to transform.

        Return:
            The transformed image.
        """
        ...

    def forward(self, img: Tensor):
        """Call :meth:`apply_transform` based on random sample.

        Args:
            img: Image to randomly transform based on :attr:`~p`.

        Return:
            The randomly transformed image.
        """
        if self.p == 0.0:
            return img
        elif self.p == 1.0 or torch.rand(1).item() < self.p:
            return self.apply_transform(img)
        return img


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
    The tensor shoud be float.

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

        if not isinstance(transforms, nn.Module) and not isinstance(
            transforms, nn.ModuleList
        ):
            transforms = nn.ModuleList(transforms)
        elif not isinstance(transforms, nn.ModuleList):
            transforms = nn.ModuleList([transforms])

        self.transforms = transforms

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
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    p={self.p},"
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class RandomColorJitter(RandomTransform):
    """Randomly change the brightness, contrast, saturation and hue to images.

    The images is expected to have [..., 1 or 3, H, W] shape, where ...
    means an arbitrary number of leading dimensions.

    Args:
        brightness: How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast: How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non-negative numbers.
        saturation: How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue: How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
        p: Probability to apply color jitter.
    """

    def __init__(
        self,
        brightness: float | tuple[float, float] = 0,
        contrast: float | tuple[float, float] = 0,
        saturation: float | tuple[float, float] = 0,
        hue: float | tuple[float, float] = 0,
        p: float = 0.0,
    ):
        super().__init__(p=p)
        _log_api_usage_once(self)

        self.p = p
        self.brightness = _check_input(brightness, "brightness")
        self.contrast = _check_input(contrast, "contrast")
        self.saturation = _check_input(saturation, "saturation")
        self.hue = _check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )

    @staticmethod
    def get_params(
        brightness: list[float] | None,
        contrast: list[float] | None,
        saturation: list[float] | None,
        hue: list[float] | None,
    ) -> tuple[Tensor, float | None, float | None, float | None, float | None]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness: The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast: The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation: The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue: The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            The parameters used to apply the randomized transform along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = (
            None
            if brightness is None
            else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        )
        c = (
            None
            if contrast is None
            else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        )
        s = (
            None
            if saturation is None
            else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        )
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def apply_transform(self, img: Tensor) -> Tensor:
        """Color jitter the image.

        Args:
            img: Input image.

        Returns:
            Color jittered image.
        """
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F_tv.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F_tv.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F_tv.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F_tv.adjust_hue(img, hue_factor)

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue}"
            f", p={self.p})"
        )
        return s


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


class RandomSolarize(RandomTransform):
    """Solarize the image randomly with a given probability by inverting all pixel values above a threshold.

    The image is expected to be in [..., 1 or 3, H, W] format, where ... means it can have an arbitrary number of
    leading dimensions.

    Args:
        threshold: all pixels equal or above this value are inverted.
        p: probability of the image being solarized.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization.
    """

    def __init__(
        self,
        threshold: float,
        p: float = 0.5,
        value_check: bool = False,
    ):
        super().__init__(p=p)
        _log_api_usage_once(self)

        self.register_buffer("threshold", torch.as_tensor(threshold))
        self.value_check = value_check

    def apply_transform(self, img: Tensor):
        """Solarize the image.

        Args:
            img: Image to be solarized.

        Returns:
            Solarized image.
        """
        return F.solarize(img, self.threshold, self.value_check)

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
        transform: The transform to wrap.
        video_format: Format of the video. Either ``CTHW`` or ``TCHW``.
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

    def forward(self, video: Tensor) -> Tensor:
        """Apply :attr:`~transform` on the video.

        Args:
            video: The video to transform.

        Returns:
            The transformed video.
        """
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
            f"    transform={self.transform},\n"
            f"    video_format={self.video_format}\n)"
        )
