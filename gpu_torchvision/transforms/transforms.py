import numbers
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torchvision import transforms
from torchvision.transforms.transforms import _setup_size

from gpu_torchvision.transforms.functional import (gaussian_blur, normalize,
                                                   solarize)


class Normalize(transforms.Normalize):
    def __init__(
        self,
        mean: Sequence[float] | float,
        std: Sequence[float] | float,
        inplace: bool = False,
    ) -> None:
        """Normalize a tensor image with mean and standard deviation. This transform does not support PIL Image.
        Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n`` channels, this transform
        will normalize each channel of the input ``torch.*Tensor`` i.e.,

        ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

        .. note::
            This transform acts out of place, i.e., it does not mutate the input tensor.

        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace (bool): Bool to make this operation in-place.
        """

        super().__init__(mean=mean, std=std)

        del self.mean
        del self.std

        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)

        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        return normalize(tensor, self.mean, self.std, inplace=self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean.tolist()}, std={self.std.tolist()}, inplace={self.inplace})"


class RandomApply(transforms.RandomApply):
    """Apply randomly a list of transformations with a given probability.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(
        self, transforms: Sequence[nn.Module] | nn.Module, p: float = 0.5
    ) -> None:
        super().__init__(transforms, p)

        if not issubclass(type(self.transforms), nn.Module) and not issubclass(
            type(self.transforms), nn.ModuleList
        ):
            self.transforms = nn.ModuleList(self.transforms)
        else:
            self.transforms = nn.ModuleList([self.transforms])

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): Images to transform.

        Returns:
            Tensor: Randomly transformed image.
        """

        if self.p < torch.rand(1):
            return img
        img = self.transforms(img)
        return img


class RandomColorJitter(transforms.ColorJitter):
    def __init__(
        self,
        brightness: float | Tuple[float, float] | None = 0,
        contrast: float | Tuple[float, float] | None = 0,
        saturation: float | Tuple[float, float] | None = 0,
        hue: float | Tuple[float, float] | None = 0,
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
    Returns:
        Tensor: Gaussian blurred version of the input image.
    """

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        sigma: float | Tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
    ):
        super().__init__()
        self.kernel_size = _setup_size(
            kernel_size, "Kernel size should be a tuple/list of two integers"
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
        return gaussian_blur(img, self.kernel_size, [sigma, sigma])

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma.tolist()}, p={self.p})"
        return s


class RandomSolarize(transforms.RandomSolarize):
    def __init__(
        self,
        threshold: float,
        p: float = 0.5,
    ):
        """Solarize the image randomly with a given probability by inverting all pixel values above a threshold.
        The img is expected to be in [..., 1 or 3, H, W] format, where ... means it can have an arbitrary number of
        leading dimensions.

        Args:
            threshold (float): all pixels equal or above this value are inverted.
            p (float): probability of the image being solarized. Default value is 0.5.
        """
        super().__init__(threshold=threshold, p=p)
        del self.threshold

        self.register_buffer("threshold", torch.as_tensor(threshold))

    def forward(self, img: Tensor):
        """
        Args:
            img (Tensor): Image to be solarized.
        Returns:
            Tensor: Randomly solarized image.
        """

        if torch.rand(1).item() < self.p:
            return solarize(img, self.threshold)
        return img

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(threshold={self.threshold.item()}, p={self.p})"
        )
