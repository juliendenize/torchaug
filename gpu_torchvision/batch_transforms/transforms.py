import numbers
from math import ceil
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torchvision import transforms
from torchvision.transforms.functional import (InterpolationMode, hflip,
                                               resize, resized_crop)
from torchvision.transforms.transforms import _setup_size

from gpu_torchvision.batch_transforms.functional import (
    batch_adjust_brightness, batch_adjust_contrast, batch_adjust_hue,
    batch_adjust_saturation, batch_gaussian_blur)
from gpu_torchvision.transforms.functional import solarize
from gpu_torchvision.transforms.transforms import RandomApply, RandomSolarize


class BatchRandomApply(RandomApply):
    def __init__(
        self, transforms: Sequence[nn.Module] | nn.Module, p: float = 0.5
    ) -> None:
        super().__init__(transforms, p)

    def forward(self, imgs: Tensor) -> Tensor:
        batch_size = imgs.shape[0]
        num_apply = round(self.p * batch_size)

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)[:num_apply]
        imgs[indices_do_apply] = self.transforms(imgs[indices_do_apply])

        return imgs


class BatchRandomColorJitter(transforms.ColorJitter):
    def __init__(
        self,
        brightness: float | Tuple[float, float] | None = 0,
        contrast: float | Tuple[float, float] | None = 0,
        saturation: float | Tuple[float, float] | None = 0,
        hue: float | Tuple[float, float] | None = 0,
        p: float = 0.0,
        num_rand_calls: int = 4,
    ) -> None:
        super().__init__(brightness, contrast, saturation, hue)
        self.p = p
        self.num_rand_calls = num_rand_calls

        brightness = self.brightness
        contrast = self.contrast
        saturation = self.saturation
        hue = self.hue

        if brightness is not None:
            del self.brightness
            self.register_buffer("brightness", torch.as_tensor(brightness))

        if contrast is not None:
            del self.contrast
            self.register_buffer("contrast", torch.as_tensor(contrast))

        if saturation is not None:
            del self.saturation
            self.register_buffer("saturation", torch.as_tensor(saturation))

        if hue is not None:
            del self.hue
            self.register_buffer("hue", torch.as_tensor(hue))

    @staticmethod
    def get_params(
        brightness: Tensor | None,
        contrast: Tensor | None,
        saturation: Tensor | None,
        hue: Tensor | None,
        batch_size: int,
    ) -> Tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = (
            None
            if brightness is None
            else torch.empty((batch_size,), device=brightness.device).uniform_(
                brightness[0], brightness[1]
            )
        )
        c = (
            None
            if contrast is None
            else torch.empty((batch_size,), device=contrast.device).uniform_(
                contrast[0], contrast[1]
            )
        )
        s = (
            None
            if saturation is None
            else torch.empty((batch_size,), device=saturation.device).uniform_(
                saturation[0], saturation[1]
            )
        )
        h = (
            None
            if hue is None
            else torch.empty((batch_size,), device=hue.device).uniform_(hue[0], hue[1])
        )

        return fn_idx, b, c, s, h

    def forward(self, imgs: Tensor) -> Tensor:
        batch_size = imgs.shape[0]
        num_apply = round(self.p * batch_size)

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)[:num_apply]

        num_combination = min(num_apply, self.num_rand_calls)
        num_apply_per_combination = ceil(num_apply / num_combination)

        # Avoid inplace operation
        output = imgs.clone()

        for i in range(min(num_apply, self.num_rand_calls)):

            indices_combination = indices_do_apply[
                i * num_apply_per_combination : (i + 1) * num_apply_per_combination
            ]

            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = self.get_params(
                self.brightness,
                self.contrast,
                self.saturation,
                self.hue,
                indices_combination.shape[0],
            )

            imgs_combination = imgs[indices_combination]

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    imgs_combination[:] = batch_adjust_brightness(
                        imgs_combination, brightness_factor
                    )
                elif fn_id == 1 and contrast_factor is not None:
                    imgs_combination[:] = batch_adjust_contrast(
                        imgs_combination, contrast_factor
                    )
                elif fn_id == 2 and saturation_factor is not None:
                    imgs_combination[:] = batch_adjust_saturation(
                        imgs_combination, saturation_factor
                    )
                elif fn_id == 3 and hue_factor is not None:
                    imgs_combination[:] = batch_adjust_hue(imgs_combination, hue_factor)

            output[indices_combination] = imgs_combination

        return output

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness.tolist() if self.brightness is not None else None}"
            f", contrast={self.contrast.tolist() if self.contrast is not None else None}"
            f", saturation={self.saturation.tolist() if self.saturation is not None else None}"
            f", hue={self.hue.tolist() if self.hue is not None else None})"
            f", p={self.p})"
        )
        return s


class BatchRandomGaussianBlur(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.

    If the image is torch Tensor, it is expected
    to have [B, ..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.
    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.
    """

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        sigma: int | Tuple[int, int] = (0.1, 2.0),
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
    def get_params(sigma_min: Tensor, sigma_max: Tensor, batch_size: int) -> Tensor:
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
            torch.rand(batch_size, 1, dtype=dtype, device=device).expand(batch_size, 2)
            * (sigma_max - sigma_min)
            + sigma_min
        )

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.
        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        batch_size = imgs.shape[0]
        num_apply = round(self.p * batch_size)

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)[:num_apply]

        sigma: Tensor = self.get_params(self.sigma[0], self.sigma[1], num_apply)

        imgs[indices_do_apply] = batch_gaussian_blur(
            imgs[indices_do_apply], self.kernel_size, sigma
        )

        return imgs

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma.tolist()}, p={self.p})"
        return s


class BatchRandomGrayScale(transforms.Grayscale):
    def __init__(
        self,
        num_output_channels: int = 3,
        p: float = 0.5,
    ):
        super().__init__(num_output_channels)
        self.p = p

    def forward(self, imgs: Tensor) -> Tensor:
        batch_size = imgs.shape[0]
        num_apply = round(self.p * batch_size)

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)[:num_apply]
        imgs[indices_do_apply] = super().forward(imgs[indices_do_apply])

        return imgs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels}, p={self.p})"


class BatchRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(
        self,
        p: float = 0.5,
    ):
        super().__init__(p=p)

    def forward(self, imgs: Tensor) -> Tensor:
        batch_size = imgs.shape[0]
        num_apply = round(self.p * batch_size)

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)[:num_apply]
        imgs[indices_do_apply] = hflip(imgs[indices_do_apply])

        return imgs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class BatchRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        size: int | Sequence[int],
        scale: Sequence[float] = (0.08, 1.0),
        ratio: Sequence[float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        num_rand_calls: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(size, scale, ratio, interpolation, antialias, **kwargs)

        if isinstance(size, int):
            self.size = [size, size]
        elif isinstance(size, Sequence) and len(size) == 1:
            self.size = [size[0], size[0]]

        if not num_rand_calls >= 0:
            raise ValueError(
                f"num_rand_calls attribute should be superior to 0, {num_rand_calls} given."
            )

        self.num_rand_calls = num_rand_calls

    def single_forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): Image to be cropped and resized.

        Returns:
            Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return resized_crop(
            img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias
        )

    def forward(self, imgs: Tensor) -> Tensor:
        if self.num_rand_calls == 0:
            return torch.stack([self.single_forward(img) for img in imgs])
        else:
            # Avoid inplace operation
            output = torch.empty(
                size=(*imgs.shape[:-2], *self.size),
                dtype=imgs.dtype,
                device=imgs.device,
            )

            batch_size = imgs.shape[0]
            num_combination = min(batch_size, self.num_rand_calls)
            num_apply_per_combination = ceil(batch_size / num_combination)

            indices_do_apply = torch.randperm(batch_size, device=imgs.device)

            for i in range(min(batch_size, self.num_rand_calls)):
                indices_combination = indices_do_apply[
                    i * num_apply_per_combination : (i + 1) * num_apply_per_combination
                ]

                imgs_combination = imgs[indices_combination]
                imgs_combination = self.single_forward(imgs_combination)

            output[indices_combination] = imgs_combination

        return output


class BatchRandomSolarize(RandomSolarize):
    def __init__(self, threshold: float, p: float = 0.5) -> None:
        super().__init__(threshold, p)

    def forward(self, imgs: Tensor) -> Tensor:
        batch_size = imgs.shape[0]
        num_apply = round(self.p * batch_size)

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)[:num_apply]
        imgs[indices_do_apply] = solarize(imgs[indices_do_apply], self.threshold)

        return imgs
