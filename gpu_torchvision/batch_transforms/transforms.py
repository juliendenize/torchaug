import numbers
from math import ceil
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torchvision import transforms
from torchvision.transforms.functional import (InterpolationMode, hflip,
                                               resized_crop)
from torchvision.transforms.transforms import _setup_size

from gpu_torchvision.batch_transforms.functional import (
    batch_adjust_brightness, batch_adjust_contrast, batch_adjust_hue,
    batch_adjust_saturation, batch_gaussian_blur)
from gpu_torchvision.transforms.functional import solarize
from gpu_torchvision.transforms.transforms import RandomApply, RandomSolarize


class BatchRandomApply(RandomApply):
    """Apply randomly a list of transformations to a batch of images with a given probability.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
        inplace (bool): If True, perform inplace operation to save memory.
    """

    def __init__(
        self,
        transforms: Sequence[nn.Module] | nn.Module,
        p: float = 0.5,
        inplace: bool = True,
    ) -> None:
        super().__init__(transforms, p)
        self.inplace = inplace

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Args:
            imgs (Tensor): Batch of images to transform.

        Returns:
            Tensor: Randomly transformed batch of images.
        """
        batch_size = imgs.shape[0]
        num_apply = round(self.p * batch_size)

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)[:num_apply]
        output = imgs if self.inplace else imgs.clone()

        for t in self.transforms:
            output[indices_do_apply] = t(output[indices_do_apply])

        return output

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    p={self.p}"
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n"
        format_string += f"    inplace={self.inplace}"
        format_string += "\n)"
        return format_string


class BatchRandomColorJitter(transforms.ColorJitter):
    """Randomly change the brightness, contrast, saturation and hue to a batch of images. The batch is expected to
    have [B, ..., 1 or 3, H, W] shape, where ... means an arbitrary number of dimensions.

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
        num_rand_calls (int): Number of random calls performed to apply augmentations at different orders on sub-batches.
            If -1, B calls are performed. The maximum is 24 = 4!.
        inplace (bool): If True, perform inplace operation to save memory.
    """

    def __init__(
        self,
        brightness: float | Tuple[float, float] | None = 0,
        contrast: float | Tuple[float, float] | None = 0,
        saturation: float | Tuple[float, float] | None = 0,
        hue: float | Tuple[float, float] | None = 0,
        p: float = 0.5,
        num_rand_calls: int = 1,
        inplace: bool = True,
    ) -> None:
        super().__init__(brightness, contrast, saturation, hue)
        self.p = p

        if not num_rand_calls >= -1:
            raise ValueError(
                f"num_rand_calls attribute should be superior to -1, {num_rand_calls} given."
            )

        self.num_rand_calls = num_rand_calls
        self.inplace = inplace

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
            brightness (tensor (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tensor (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tensor (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tensor (min, max), optional): The range from which the hue_factor is chosen uniformly.
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
        """
        Args:
            imgs (Tensor): Input batch of images.

        Returns:
            Tensor: Randomly color jittered batch of images.
        """
        if self.p == 0.0 or self.num_rand_calls == 0:
            return imgs

        batch_size = imgs.shape[0]
        num_apply = round(self.p * batch_size)

        if self.num_rand_calls == -1:
            num_combination = num_apply
        else:
            num_combination = min(num_apply, self.num_rand_calls)

        # At most 24 (= 4!) different combinations possible.
        num_combination = min(num_combination, 24)

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)[:num_apply]
        num_apply_per_combination = ceil(num_apply / num_combination)

        output = imgs if self.inplace else imgs.clone()

        for i in range(num_combination):

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

            imgs_combination = output[indices_combination]

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
            f", hue={self.hue.tolist() if self.hue is not None else None}"
            f", p={self.p}"
            f", inplace={self.inplace})"
        )
        return s


class BatchRandomGaussianBlur(torch.nn.Module):
    """Blurs batch of images with randomly chosen Gaussian blur.

    The batch of images is expected to be of shape [B, ..., C, H, W] where ... means an arbitrary number of dimensions.
    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (tensor (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
        p (float): Probability to apply gaussian blur.
        inplace (bool): If True, perform inplace operation to save memory.
    Returns:
        Tensor: Gaussian blurred version of the input batch of images.
    """

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        sigma: float | Tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
        inplace: bool = True,
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
        self.inplace = inplace

    @staticmethod
    def get_params(sigma_min: Tensor, sigma_max: Tensor, batch_size: int) -> Tensor:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (tensor): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (tensor): Maximum standard deviation that can be chosen for blurring kernel.
        Returns:
            tensor: Standard deviation to be passed to calculate kernel for gaussian blurring.
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
            imgs (Tensor): Batch of images to be blurred.
        Returns:
            Tensor: Randomly gaussian blurred images
        """

        batch_size = imgs.shape[0]
        num_apply = round(self.p * batch_size)

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)[:num_apply]

        sigma: Tensor = self.get_params(self.sigma[0], self.sigma[1], num_apply)

        output: Tensor = imgs.clone() if not self.inplace else imgs
        output[indices_do_apply] = batch_gaussian_blur(
            output[indices_do_apply], self.kernel_size, sigma
        )

        return output

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"kernel_size={self.kernel_size}"
            f", sigma={self.sigma.tolist()}"
            f", p={self.p}"
            f", inplace={self.inplace})"
        )
        return s


class BatchRandomGrayScale(transforms.Grayscale):
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = True,
    ):
        """Convert batch of images to grayscale. The batch of images is expected to be of shape [B, ..., C, H, W]
        where ... means an arbitrary number of dimensions.

        Args:
            p: Probability of the images to be grayscaled.
            inplace (bool): If True, perform inplace operation to save memory.

        Returns:
            Tensor: Grayscale of the batch of images.
        """

        super().__init__(num_output_channels=3)
        self.p = p
        self.inplace = inplace

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Args:
            imgs (Tensor): Batch of images to be grayscaled.

        Returns:
            Tensor: Randomly grayscaled batch of images.
        """

        batch_size = imgs.shape[0]
        num_apply = round(self.p * batch_size)

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)[:num_apply]

        output: Tensor = imgs if self.inplace else imgs.clone()
        output[indices_do_apply] = super().forward(output[indices_do_apply])

        return output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f" num_output_channels={self.num_output_channels}"
            f", p={self.p}"
            f", inplace={self.inplace})"
        )


class BatchRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = True,
    ):
        """Horizontally flip the given batch of images randomly with a given probability. The batch of images is
        expected to be of shape [B, ..., C, H, W] where ... means an arbitrary number of dimensions.

        Args:
            p (float): probability of the images being flipped.
            inplace (bool): If True, perform inplace operation to save memory.

        Returns:
            Tensor: Grayscale batch of images.
        """

        super().__init__(p=p)
        self.inplace = inplace

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Args:
            imgs (Tensor): Batch of images to be horizontally fliped.

        Returns:
            Tensor: Randomly horizontally fliped batch of images.
        """

        batch_size = imgs.shape[0]
        num_apply = round(self.p * batch_size)

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)[:num_apply]

        output: Tensor = imgs if self.inplace else imgs.clone()
        output[indices_do_apply] = hflip(output[indices_do_apply])

        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, inplace={self.inplace})"


class BatchRandomResizedCrop(transforms.RandomResizedCrop):
    """Crop a random portion of a batch of images and resize it to a given size.

    The batch shape is expected to be [B, ..., H, W], where ... means an arbitrary number of dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True``: will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The current default is ``None`` **but will change to** ``True`` **in
            v0.17** for the PIL and Tensor backends to be consistent.
        num_rand_calls (int): Number of random calls performed to apply augmentations at different orders on sub-batches.
            If -1, B calls are performed. The maximum is 24 = 4!.
    """

    def __init__(
        self,
        size: int | Sequence[int],
        scale: Sequence[float] = (0.08, 1.0),
        ratio: Sequence[float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        num_rand_calls: int = -1,
        **kwargs,
    ) -> None:
        super().__init__(size, scale, ratio, interpolation, antialias, **kwargs)

        if isinstance(size, int):
            self.size = [size, size]
        elif isinstance(size, Sequence) and len(size) == 1:
            self.size = [size[0], size[0]]

        if not num_rand_calls >= -1:
            raise ValueError(
                f"num_rand_calls attribute should be superior to -1, {num_rand_calls} given."
            )

        self.num_rand_calls = num_rand_calls

    def single_forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): Batch of images to be cropped and resized.

        Returns:
            Tensor: Cropped and resized batch of images.
        """

        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return resized_crop(
            img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias
        )

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Args:
            imgs (Tensor): Batch of images to be cropped and resized.

        Returns:
            Tensor: Randomly cropped and resized batch of images.
        """

        if self.num_rand_calls == -1:
            return torch.stack([self.single_forward(img) for img in imgs])
        elif self.num_rand_calls == 0:
            return imgs
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

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", antialias={self.antialias}"
        format_string += f", num_rand_calls={self.num_rand_calls})"
        return format_string


class BatchRandomSolarize(RandomSolarize):
    def __init__(self, threshold: float, p: float = 0.5, inplace: bool = True) -> None:
        """Solarize the image randomly with a given probability by inverting all pixel values above a threshold.
        The img it is expected to be in [..., 1 or 3, H, W] format, where ... means it can have an arbitrary number
        of leading dimensions.

        Args:
            threshold (float): all pixels equal or above this value are inverted.
            p (float): probability of the image being solarized. Default value is 0.5.
            inplace (bool): If True, perform inplace operation to save memory.
        """

        super().__init__(threshold, p)
        self.inplace = inplace

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Args:
            imgs (Tensor): Batch of images to be solarized.

        Returns:
            Tensor: Randomly solarized batch of images.
        """

        batch_size = imgs.shape[0]
        num_apply = round(self.p * batch_size)

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)[:num_apply]

        output: Tensor = imgs if self.inplace else imgs.clone()
        output[indices_do_apply] = solarize(output[indices_do_apply], self.threshold)

        return output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold.item()}"
            f", p={self.p}"
            f", inplace={self.inplace})"
        )
