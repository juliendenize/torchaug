from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from itertools import permutations
from math import ceil, floor
from typing import Sequence

import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as F_tv
from torch import Tensor, nn
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import (_interpolation_modes_from_int,
                                               _setup_size)

import torchaug.batch_transforms.functional as F_b
import torchaug.transforms.functional as F
from torchaug.batch_transforms._utils import _assert_batch_videos_tensor
from torchaug.transforms._utils import _check_input, transfer_tensor_on_device
from torchaug.utils import _log_api_usage_once


class BatchRandomTransform(nn.Module, ABC):
    """Abstract class to make a base class for all batch random transforms.

    args:
        p: Probability to apply the transform.
        inplace: If True, perform inplace operation to save memory.
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
            Should be overriden by subclasses.

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

        if num_apply > 0:
            output[indices_do_apply] = self.apply_transform(output[indices_do_apply])

        return output


class BatchRandomApply(BatchRandomTransform):
    """Apply randomly a list of transformations to a batch of images with a given probability.

    Args:
        transforms: List of transformations.
        p: Probability to apply the transform.
        inplace: If ``True``, perform inplace operation to save memory.
    """

    def __init__(
        self,
        transforms: Sequence[nn.Module] | nn.Module,
        p: float = 0.5,
        inplace: bool = False,
    ) -> None:
        super().__init__(p=p, inplace=inplace)
        _log_api_usage_once(self)

        if not isinstance(transforms, nn.Module) and not isinstance(
            transforms, nn.ModuleList
        ):
            transforms = nn.ModuleList(transforms)
        elif not isinstance(transforms, nn.ModuleList):
            transforms = nn.ModuleList([transforms])

        self.transforms = transforms
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
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    p={self.p},"
        format_string += f" inplace={self.inplace},"
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class BatchRandomColorJitter(BatchRandomTransform):
    """Randomly change the brightness, contrast, saturation and hue to a batch of images.

    The batch is expected to have [B, ..., 1 or 3, H, W] shape,
    where ... means an arbitrary number of dimensions.

    Args:
        brightness: How much to jitter brightness.
            brightness factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast: How much to jitter contrast.
            contrast factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non-negative numbers.
        saturation: How much to jitter saturation.
            saturation factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue: How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
        p: Probability to apply color jitter.
        num_rand_calls: Number of random calls performed to apply augmentations at different orders on sub-batches.
            If -1, B calls are performed. The maximum is 24 = 4!, adjusted automatically if num_rand_calls > 24.
        inplace: If True, perform inplace operation to save memory.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization or large batch size.
    """

    def __init__(
        self,
        brightness: float | tuple[float, float] = 0,
        contrast: float | tuple[float, float] = 0,
        saturation: float | tuple[float, float] = 0,
        hue: float | tuple[float, float] = 0,
        p: float = 0.5,
        num_rand_calls: int = -1,
        inplace: bool = False,
        value_check: bool = False,
    ) -> None:
        super().__init__(p=p, inplace=inplace)
        _log_api_usage_once(self)

        if not isinstance(num_rand_calls, int) or not num_rand_calls >= -1:
            raise ValueError(
                f"num_rand_calls attribute should be an int superior to -1, {num_rand_calls} given."
            )

        self.p = p
        self.num_rand_calls = num_rand_calls
        self.inplace = inplace

        brightness = _check_input(brightness, "brightness")
        if brightness is not None:
            self.register_buffer("brightness", torch.as_tensor(brightness))
        else:
            self.brightness = None

        contrast = _check_input(contrast, "contrast")
        if contrast is not None:
            self.register_buffer("contrast", torch.as_tensor(contrast))
        else:
            self.contrast = None

        saturation = _check_input(saturation, "saturation")
        if saturation is not None:
            self.register_buffer("saturation", torch.as_tensor(saturation))
        else:
            self.saturation = None

        hue = _check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        if hue is not None:
            self.register_buffer("hue", torch.as_tensor(hue))
        else:
            self.hue = None

        self.value_check = value_check

    @staticmethod
    def get_params(
        brightness: Tensor | None,
        contrast: Tensor | None,
        saturation: Tensor | None,
        hue: Tensor | None,
        batch_size: int,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tensor (min, max), optional): The range from which the brightness factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tensor (min, max), optional): The range from which the contrast factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tensor (min, max), optional): The range from which the saturation factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue: The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.
            batch_size: The number of samples to draw.
        """
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

        return b, c, s, h

    def apply_transform(self, imgs: Tensor) -> Tensor:
        """Color jitter the batch of images.

        .. note:: Apply different transformation orders based on :attr:`~num_rand_calls`.

        Args:
            imgs: Batch of images to jitter.

        Returns:
            Randomly color jittered batch of images.
        """
        if self.num_rand_calls == 0:
            return imgs

        batch_size = imgs.shape[0]
        # At most 24 (= 4!) different combinations possible.
        num_combination = min(
            batch_size,
            self.num_rand_calls if self.num_rand_calls != -1 else batch_size,
            24,
        )

        combinations = list(permutations(range(0, 4)))
        idx_perms = torch.randperm(len(combinations))[:num_combination]

        num_apply_per_combination = ceil(batch_size / num_combination)

        (
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue,
            batch_size,
        )

        indices_do_apply = torch.randperm(batch_size, device=imgs.device)

        for i, idx_perm in enumerate(idx_perms):
            indices_combination = indices_do_apply[
                i * num_apply_per_combination : (i + 1) * num_apply_per_combination
            ]

            fn_idx = combinations[idx_perm]
            imgs_combination = imgs[indices_combination]

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    imgs_combination[:] = F_b.batch_adjust_brightness(
                        imgs_combination,
                        brightness_factor[
                            i
                            * num_apply_per_combination : (i + 1)
                            * num_apply_per_combination
                        ],
                        self.value_check,
                    )
                elif fn_id == 1 and contrast_factor is not None:
                    imgs_combination[:] = F_b.batch_adjust_contrast(
                        imgs_combination,
                        contrast_factor[
                            i
                            * num_apply_per_combination : (i + 1)
                            * num_apply_per_combination
                        ],
                        self.value_check,
                    )
                elif fn_id == 2 and saturation_factor is not None:
                    imgs_combination[:] = F_b.batch_adjust_saturation(
                        imgs_combination,
                        saturation_factor[
                            i
                            * num_apply_per_combination : (i + 1)
                            * num_apply_per_combination
                        ],
                        self.value_check,
                    )
                elif fn_id == 3 and hue_factor is not None:
                    imgs_combination[:] = F_b.batch_adjust_hue(
                        imgs_combination,
                        hue_factor[
                            i
                            * num_apply_per_combination : (i + 1)
                            * num_apply_per_combination
                        ],
                        self.value_check,
                    )

            imgs[indices_combination] = imgs_combination

        return imgs

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness.tolist() if self.brightness is not None else None}"
            f", contrast={self.contrast.tolist() if self.contrast is not None else None}"
            f", saturation={self.saturation.tolist() if self.saturation is not None else None}"
            f", hue={self.hue.tolist() if self.hue is not None else None}"
            f", p={self.p}"
            f", num_rand_calls={self.num_rand_calls}"
            f", inplace={self.inplace}"
            f", value_check={self.value_check})"
        )
        return s


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
        inplace: If True, perform inplace operation to save memory.
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


class BatchRandomGrayScale(BatchRandomTransform):
    """Convert batch of images to grayscale.

    The batch of images is expected to be of shape [B, ..., C, H, W]
    where ... means an arbitrary number of dimensions.

    Args:
        p: Probability of the images to be grayscaled.
        inplace: If True, perform inplace operation to save memory.
    """

    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
    ):
        super().__init__(p=p, inplace=inplace)
        _log_api_usage_once(self)

    def apply_transform(self, imgs: Tensor) -> Tensor:
        """Apply grayscale on the batch of images.

        Args:
            imgs: Batch of images to be grayscaled.

        Returns:
            Grayscaled batch of images.
        """
        imgs = F_tv.rgb_to_grayscale(imgs, 3)
        return imgs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"p={self.p}" f", inplace={self.inplace})"


class BatchMixUp(nn.Module):
    """Mix input tensor with linear interpolation drawn according a Beta law.

    The shape of the tensors is expected to be [B, ...] with ... any number of dimensions.
    The tensor shoud be float.

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


class BatchRandomHorizontalFlip(BatchRandomTransform):
    """Horizontally flip the given batch of images randomly with a given probability.

    The batch of images is expected to be of shape [B, ..., C, H, W]
    where ... means an arbitrary number of dimensions.

    Args:
        p: probability of the images being flipped.
        inplace: If True, perform inplace operation to save memory.
    """

    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
    ):
        super().__init__(p=p, inplace=inplace)
        _log_api_usage_once(self)

    def apply_transform(self, imgs: Tensor) -> Tensor:
        """Flip the batch of images.

        Args:
            imgs: Batch of images to be horizontally fliped.

        Returns:
            Horizontally fliped batch of images.
        """
        imgs = F_tv.hflip(imgs)
        return imgs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"p={self.p}" f", inplace={self.inplace})"


class BatchRandomResizedCrop(tv_transforms.RandomResizedCrop):
    """Crop a random portion of a batch of images and resize it to a given size.

    The batch shape is expected to be [B, ..., H, W], where ... means an arbitrary number of dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size: Expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale: Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio: lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation: Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
        antialias: Whether to apply antialiasing.
            It only affects bilinear or bicubic modes and it is
            ignored otherwise.
        num_rand_calls: Number of random calls performed to apply augmentations at different orders on sub-batches.
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
    ) -> None:
        if isinstance(size, int):
            self.size = [size, size]
        elif isinstance(size, Sequence) and isinstance(size[0], int) and len(size) == 1:
            self.size = [size[0], size[0]]
        elif not isinstance(size, Sequence) or not all(
            [isinstance(s, int) for s in size]
        ):
            raise TypeError(f"size should be a int or a sequence of int. Got {size}.")

        super().__init__(size, scale, ratio, interpolation, antialias)
        _log_api_usage_once(self)

        if not isinstance(num_rand_calls, int) or not num_rand_calls >= -1:
            raise ValueError(
                f"num_rand_calls attribute should be an int superior to -1, {num_rand_calls} given."
            )

        self.num_rand_calls = num_rand_calls

    def single_forward(self, imgs: Tensor) -> Tensor:
        """Peform a random resized crop of same scale and shape for the batch of images.

        Args:
            img: Batch of images to be cropped and resized.

        Returns:
            Cropped and resized batch of images.
        """
        i, j, h, w = self.get_params(imgs, self.scale, self.ratio)
        return F_tv.resized_crop(
            imgs, i, j, h, w, self.size, self.interpolation, antialias=self.antialias
        )

    def forward(self, imgs: Tensor) -> Tensor:
        """Resize and crop the batch of images.

        .. note:: Apply different scaling and cropping based on :attr:`~num_rand_calls`.

        Args:
            imgs: Batch of images to be cropped and resized.

        Returns:
            Randomly cropped and resized batch of images.
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


class BatchRandomSolarize(BatchRandomTransform):
    """Solarize the batch of images randomly with a given probability by inverting all pixel values above a
    threshold.

    The batch if images it is expected to be in [B, ..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of dimensions.

    Args:
        threshold: all pixels equal or above this value are inverted.
        p: probability of the image being solarized.
        inplace: If True, perform inplace operation to save memory.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization or large batch size.
    """

    def __init__(
        self,
        threshold: float,
        p: float = 0.5,
        inplace: bool = False,
        value_check: bool = False,
    ) -> None:
        super().__init__(p=p, inplace=inplace)
        _log_api_usage_once(self)

        self.register_buffer("threshold", torch.as_tensor(threshold))
        self.value_check = value_check

    def apply_transform(self, imgs: Tensor) -> Tensor:
        """Solarize the batch of images.

        Args:
            imgs: Batch of images to be solarized.

        Returns:
            Randomly solarized batch of images.
        """
        imgs = F.solarize(imgs, self.threshold, self.value_check)
        return imgs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold.item()}"
            f", p={self.p}"
            f", inplace={self.inplace}"
            f", value_check={self.value_check})"
        )


class BatchVideoWrapper(nn.Module):
    """Wrap a transform to handle batched video data.

    The transform is expected to handle the leading dimension differently.

    The batch of videos is expected to be in format [B, C, T, H, W] or [B, T, C, H, W].

    Args:
        transform: The transform to wrap.
        same_on_frames: If True, apply the same transform on all the frames, else it
            flattens the batch and temporal dimensions to apply different transformations to each frame.
        video_format: Format of the video. Either ``CTHW`` or ``TCHW``.
    """

    def __init__(
        self,
        transform: nn.Module,
        same_on_frames: bool = True,
        video_format: str = "CTHW",
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        self.transform = transform
        self.video_format = video_format
        self.same_on_frames = same_on_frames

        if self.video_format == "CTHW":
            self.time_before_channel = False
        elif self.video_format == "TCHW":
            self.time_before_channel = True
        else:
            raise ValueError(
                f"video_format should be either 'CTHW' or 'TCHW'. Got {self.video_format}."
            )

    def forward(self, videos: Tensor):
        """Apply :attr:`~transform` on the batch of videos.

        .. note:: If :attr:`~same_on_frames` is ``False``, the batch and frames
            dimensions are merged.

        Args:
            videos: The batch of videos to transform.

        Returns:
            The transformed videos.
        """
        _assert_batch_videos_tensor(videos)

        if self.time_before_channel:
            b, t, c, h, w = videos.shape
        else:
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)

        if not self.same_on_frames:
            videos = videos.reshape(b * t, c, h, w)

        videos = self.transform(videos)

        if not self.same_on_frames:
            videos = videos.reshape(b, t, *videos.shape[-3:])

        if not self.time_before_channel:
            videos = videos.permute(0, 2, 1, 3, 4)

        return videos

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    transform={self.transform},\n"
            f"    same_on_frames={self.same_on_frames},\n"
            f"    video_format={self.video_format}\n)"
        )


class BatchVideoResize(nn.Module):
    """Resize the input video to the given size. The video is expected to have [B, ..., H, W] shape, where ...
    means an arbitrary number of dimensions.

    Args:
        size: Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).

            .. note::
                In torchscript mode size as single int is not supported,
                use a sequence of length 1: ``[size, ]``.
        interpolation: Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are
            supported.
        max_size : The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e. the
            smaller edge may be shorter than ``size``. This is only supported
            if ``size`` is an int (or a sequence of length 1 in torchscript
            mode).
        antialias: Whether to apply antialiasing.  If ``True``, will apply antialiasing for bilinear or bicubic modes.
            Other mode aren't affected.
        video_format: Format of the video. Either ``CTHW`` or ``TCHW``.
    """

    def __init__(
        self,
        size: int | list[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: int | None = None,
        antialias: bool = True,
        video_format: str = "CTHW",
    ):
        super().__init__()
        _log_api_usage_once(self)

        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}.")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values.")
        self.size = size
        self.max_size = max_size

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.video_format = video_format

        if self.video_format == "CTHW":
            self.time_before_channel = False
        elif self.video_format == "TCHW":
            self.time_before_channel = True
        else:
            raise ValueError(
                f"video_format should be either 'CTHW' or 'TCHW'. Got {self.video_format}."
            )

    def forward(self, videos: Tensor):
        """Resize the batch of videos.

        Args:
            videos: The batch of videos to resize.

        Returns:
            The resized videos.
        """
        _assert_batch_videos_tensor(videos)

        if self.time_before_channel:
            b, t, c, h, w = videos.shape
        else:
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)

        videos = videos.reshape(b * t, c, h, w)

        videos = F_tv.resize(
            videos, self.size, self.interpolation, self.max_size, self.antialias
        )

        videos = videos.reshape(b, t, *videos.shape[-3:])

        if not self.time_before_channel:
            videos = videos.permute(0, 2, 1, 3, 4)

        return videos

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size={self.size}, "
            f"interpolation={self.interpolation.value}, "
            f"max_size={self.max_size}, "
            f"antialias={self.antialias}, "
            f"video_format={self.video_format})"
        )
