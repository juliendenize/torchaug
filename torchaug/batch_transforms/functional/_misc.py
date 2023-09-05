from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn.functional import conv2d
from torch.nn.functional import pad as torch_pad
from torchvision.transforms._functional_tensor import (_cast_squeeze_in,
                                                       _cast_squeeze_out)

from torchaug.batch_transforms._utils import _assert_batch_images_tensor
from torchaug.transforms._utils import (_assert_tensor,
                                        transfer_tensor_on_device)
from torchaug.utils import _log_api_usage_once


def _get_batch_gaussian_kernel1d(
    kernel_size: int, sigma: Tensor, dtype: torch.dtype, device: torch.device
) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = (
        torch.linspace(
            -ksize_half, ksize_half, steps=kernel_size, dtype=dtype, device=device
        )
        .view(1, -1)
        .expand(sigma.shape[0], -1)
    )
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum(-1, keepdim=True)

    return kernel1d


def _get_batch_gaussian_kernel2d(
    kernel_size: list[int], sigma: Tensor, dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_batch_gaussian_kernel1d(
        kernel_size[0], sigma[:, 0, None], dtype, device
    )[..., None]
    kernel1d_y = _get_batch_gaussian_kernel1d(
        kernel_size[1], sigma[:, 1, None], dtype, device
    )[..., None]
    kernel2d = kernel1d_y * kernel1d_x.view(-1, 1, kernel_size[0])
    return kernel2d


def batch_gaussian_blur(
    imgs: Tensor,
    kernel_size: int | list[int],
    sigma: int | float | list[int] | list[float] | Tensor | None = None,
    value_check: bool = False,
) -> Tensor:
    """Performs Gaussian blurring on the batch of images by given kernel. It is expected to have [B, ..., C, H, W]
    shape, where ... means an arbitrary number of dimensions.

    Args:
        img: Image to be blurred
        kernel_size (sequence of ints or int): Gaussian kernel size. Can be a sequence of integers
            like ``(kx, ky)`` or a single integer for square kernels.

            .. note::
                In torchscript mode kernel_size as single int is not supported, use a sequence of
                length 1: ``[ksize, ]``.
        sigma: Gaussian kernel standard deviation.
            Can be a sequence of floats like ``(sigma_x, sigma_y)`` or a single float to define the
            same sigma in both X/Y directions. If None, then it is computed using
            ``kernel_size`` as ``sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8``.
            If Tensor it is expected to have [B] shape.
        value_check: Bool to perform tensor value check.
            Might cause slow down on some devices because of synchronization or large batch size.

    Returns:
        Gaussian Blurred version of the batch of images.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batch_gaussian_blur)

    _assert_tensor(imgs)
    _assert_batch_images_tensor(imgs)

    if not isinstance(kernel_size, (int, list, tuple)):
        raise TypeError(
            f"kernel_size should be int or a sequence of integers. Got {type(kernel_size)}."
        )
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if len(kernel_size) != 2:
        raise ValueError(
            f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}."
        )
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError(
                f"kernel_size should have odd and positive integers. Got {kernel_size}."
            )

    batch_size = imgs.shape[0]
    if sigma is None:
        sigma_t = torch.tensor(
            [[ksize * 0.15 + 0.35 for ksize in kernel_size]], device=imgs.device
        ).expand((batch_size, 2))
    else:
        if isinstance(sigma, (list, tuple)):
            length = len(sigma)
            if length == 1:
                s = float(sigma[0])
                sigma_t = torch.tensor([[s, s]], device=imgs.device).expand(
                    (batch_size, 2)
                )
            elif length != 2:
                raise ValueError(
                    f"If sigma is a sequence, its length should be 2. Got {length}."
                )
            else:
                sigma_t = torch.tensor([list(sigma)], device=imgs.device).expand(
                    (batch_size, 2)
                )
        elif isinstance(sigma, (int, float)):
            s = float(sigma)
            sigma_t = torch.tensor([[s, s]], device=imgs.device).expand((batch_size, 2))
        elif isinstance(sigma, Tensor):
            sigma_t = transfer_tensor_on_device(sigma, imgs.device, non_blocking=True)

            dim_sigma = sigma_t.ndim
            if dim_sigma in [0, 1]:
                sigma_t = sigma_t.view(-1, 1)
            elif dim_sigma != 2:
                raise ValueError(
                    f"If sigma is a tensor, its dimension should be 0, 1 or 2. Got {dim_sigma}."
                )

            len_sigma, feature_dim_sigma = sigma_t.shape

            if len_sigma == 1:
                sigma_t = sigma_t.view(1, -1).expand((batch_size, feature_dim_sigma))
            elif len_sigma != batch_size:
                raise ValueError(
                    f"If sigma is a tensor, it should contain one or batch size elements. Got {len_sigma}."
                )

            if feature_dim_sigma == 1:
                sigma_t = sigma_t.view(-1, 1).expand((batch_size, 2))
            elif feature_dim_sigma != 2:
                raise ValueError(
                    f"If sigma is a tensor, it should have a feature dim of size one or two. Got {feature_dim_sigma}."
                )

        else:
            raise TypeError(
                f"sigma should be either int, float or sequence of floats or int or tensor. Got {type(sigma)}."
            )

    if (
        (isinstance(sigma, (float, int)) and sigma <= 0)
        or (isinstance(sigma, (list, tuple)) and any([s <= 0 for s in sigma]))
        or (
            isinstance(sigma, (Tensor))
            and value_check
            and not torch.all(torch.gt(sigma, 0))
        )
    ):
        raise ValueError(f"sigma should have positive values.")

    dtype = imgs.dtype if torch.is_floating_point(imgs) else torch.float32

    b, *rest_dims, h, w = imgs.shape
    imgs = imgs.reshape(b, math.prod(rest_dims), h, w)

    device = imgs.device
    kernel = _get_batch_gaussian_kernel2d(
        kernel_size, sigma_t, dtype=dtype, device=device
    )
    kernel = kernel[:, None, ...]
    kernel = kernel.expand(-1, imgs.shape[-3], kernel_size[0], kernel_size[1])
    kernel = kernel.reshape(-1, 1, kernel_size[0], kernel_size[1])

    imgs, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(imgs, [kernel.dtype])

    # padding = (left, right, top, bottom)
    padding = [
        kernel_size[0] // 2,
        kernel_size[0] // 2,
        kernel_size[1] // 2,
        kernel_size[1] // 2,
    ]
    imgs = torch_pad(imgs, padding, mode="reflect")
    imgs = imgs.view(-1, kernel.size(0), imgs.size(-2), imgs.size(-1))
    imgs = conv2d(imgs, kernel, groups=imgs.shape[-3])

    imgs = _cast_squeeze_out(imgs, need_cast, need_squeeze, out_dtype)

    imgs = imgs.reshape(b, *rest_dims, h, w)

    return imgs


def batch_mixup(
    tensor_1: Tensor, tensor_2: Tensor, lam: float | Tensor, inplace: bool = False
) -> Tensor:
    """Mix two tensors with linear interpolation.

    The shape of the tensors is expected to be [B, ...] with ... any number of dimensions.
    The tensors should be floats.

    Args:
        tensor_1: First tensor.
        tensor_2: Second tensor.
        lam: Mixing coefficient. If float, same coefficient is applied on whole batch. If tensor,
            its expecting shape is [B, 1] or [B] or [1].
        inplace: Whether to perform the operation inplace.

    Returns:
        The mixed tensor.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batch_mixup)

    _assert_tensor(tensor_1)
    _assert_tensor(tensor_2)

    if not tensor_1.is_floating_point() or not tensor_2.is_floating_point():
        raise TypeError(
            f"Tensors should be float. Got {tensor_1.dtype} and {tensor_2.dtype}."
        )

    device = tensor_1.device
    batch_size = tensor_1.shape[0]

    if isinstance(lam, float):
        lam = torch.tensor(lam, device=device).expand(batch_size)
    elif isinstance(lam, Tensor):
        lam = transfer_tensor_on_device(lam, device, True)
        dim_lam = lam.ndim
        if dim_lam in [0, 1]:
            lam = lam.view(-1, 1)
        elif dim_lam == 2:
            if lam.shape[1] > 1:
                raise ValueError(
                    f"If lam is a two dimensional tensor, its second dimension should be 1. Got {lam.shape[1]}."
                )
        else:
            raise ValueError(
                f"If lam is a tensor, its dimension should be 0, 1 or 2. Got {dim_lam}."
            )

        len_lam = lam.shape[0]

        if len_lam == 1:
            lam = lam.view(1, 1).expand(batch_size, 1)
        elif len_lam != batch_size:
            raise ValueError(
                f"If lam is a tensor, it should contain one or batch size elements. Got {len_lam}."
            )
    else:
        raise TypeError(f"lam should be either float or tensor. Got {type(lam)}.")

    lam = lam.view(-1, *[1 for _ in range(tensor_1.ndim - 1)])

    tensor_1 = tensor_1 if inplace else tensor_1.clone()
    tensor_2 = tensor_2 if inplace else tensor_2.clone()
    return tensor_1.mul_(lam).add_(tensor_2.mul_(1 - lam))
