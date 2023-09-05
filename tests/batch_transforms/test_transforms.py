from __future__ import annotations

import re
from typing import Any, Sequence

import pytest
import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as F_tv
from torch import nn
from torchvision.transforms.functional import InterpolationMode

import torchaug.batch_transforms as transforms
import torchaug.batch_transforms.functional as F_b
import torchaug.transforms as mono_transforms
import torchaug.transforms.functional as F


class TestBatchImageWrapper:
    def test_output_values(self):
        torch.manual_seed(28)

        transform = (
            mono_transforms.Normalize((0.5,), (0.5,), inplace=False, value_check=True),
            mono_transforms.Mul255(),
        )

        tensor = torch.rand((3, 2, 16, 16))
        expected_out = (
            mono_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor) * 255.0
        )
        out = transforms.BatchImageWrapper(transform)(tensor)
        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "list_transforms,inplace",
        [
            (
                mono_transforms.Normalize(
                    (0.5,), (0.5,), inplace=False, value_check=True
                ),
                False,
            ),
            (
                [
                    mono_transforms.Normalize(
                        (0.5,), (0.5,), inplace=False, value_check=True
                    ),
                    mono_transforms.Div255(inplace=False),
                ],
                True,
            ),
        ],
    )
    def test_functional(
        self, list_transforms: nn.Module | Sequence[nn.Module], inplace: bool
    ):
        torch.manual_seed(28)

        tensor = torch.rand((2, 3, 16, 16))

        input_tensor = tensor.clone() if inplace else tensor

        wrapper = transforms.BatchImageWrapper(list_transforms, inplace)

        if not isinstance(list_transforms, Sequence):
            list_transforms = [list_transforms]

        for transform in list_transforms:
            if hasattr(transform, "inplace"):
                assert transform.inplace

        output = wrapper(input_tensor)

        if inplace:
            torch.testing.assert_close(output, input_tensor)

    @pytest.mark.parametrize(
        "list_transforms,inplace,repr",
        [
            (
                mono_transforms.Normalize(
                    (0.5,), (0.5,), inplace=False, value_check=True
                ),
                False,
                "BatchImageWrapper(\n    inplace=False,\n    transforms=ModuleList(\n      (0): Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True)\n    )\n)",
            ),
            (
                [
                    mono_transforms.Normalize(
                        (0.5,), (0.5,), inplace=False, value_check=True
                    ),
                    mono_transforms.Div255(inplace=False),
                ],
                True,
                "BatchImageWrapper(\n    inplace=True,\n    transforms=ModuleList(\n      (0): Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True)\n      (1): Div255(inplace=True)\n    )\n)",
            ),
        ],
    )
    def test_repr(
        self, list_transforms: nn.Module | Sequence[nn.Module], inplace: bool, repr: str
    ):
        assert transforms.BatchImageWrapper(list_transforms, inplace).__repr__() == repr

    def test_wrong_tensor(self):
        transform = mono_transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        tensor = torch.rand((3, 16, 16))
        with pytest.raises(TypeError):
            transforms.BatchVideoWrapper(transform)(tensor)

    def test_wrong_transforms(self):
        with pytest.raises(TypeError):
            transforms.BatchImageWrapper("ahah")


class TestBatchMixUp:
    def test_functional(self):
        torch.manual_seed(28)

        imgs = torch.randn(4, 3, 8, 8)
        labels = torch.randint(0, 5, (4, 1)).to(torch.float)
        inpt_imgs = imgs.clone()
        inpt_labels = labels.clone()
        out_imgs, out_labels, out_lam = transforms.BatchMixUp(0.5, inplace=True)(
            inpt_imgs, inpt_labels
        )

        expected_lam = torch.tensor(
            [[0.8844036], [0.9952562], [0.0072862], [0.0252598]]
        )
        expected_out_labels = F_b.batch_mixup(
            labels, labels.roll(1, 0), expected_lam, False
        )
        expected_out_imgs = F_b.batch_mixup(imgs, imgs.roll(1, 0), expected_lam, False)

        torch.testing.assert_close(out_lam, expected_lam.view(4, 1))
        torch.testing.assert_close(out_labels, expected_out_labels)
        torch.testing.assert_close(out_imgs, expected_out_imgs)
        torch.testing.assert_close(out_imgs, inpt_imgs)
        torch.testing.assert_close(out_labels, inpt_labels)

        out_imgs, out_labels, out_lam = transforms.BatchMixUp(0.5, inplace=True)(
            inpt_imgs, None
        )
        assert out_labels is None

        out_imgs, out_labels, out_lam = transforms.BatchMixUp(0.5, inplace=False)(
            imgs, None
        )
        assert out_labels is None

        expected_lam = torch.tensor(
            [[0.9992378], [0.0766915], [0.9603495], [0.9419856]]
        )
        torch.testing.assert_close(out_lam, expected_lam.view(4, 1))
        expected_out_imgs = F_b.batch_mixup(imgs, imgs.roll(1, 0), expected_lam, False)
        torch.testing.assert_close(out_imgs, expected_out_imgs)

    def test_repr(self):
        assert (
            transforms.BatchMixUp(0.5, True).__repr__()
            == "BatchMixUp(alpha=0.5, inplace=True)"
        )
        assert (
            transforms.BatchMixUp(0.5, False).__repr__()
            == "BatchMixUp(alpha=0.5, inplace=False)"
        )


class TestBatchRandomApply:
    @pytest.mark.parametrize(
        "p,apply_transforms,inplace,seed",
        [
            (
                0,
                [
                    mono_transforms.Normalize((0.5,), (0.5,)),
                    transforms.BatchRandomGaussianBlur((3, 3), (0.1, 2.0)),
                ],
                False,
                28,
            ),
            (
                0.5,
                [
                    mono_transforms.Normalize((0.5,), (0.5,)),
                    transforms.BatchRandomGaussianBlur((3, 3), (0.1, 2.0)),
                ],
                False,
                28,
            ),
            (
                1.0,
                [
                    mono_transforms.Normalize((0.5,), (0.5,)),
                    transforms.BatchRandomGaussianBlur((3, 3), (0.1, 2.0)),
                ],
                False,
                28,
            ),
            (1.0, mono_transforms.Normalize((0.5,), (0.5,)), False, 28),
            (1.0, mono_transforms.Normalize((0.5,), (0.5,)), True, 28),
            (0.1, mono_transforms.Normalize((0.5,), (0.5,)), True, 28),
            (0.3, mono_transforms.Normalize((0.5,), (0.5,)), True, 28),
        ],
    )
    def test_functional(
        self,
        p: float,
        apply_transforms: torch.nn.Module | list[torch.nn.Module],
        inplace: bool,
        seed: int,
    ):
        torch.manual_seed(seed)
        imgs = torch.randn(4, 3, 8, 8)
        transforms.BatchRandomApply(apply_transforms, p, inplace)(imgs)

    @pytest.mark.parametrize(
        "p,apply_transforms,inplace,repr",
        [
            (
                1.0,
                mono_transforms.Normalize((0.5,), (0.5,)),
                False,
                "BatchRandomApply(\n    p=1.0,\n    inplace=False,\n    transforms=ModuleList(\n      (0): Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=False, value_check=False)\n    )\n)",
            ),
            (
                1.0,
                mono_transforms.Normalize((0.5,), (0.5,)),
                True,
                "BatchRandomApply(\n    p=1.0,\n    inplace=True,\n    transforms=ModuleList(\n      (0): Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=False, value_check=False)\n    )\n)",
            ),
        ],
    )
    def test_repr(
        self,
        apply_transforms: torch.nn.Module | list[torch.nn.Module],
        p: float,
        inplace: bool,
        repr: str,
    ):
        assert (
            transforms.BatchRandomApply(apply_transforms, p, inplace).__repr__() == repr
        )

    @pytest.mark.parametrize("p, error_type", [(-0.1, ValueError), (1.1, ValueError)])
    def test_wrong_p(self, p: float, error_type: Exception):
        with pytest.raises(error_type):
            transforms.BatchRandomApply(
                [mono_transforms.Normalize([225, 225, 225], [0.25, 0.25, 0.25])],
                p,
            )


class TestBatchRandomColorJitter:
    def test_output_values(self):
        torch.manual_seed(28)

        imgs = torch.randn(8, 3, 8, 8)
        indices_to_apply = torch.tensor([[1, 0], [7, 5]])
        orders = torch.tensor([[1, 0, 2, 3], [2, 0, 1, 3]])
        brightnesss = torch.tensor(
            [
                [1.4192447662353516, 0.9388960599899292],
                [0.9894483685493469, 1.0712565183639526],
            ]
        )
        contrasts = torch.tensor(
            [
                [1.1582083702087402, 1.0906667709350586],
                [1.1031641960144043, 0.5347495079040527],
            ]
        )
        saturations = torch.tensor(
            [
                [0.76619952917099, 0.6456150412559509],
                [1.4253023862838745, 1.3242526054382324],
            ]
        )
        hues = torch.tensor(
            [
                [-0.016170598566532135, -0.03358305245637894],
                [0.06692638993263245, 0.06578600406646729],
            ]
        )

        expected_out = imgs.clone()

        for indices, order, brightnesss, contrast, saturation, hue in zip(
            indices_to_apply, orders, brightnesss, contrasts, saturations, hues
        ):
            for fn_id in order:
                if fn_id == 0 and brightnesss is not None:
                    expected_out[indices] = F_b.batch_adjust_brightness(
                        expected_out[indices], brightnesss
                    )
                elif fn_id == 1 and contrast is not None:
                    expected_out[indices] = F_b.batch_adjust_contrast(
                        expected_out[indices], contrast
                    )
                elif fn_id == 2 and saturation is not None:
                    expected_out[indices] = F_b.batch_adjust_saturation(
                        expected_out[indices], saturation
                    )
                elif fn_id == 3 and hue is not None:
                    expected_out[indices] = F_b.batch_adjust_hue(
                        expected_out[indices], hue
                    )
        out = transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 2)(imgs)
        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "brightness,contrast,saturation,hue,p,num_rand_calls,inplace,value_check",
        [
            (0.5, 0.5, 0.5, 0.1, 0.5, 16, False, False),
            (0.5, 0.5, 0.5, 0.1, 0.5, 16, False, True),
            (0.5, 0.5, 0.5, 0.1, 0.5, 16, True, False),
            (0.0, 0.5, 0.5, 0.1, 0.5, 16, False, False),
            (0.5, 0, 0.5, 0.1, 0.5, 16, False, False),
            (0.5, 0.5, 0.0, 0.1, 0.5, 16, False, False),
            (0.5, 0.5, 0.5, 0.0, 0.5, 16, False, False),
            (0.5, 0.5, 0.5, 0.5, 0.0, 16, False, False),
            (0.5, 0.5, 0.5, 0.5, 1.0, 16, False, False),
            (0.5, 0.5, 0.5, 0.5, 1.0, -1, False, False),
            (0.5, 0.5, 0.5, 0.5, 1.0, 0, False, False),
            (0.5, 0.5, 0.5, 0.5, 1.0, 64, False, False),
        ],
    )
    def test_functional(
        self,
        brightness: float | tuple[float, float] | None,
        contrast: float | tuple[float, float] | None,
        saturation: float | tuple[float, float] | None,
        hue: float | tuple[float, float] | None,
        p: float,
        num_rand_calls: int,
        inplace: bool,
        value_check: bool,
    ):
        imgs = torch.randn(8, 3, 8, 8)
        transforms.BatchRandomColorJitter(
            brightness,
            contrast,
            saturation,
            hue,
            p,
            num_rand_calls,
            inplace,
            value_check,
        )(imgs)

    @pytest.mark.parametrize(
        "brightness,contrast,saturation,hue,p,num_rand_calls,inplace,value_check,repr",
        [
            (
                0.5,
                0.5,
                0.5,
                0.1,
                0.5,
                16,
                False,
                False,
                r"BatchRandomColorJitter\(brightness=\[0\.\d+, 1\.\d+\], contrast=\[0\.\d+, 1\.\d+\], saturation=\[0\.\d+, 1\.\d+\], hue=\[-0\.\d+, 0\.\d+\], p=0.5, num_rand_calls=16, inplace=False, value_check=False\)",
            ),
            (
                0.5,
                0.5,
                0.5,
                0.1,
                0.5,
                16,
                False,
                True,
                r"BatchRandomColorJitter\(brightness=\[0\.\d+, 1\.\d+\], contrast=\[0\.\d+, 1\.\d+\], saturation=\[0\.\d+, 1\.\d+\], hue=\[-0\.\d+, 0\.\d+\], p=0.5, num_rand_calls=16, inplace=False, value_check=True\)",
            ),
            (
                0.5,
                0.5,
                0.5,
                0.1,
                0.5,
                16,
                True,
                False,
                r"BatchRandomColorJitter\(brightness=\[0\.\d+, 1\.\d+\], contrast=\[0\.\d+, 1\.\d+\], saturation=\[0\.\d+, 1\.\d+\], hue=\[-0\.\d+, 0\.\d+\], p=0.5, num_rand_calls=16, inplace=True, value_check=False\)",
            ),
            (
                0.0,
                0.5,
                0.5,
                0.1,
                0.5,
                16,
                False,
                False,
                r"BatchRandomColorJitter\(brightness=None, contrast=\[0\.\d+, 1\.\d+\], saturation=\[0\.\d+, 1\.\d+\], hue=\[-0\.\d+, 0\.\d+\], p=0.5, num_rand_calls=16, inplace=False, value_check=False\)",
            ),
            (
                0.5,
                0,
                0.5,
                0.1,
                0.5,
                16,
                False,
                False,
                r"BatchRandomColorJitter\(brightness=\[0\.\d+, 1\.\d+\], contrast=None, saturation=\[0\.\d+, 1\.\d+\], hue=\[-0\.\d+, 0\.\d+\], p=0.5, num_rand_calls=16, inplace=False, value_check=False\)",
            ),
            (
                0.5,
                0.5,
                0.0,
                0.1,
                0.5,
                16,
                False,
                False,
                r"BatchRandomColorJitter\(brightness=\[0\.\d+, 1\.\d+\], contrast=\[0\.\d+, 1\.\d+\], saturation=None, hue=\[-0\.\d+, 0\.\d+\], p=0.5, num_rand_calls=16, inplace=False, value_check=False\)",
            ),
            (
                0.5,
                0.5,
                0.5,
                0.0,
                0.5,
                16,
                False,
                False,
                r"BatchRandomColorJitter\(brightness=\[0\.\d+, 1\.\d+\], contrast=\[0\.\d+, 1\.\d+\], saturation=\[0\.\d+, 1\.\d+\], hue=None, p=0.5, num_rand_calls=16, inplace=False, value_check=False\)",
            ),
            (
                0.5,
                0.5,
                0.5,
                0.5,
                0.0,
                16,
                False,
                False,
                r"BatchRandomColorJitter\(brightness=\[0\.\d+, 1\.\d+\], contrast=\[0\.\d+, 1\.\d+\], saturation=\[0\.\d+, 1\.\d+\], hue=\[-0\.\d+, 0\.\d+\], p=0.0, num_rand_calls=16, inplace=False, value_check=False\)",
            ),
            (
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                16,
                False,
                False,
                r"BatchRandomColorJitter\(brightness=\[0\.\d+, 1\.\d+\], contrast=\[0\.\d+, 1\.\d+\], saturation=\[0\.\d+, 1\.\d+\], hue=\[-0\.\d+, 0\.\d+\], p=1.0, num_rand_calls=16, inplace=False, value_check=False\)",
            ),
            (
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                -1,
                False,
                False,
                r"BatchRandomColorJitter\(brightness=\[0\.\d+, 1\.\d+\], contrast=\[0\.\d+, 1\.\d+\], saturation=\[0\.\d+, 1\.\d+\], hue=\[-0\.\d+, 0\.\d+\], p=1.0, num_rand_calls=-1, inplace=False, value_check=False\)",
            ),
            (
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                64,
                False,
                False,
                r"BatchRandomColorJitter\(brightness=\[0\.\d+, 1\.\d+\], contrast=\[0\.\d+, 1\.\d+\], saturation=\[0\.\d+, 1\.\d+\], hue=\[-0\.\d+, 0\.\d+\], p=1.0, num_rand_calls=64, inplace=False, value_check=False\)",
            ),
        ],
    )
    def test_repr(
        self,
        brightness: float | tuple[float, float] | None,
        contrast: float | tuple[float, float] | None,
        saturation: float | tuple[float, float] | None,
        hue: float | tuple[float, float] | None,
        p: float,
        num_rand_calls: int,
        inplace: bool,
        value_check: bool,
        repr: str,
    ):
        assert re.match(
            repr,
            transforms.BatchRandomColorJitter(
                brightness,
                contrast,
                saturation,
                hue,
                p,
                num_rand_calls,
                inplace,
                value_check,
            ).__repr__(),
        )

    @pytest.mark.parametrize(
        "num_rand_calls, error_type", [(-10, ValueError), ("ahah", ValueError)]
    )
    def test_wrong_num_rand_calls(self, num_rand_calls: Any, error_type: Exception):
        with pytest.raises(error_type):
            transforms.BatchRandomColorJitter(num_rand_calls=num_rand_calls)


class TestBatchRandomGaussianBlur:
    def test_output_values(self):
        torch.manual_seed(28)

        imgs = torch.randn(4, 3, 8, 8)

        indices_to_apply = torch.tensor([2, 1])
        sigmas = torch.tensor([[0.7487103, 0.7487103], [0.8353596, 0.8353596]])

        expected_out = imgs.clone()
        expected_out[indices_to_apply] = F_b.batch_gaussian_blur(
            imgs[indices_to_apply], [3, 3], sigmas
        )

        out = transforms.BatchRandomGaussianBlur([3, 3], [0.1, 2.0], 0.5)(imgs)

        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "kernel_size,sigma,p,inplace,value_check",
        [
            ([3, 3], [0.1, 2.0], 0.5, False, False),
            ([3, 3], [0.1, 2.0], 0.0, False, False),
            ([3, 3], [0.1, 2.0], 1.0, False, False),
            ([3, 3], [0.1, 2.0], 1.0, True, False),
            ([3, 3], [0.1, 2.0], 1.0, True, True),
            ([3, 3], 1.0, 1.0, True, True),
            ([3], 1.0, 1.0, True, True),
            (3, 1.0, 1.0, True, True),
        ],
    )
    def test_functional(
        self,
        kernel_size: int | tuple[int, int],
        sigma: float | tuple[float, float],
        p: float,
        inplace: bool,
        value_check: bool,
    ):
        imgs = torch.randn(4, 3, 8, 8)
        transforms.BatchRandomGaussianBlur(kernel_size, sigma, p, inplace, value_check)(
            imgs
        )

    @pytest.mark.parametrize(
        "kernel_size,sigma,p,inplace,value_check,repr",
        [
            (
                [3, 3],
                [0.1, 2.0],
                0.5,
                False,
                False,
                r"BatchRandomGaussianBlur\(kernel_size=\[3, 3\], sigma=\[0\.\d+, \d\.\d+\], p=0.5, inplace=False, value_check=False\)",
            ),
            (
                [3, 3],
                [0.1, 2.0],
                0.0,
                False,
                False,
                r"BatchRandomGaussianBlur\(kernel_size=\[3, 3\], sigma=\[0\.\d+, \d\.\d+\], p=0.0, inplace=False, value_check=False\)",
            ),
            (
                [3, 3],
                [0.1, 2.0],
                1.0,
                False,
                False,
                r"BatchRandomGaussianBlur\(kernel_size=\[3, 3\], sigma=\[0\.\d+, \d\.\d+\], p=1.0, inplace=False, value_check=False\)",
            ),
            (
                [3, 3],
                [0.1, 2.0],
                1.0,
                True,
                False,
                r"BatchRandomGaussianBlur\(kernel_size=\[3, 3\], sigma=\[0\.\d+, \d\.\d+\], p=1.0, inplace=True, value_check=False\)",
            ),
            (
                [3, 3],
                [0.1, 2.0],
                1.0,
                True,
                True,
                r"BatchRandomGaussianBlur\(kernel_size=\[3, 3\], sigma=\[0\.\d+, \d\.\d+\], p=1.0, inplace=True, value_check=True\)",
            ),
            (
                [3, 3],
                1.0,
                1.0,
                True,
                True,
                r"BatchRandomGaussianBlur\(kernel_size=\[3, 3\], sigma=\[1\.\d+, 1\.\d+\], p=1.0, inplace=True, value_check=True\)",
            ),
            (
                [3],
                1.0,
                1.0,
                True,
                True,
                r"BatchRandomGaussianBlur\(kernel_size=\(3, 3\), sigma=\[1\.\d+, 1\.\d+\], p=1.0, inplace=True, value_check=True\)",
            ),
            (
                3,
                1.0,
                1.0,
                True,
                True,
                r"BatchRandomGaussianBlur\(kernel_size=\(3, 3\), sigma=\[1\.\d+, 1\.\d+\], p=1.0, inplace=True, value_check=True\)",
            ),
        ],
    )
    def test_repr(
        self,
        kernel_size: int | tuple[int, int],
        sigma: float | tuple[float, float],
        p: float,
        inplace: bool,
        value_check: bool,
        repr: str,
    ):
        assert re.match(
            repr,
            transforms.BatchRandomGaussianBlur(
                kernel_size,
                sigma,
                p,
                inplace,
                value_check,
            ).__repr__(),
        )

    @pytest.mark.parametrize(
        "kernel_size,error_type", [(0, ValueError), (-1, ValueError)]
    )
    def test_wrong_kernel_size(self, kernel_size: Any, error_type: Exception):
        with pytest.raises(error_type):
            transforms.BatchRandomGaussianBlur(
                kernel_size, [0.1, 2.0], 0.5, inplace=False
            )

    @pytest.mark.parametrize(
        "sigma,error_type",
        [
            (0, ValueError),
            (-1, ValueError),
            (0.0, ValueError),
            (-1.0, ValueError),
            ([2.0, 1.0], ValueError),
            ([0.0, 1.0], ValueError),
            ([2.0, 3.0, 4.0], ValueError),
            ("ahah", ValueError),
        ],
    )
    def test_wrong_sigma(self, sigma: Any, error_type: Exception):
        with pytest.raises(error_type):
            transforms.BatchRandomGaussianBlur(3, sigma, 0.5, inplace=False)


class TestBatchRandomGrayScale:
    def test_output_values(self):
        torch.manual_seed(28)
        imgs = torch.randn(4, 3, 8, 8)
        indices_to_apply = torch.tensor([2, 1])

        expected_out = imgs.clone()
        expected_out[indices_to_apply] = F_tv.rgb_to_grayscale(
            imgs[indices_to_apply], 3
        )

        out = transforms.BatchRandomGrayScale(0.5)(imgs)
        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "p,inplace", [(0.5, False), (1.0, False), (0.0, False), (0.5, True)]
    )
    def test_functional(self, p: float, inplace: bool):
        imgs = torch.randn(4, 3, 8, 8)
        transforms.BatchRandomGrayScale(p, inplace)(imgs)

    @pytest.mark.parametrize(
        "p,inplace,repr",
        [
            (0.5, False, "BatchRandomGrayScale(p=0.5, inplace=False)"),
            (1.0, False, "BatchRandomGrayScale(p=1.0, inplace=False)"),
            (0.0, False, "BatchRandomGrayScale(p=0.0, inplace=False)"),
            (0.5, True, "BatchRandomGrayScale(p=0.5, inplace=True)"),
        ],
    )
    def test_repr(self, p: float, inplace: bool, repr: str):
        assert transforms.BatchRandomGrayScale(p, inplace).__repr__() == repr


class TestBatchRandomHorizontalFlip:
    def test_output_values(self):
        torch.manual_seed(28)
        imgs = torch.randn(4, 3, 8, 8)
        indices_to_apply = torch.tensor([2, 1])

        expected_out = imgs.clone()
        expected_out[indices_to_apply] = F_tv.hflip(imgs[indices_to_apply])

        out = transforms.BatchRandomHorizontalFlip(0.5)(imgs)
        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "p,inplace", [(0.5, False), (1.0, False), (0.0, False), (0.5, True)]
    )
    def test_functional(self, p: float, inplace: bool):
        imgs = torch.randn(4, 3, 8, 8)
        transforms.BatchRandomHorizontalFlip(p, inplace)(imgs)

    @pytest.mark.parametrize(
        "p,inplace,repr",
        [
            (0.5, False, "BatchRandomHorizontalFlip(p=0.5, inplace=False)"),
            (1.0, False, "BatchRandomHorizontalFlip(p=1.0, inplace=False)"),
            (0.0, False, "BatchRandomHorizontalFlip(p=0.0, inplace=False)"),
            (0.5, True, "BatchRandomHorizontalFlip(p=0.5, inplace=True)"),
        ],
    )
    def test_repr(self, p: float, inplace: bool, repr: str):
        assert transforms.BatchRandomHorizontalFlip(p, inplace).__repr__() == repr


class TestBatchRandomResizedCrop:
    def test_output_values(self):
        torch.manual_seed(28)

        imgs = torch.randn(8, 3, 8, 8)
        indices_to_apply = torch.tensor([0, 1, 7, 5, 6, 4, 2, 3])

        i_s = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
        j_s = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2])
        h_s = torch.tensor([7, 7, 7, 7, 4, 4, 4, 4])
        w_s = torch.tensor([8, 8, 8, 8, 4, 4, 4, 4])

        expected_out = torch.empty((8, 3, 4, 4))

        for idx, i, j, h, w in zip(indices_to_apply, i_s, j_s, h_s, w_s):
            expected_out[idx] = F_tv.resized_crop(
                imgs[idx],
                i,
                j,
                h,
                w,
                [4, 4],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

        out = transforms.BatchRandomResizedCrop(
            4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=2
        )(imgs)

        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "size,scale,ratio,interpolation,antialias,num_rand_calls",
        [
            (4, (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BILINEAR, True, -1),
            (4, (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BILINEAR, True, 0),
            (4, (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BILINEAR, True, 16),
            ([4], (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BILINEAR, True, 16),
            ([4], (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BICUBIC, False, 16),
        ],
    )
    def test_functional(
        self,
        size: int | Sequence[int],
        scale: Sequence[float],
        ratio: Sequence[float],
        interpolation: InterpolationMode,
        antialias: bool,
        num_rand_calls: int,
    ):
        imgs = torch.randn(8, 3, 8, 8)

        transforms.BatchRandomResizedCrop(
            size, scale, ratio, interpolation, antialias, num_rand_calls
        )(imgs)

    @pytest.mark.parametrize(
        "size,scale,ratio,interpolation,antialias,num_rand_calls,repr",
        [
            (
                4,
                (0.08, 1),
                (3 / 4, 4 / 3),
                InterpolationMode.BILINEAR,
                True,
                -1,
                "BatchRandomResizedCrop(size=(4, 4), scale=(0.08, 1), ratio=(0.75, 1.3333), interpolation=bilinear, antialias=True, num_rand_calls=-1)",
            ),
            (
                4,
                (0.08, 1),
                (3 / 4, 4 / 3),
                InterpolationMode.BILINEAR,
                True,
                0,
                "BatchRandomResizedCrop(size=(4, 4), scale=(0.08, 1), ratio=(0.75, 1.3333), interpolation=bilinear, antialias=True, num_rand_calls=0)",
            ),
            (
                4,
                (0.08, 1),
                (3 / 4, 4 / 3),
                InterpolationMode.BILINEAR,
                True,
                16,
                "BatchRandomResizedCrop(size=(4, 4), scale=(0.08, 1), ratio=(0.75, 1.3333), interpolation=bilinear, antialias=True, num_rand_calls=16)",
            ),
            (
                [4],
                (0.08, 1),
                (3 / 4, 4 / 3),
                InterpolationMode.BILINEAR,
                True,
                16,
                "BatchRandomResizedCrop(size=(4, 4), scale=(0.08, 1), ratio=(0.75, 1.3333), interpolation=bilinear, antialias=True, num_rand_calls=16)",
            ),
            (
                [4],
                (0.08, 1),
                (3 / 4, 4 / 3),
                InterpolationMode.BICUBIC,
                False,
                16,
                "BatchRandomResizedCrop(size=(4, 4), scale=(0.08, 1), ratio=(0.75, 1.3333), interpolation=bicubic, antialias=False, num_rand_calls=16)",
            ),
        ],
    )
    def test_repr(
        self,
        size: int | Sequence[int],
        scale: Sequence[float],
        ratio: Sequence[float],
        interpolation: InterpolationMode,
        antialias: bool,
        num_rand_calls,
        repr: str,
    ):
        assert (
            transforms.BatchRandomResizedCrop(
                size, scale, ratio, interpolation, antialias, num_rand_calls
            ).__repr__()
            == repr
        )

    @pytest.mark.parametrize(
        "num_rand_calls,error_type", [("ahah", ValueError), (-2, ValueError)]
    )
    def test_wrong_num_rand_calls(self, num_rand_calls: Any, error_type: Exception):
        with pytest.raises(error_type):
            transforms.BatchRandomResizedCrop(
                4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=num_rand_calls
            )

    @pytest.mark.parametrize("size,error_type", [(4.0, TypeError), ("ahah", TypeError)])
    def test_wrong_size(self, size: Any, error_type: Exception):
        with pytest.raises(error_type):
            transforms.BatchRandomResizedCrop(size, (0.08, 1), (3 / 4, 4 / 3))


class TestBatchRandomSolarize:
    def test_output_values(self):
        torch.manual_seed(28)
        imgs = torch.randn(4, 3, 8, 8)
        indices_to_apply = torch.tensor([2, 1])

        expected_out = imgs.clone()
        expected_out[indices_to_apply] = F.solarize(imgs[indices_to_apply], 0.5)

        out = transforms.BatchRandomSolarize(0.5, 0.5)(imgs)
        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "threshold,p,inplace,value_check",
        [
            (0.5, 0.5, False, False),
            (0.5, 1.0, False, False),
            (0.5, 0.0, False, False),
            (0.5, 0.5, True, True),
        ],
    )
    def test_functional(
        self, threshold: float, p: float, inplace: bool, value_check: bool
    ):
        imgs = torch.randn(4, 3, 8, 8)
        transforms.BatchRandomSolarize(threshold, p, inplace, value_check)(imgs)

    @pytest.mark.parametrize(
        "threshold,p,inplace,value_check,repr",
        [
            (
                0.5,
                0.5,
                False,
                False,
                "BatchRandomSolarize(threshold=0.5, p=0.5, inplace=False, value_check=False)",
            ),
            (
                0.5,
                1.0,
                False,
                False,
                "BatchRandomSolarize(threshold=0.5, p=1.0, inplace=False, value_check=False)",
            ),
            (
                0.5,
                0.0,
                False,
                False,
                "BatchRandomSolarize(threshold=0.5, p=0.0, inplace=False, value_check=False)",
            ),
            (
                0.5,
                0.5,
                True,
                True,
                "BatchRandomSolarize(threshold=0.5, p=0.5, inplace=True, value_check=True)",
            ),
        ],
    )
    def test_repr(
        self, threshold: float, p: float, inplace: bool, value_check: bool, repr: str
    ):
        assert (
            transforms.BatchRandomSolarize(
                threshold, p, inplace, value_check
            ).__repr__()
            == repr
        )


class TestBatchVideoWrapper:
    def test_output_values(self):
        torch.manual_seed(28)

        transform = (
            mono_transforms.Normalize((0.5,), (0.5,), inplace=False, value_check=True),
            mono_transforms.Mul255(),
        )

        # test CTHW format
        tensor = torch.rand((2, 3, 2, 16, 16))
        expected_out = (
            mono_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor) * 255.0
        )
        out = transforms.BatchVideoWrapper(transform)(tensor)
        torch.testing.assert_close(out, expected_out)

        # test TCHW format
        expected_out = (
            mono_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor) * 255.0
        )
        out = transforms.BatchVideoWrapper(transform, video_format="TCHW")(
            tensor.permute(0, 2, 1, 3, 4)
        )
        torch.testing.assert_close(out.permute(0, 2, 1, 3, 4), expected_out)

        # test same_on_frames True
        image = torch.randn((3, 224, 224))
        video = torch.stack([image, image])
        batch_video = torch.stack([video, video])
        out = transforms.BatchVideoWrapper(
            transforms.BatchRandomColorJitter(0.5, p=1.0),
            video_format="TCHW",
            same_on_frames=True,
        )(batch_video)
        torch.testing.assert_close(out[:, 0], out[:, 1])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out[0], out[1])

        # test same_on_frames False
        out = transforms.BatchVideoWrapper(
            transforms.BatchRandomColorJitter(0.5, p=1.0),
            video_format="TCHW",
            same_on_frames=False,
        )(batch_video)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out[:, 0], out[:, 1])

    @pytest.mark.parametrize(
        "list_transforms,inplace,same_on_frames,video_format",
        [
            (
                mono_transforms.VideoNormalize(
                    (0.5, 0.5, 0.5),
                    (0.5,),
                    inplace=False,
                    value_check=True,
                    video_format="CTHW",
                ),
                False,
                False,
                "CTHW",
            ),
            (
                mono_transforms.VideoNormalize(
                    (0.5, 0.5, 0.5),
                    (0.5,),
                    inplace=False,
                    value_check=True,
                    video_format="TCHW",
                ),
                True,
                False,
                "CTHW",
            ),
            (
                [
                    mono_transforms.VideoNormalize(
                        (0.5, 0.5, 0.5),
                        (0.5,),
                        inplace=False,
                        value_check=True,
                        video_format="CTHW",
                    ),
                    mono_transforms.Div255(inplace=False),
                ],
                False,
                True,
                "CTHW",
            ),
            (
                [
                    mono_transforms.VideoNormalize(
                        (0.5, 0.5, 0.5),
                        (0.5,),
                        inplace=True,
                        value_check=True,
                        video_format="CTHW",
                    ),
                    mono_transforms.Div255(inplace=False),
                ],
                True,
                True,
                "TCHW",
            ),
        ],
    )
    def test_functional(
        self,
        list_transforms: nn.Module | Sequence[nn.Module],
        inplace: bool,
        same_on_frames: bool,
        video_format: str,
    ):
        torch.manual_seed(28)

        if video_format == "CTHW":
            tensor = torch.rand((2, 3, 5, 16, 16))
        else:
            tensor = torch.rand((2, 5, 3, 16, 16))

        input_tensor = tensor.clone() if inplace else tensor

        wrapper = transforms.BatchVideoWrapper(
            list_transforms, inplace, same_on_frames, video_format
        )

        if not isinstance(list_transforms, Sequence):
            list_transforms = [list_transforms]

        for transform in list_transforms:
            if hasattr(transform, "inplace"):
                assert transform.inplace
            if hasattr(transform, "video_format"):
                assert transform.video_format == "TCHW"

        output = wrapper(input_tensor)

        if inplace and not video_format == "CTHW":
            torch.testing.assert_close(output, input_tensor)

    @pytest.mark.parametrize(
        "list_transforms,inplace,same_on_frames,video_format,repr",
        [
            (
                mono_transforms.VideoNormalize(
                    (0.5,), (0.5,), inplace=False, value_check=True, video_format="CTHW"
                ),
                False,
                False,
                "TCHW",
                "BatchVideoWrapper(\n    inplace=False,\n    same_on_frames=False,\n    video_format=TCHW,\n    transforms=ModuleList(\n      (0): VideoNormalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True, video_format=TCHW)\n    )\n)",
            ),
            (
                [
                    mono_transforms.VideoNormalize(
                        (0.5,),
                        (0.5,),
                        inplace=False,
                        value_check=True,
                        video_format="TCHW",
                    ),
                    mono_transforms.Div255(inplace=False),
                ],
                False,
                True,
                "CTHW",
                "BatchVideoWrapper(\n    inplace=False,\n    same_on_frames=True,\n    video_format=CTHW,\n    transforms=ModuleList(\n      (0): VideoNormalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True, video_format=TCHW)\n      (1): Div255(inplace=True)\n    )\n)",
            ),
        ],
    )
    def test_repr(
        self,
        list_transforms: nn.Module | Sequence[nn.Module],
        inplace: bool,
        same_on_frames: bool,
        video_format: str,
        repr: str,
    ):
        assert (
            transforms.BatchVideoWrapper(
                list_transforms, inplace, same_on_frames, video_format
            ).__repr__()
            == repr
        )

    def test_wrong_video_format(self):
        transform = mono_transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        with pytest.raises(ValueError):
            transforms.BatchVideoWrapper(transform, video_format="ahah")

    def test_wrong_tensor(self):
        transform = mono_transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        tensor = torch.rand((6, 3, 2, 3, 16, 16))
        with pytest.raises(TypeError):
            transforms.BatchVideoWrapper(transform)(tensor)

    def test_wrong_transforms(self):
        with pytest.raises(TypeError):
            transforms.BatchVideoWrapper("ahah")


class TestBatchVideoResize:
    def test_functional(self):
        # test CTHW format
        tensor = torch.rand((2, 3, 8, 16, 16))
        expected_out = (
            tv_transforms.Resize(2, antialias=True)(
                tensor.permute(0, 2, 1, 3, 4).reshape(2 * 8, 3, 16, 16)
            )
            .reshape(2, 8, 3, 2, 2)
            .permute(0, 2, 1, 3, 4)
        )
        out = transforms.BatchVideoResize(size=2, video_format="CTHW")(tensor)
        torch.testing.assert_close(out, expected_out)

        # test TCHW format
        tensor = torch.rand((2, 8, 3, 16, 16))
        expected_out = tv_transforms.Resize(2, antialias=True)(
            tensor.reshape(2 * 8, 3, 16, 16)
        ).reshape(2, 8, 3, 2, 2)
        out = transforms.BatchVideoResize(size=2, video_format="TCHW")(tensor)
        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "size,interpolation,max_size,antialias,video_format,repr",
        [
            (
                2,
                InterpolationMode.BILINEAR,
                None,
                True,
                "CTHW",
                "BatchVideoResize(size=2, interpolation=bilinear, max_size=None, antialias=True, video_format=CTHW)",
            ),
            (
                [2, 2],
                InterpolationMode.BICUBIC,
                224,
                True,
                "TCHW",
                "BatchVideoResize(size=[2, 2], interpolation=bicubic, max_size=224, antialias=True, video_format=TCHW)",
            ),
            (
                [2, 2],
                3,
                224,
                True,
                "TCHW",
                "BatchVideoResize(size=[2, 2], interpolation=bicubic, max_size=224, antialias=True, video_format=TCHW)",
            ),
        ],
    )
    def test_repr(
        self,
        size: int | list[int],
        interpolation: InterpolationMode,
        max_size: int | None,
        antialias: bool,
        video_format: str,
        repr: str,
    ):
        assert (
            transforms.BatchVideoResize(
                size, interpolation, max_size, antialias, video_format
            ).__repr__()
            == repr
        )

    @pytest.mark.parametrize(
        "size,error_type", [(None, TypeError), ([2, 3, 4], ValueError)]
    )
    def test_wrong_size(self, size: Any, error_type: Exception):
        with pytest.raises(error_type):
            transforms.BatchVideoResize(size)

    def test_wrong_video_format(self):
        with pytest.raises(ValueError):
            transforms.BatchVideoResize(2, video_format="ahah")

    def test_wrong_tensor(self):
        tensor = torch.rand((6, 3, 2, 3, 16, 16))
        with pytest.raises(TypeError):
            transforms.BatchVideoResize(2)(tensor)
