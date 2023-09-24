from __future__ import annotations

import re
import sys
from typing import Any

import pytest
import torch

import torchaug.batch_transforms as transforms
import torchaug.batch_transforms.functional as F_b
import torchaug.transforms as mono_transforms

from ...base_tester import BaseTesterTransform


class TestBatchMixUp(BaseTesterTransform):
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("use_labels", [True, False])
    def test_functional_float(self, inplace: bool, use_labels: bool):
        torch.manual_seed(28)

        imgs = self.get_float_image((4, 3, 8, 8))
        labels = torch.randint(0, 5, (4, 1)).to(torch.float)
        inpt_imgs = imgs.clone()
        inpt_labels = labels.clone()

        out_imgs, out_labels, out_lam = transforms.BatchMixUp(0.5, inplace=inplace)(
            inpt_imgs, inpt_labels if use_labels else None
        )

        expected_lam = torch.tensor(
            [[0.8844036], [0.9952562], [0.0072862], [0.0252598]]
        )

        expected_out_labels = F_b.batch_mixup(
            labels, labels.roll(1, 0), expected_lam, False
        )
        expected_out_imgs = F_b.batch_mixup(imgs, imgs.roll(1, 0), expected_lam, False)

        torch.testing.assert_close(out_lam, expected_lam.view(4, 1))

        if use_labels:
            torch.testing.assert_close(out_labels, expected_out_labels)
        else:
            assert out_labels is None
        torch.testing.assert_close(out_imgs, expected_out_imgs)

        if inplace:
            assert torch.equal(out_imgs, inpt_imgs)
            if use_labels:
                assert torch.equal(out_labels, inpt_labels)
        else:
            assert not torch.equal(out_imgs, inpt_imgs)
            if use_labels:
                assert not torch.equal(out_labels, inpt_labels)

    def test_functional_uint8(self):
        pass

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_float_image((4, 3, 2, 2))
        compiled_fn = torch.compile(transforms.BatchMixUp(0.5))

        compiled_fn(x)

    def test_repr(self):
        assert (
            transforms.BatchMixUp(0.5, True).__repr__()
            == "BatchMixUp(alpha=0.5, inplace=True)"
        )
        assert (
            transforms.BatchMixUp(0.5, False).__repr__()
            == "BatchMixUp(alpha=0.5, inplace=False)"
        )


class TestBatchRandomApply(BaseTesterTransform):
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
    def test_functional_float(
        self,
        p: float,
        apply_transforms: torch.nn.Module | list[torch.nn.Module],
        inplace: bool,
        seed: int,
    ):
        torch.manual_seed(seed)
        imgs = self.get_float_image((4, 3, 8, 8))
        transforms.BatchRandomApply(apply_transforms, p, inplace)(imgs)

    @pytest.mark.parametrize(
        "p,apply_transforms,inplace,seed",
        [
            (
                0,
                [
                    mono_transforms.RandomSolarize(128, 0.5),
                    transforms.BatchRandomGaussianBlur((3, 3), (0.1, 2.0)),
                ],
                False,
                28,
            ),
            (
                0.5,
                [
                    mono_transforms.RandomSolarize(128, 0.5),
                    transforms.BatchRandomGaussianBlur((3, 3), (0.1, 2.0)),
                ],
                False,
                28,
            ),
            (
                1.0,
                [
                    mono_transforms.RandomSolarize(128, 0.5),
                    transforms.BatchRandomGaussianBlur((3, 3), (0.1, 2.0)),
                ],
                False,
                28,
            ),
            (1.0, mono_transforms.RandomSolarize(128, 0.5), False, 28),
            (1.0, mono_transforms.RandomSolarize(128, 0.5), True, 28),
            (0.1, mono_transforms.RandomSolarize(128, 0.5), True, 28),
            (0.3, mono_transforms.RandomSolarize(128, 0.5), True, 28),
        ],
    )
    def test_functional_uint8(
        self,
        p: float,
        apply_transforms: torch.nn.Module | list[torch.nn.Module],
        inplace: bool,
        seed: int,
    ):
        torch.manual_seed(seed)
        imgs = self.get_uint8_image((4, 3, 8, 8))
        transforms.BatchRandomApply(apply_transforms, p, inplace)(imgs)

    @pytest.mark.parametrize("p", [0, 0.5, 1.0])
    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self, p: float):
        torch.manual_seed(28)
        x = self.get_uint8_image((4, 3, 2, 2))
        compiled_fn = torch.compile(
            transforms.BatchRandomApply(mono_transforms.RandomSolarize(128, 0.5), p=p)
        )

        compiled_fn(x)

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


class TestBatchRandomGaussianBlur(BaseTesterTransform):
    def test_output_values_float(self):
        torch.manual_seed(28)

        imgs = self.get_float_image((4, 3, 8, 8))

        indices_to_apply = torch.tensor([2, 1])
        sigmas = torch.tensor([[0.7487103, 0.7487103], [0.8353596, 0.8353596]])

        expected_out = imgs.clone()
        expected_out[indices_to_apply] = F_b.batch_gaussian_blur(
            imgs[indices_to_apply], [3, 3], sigmas
        )

        out = transforms.BatchRandomGaussianBlur([3, 3], [0.1, 2.0], 0.5)(imgs)

        torch.testing.assert_close(out, expected_out)

    def test_output_values_uint8(self):
        torch.manual_seed(28)

        imgs = self.get_uint8_image((4, 3, 8, 8))

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
    def test_functional_float(
        self,
        kernel_size: int | tuple[int, int],
        sigma: float | tuple[float, float],
        p: float,
        inplace: bool,
        value_check: bool,
    ):
        imgs = self.get_float_image((4, 3, 8, 8))
        transforms.BatchRandomGaussianBlur(kernel_size, sigma, p, inplace, value_check)(
            imgs
        )

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
    def test_functional_uint8(
        self,
        kernel_size: int | tuple[int, int],
        sigma: float | tuple[float, float],
        p: float,
        inplace: bool,
        value_check: bool,
    ):
        imgs = self.get_uint8_image((4, 3, 8, 8))
        transforms.BatchRandomGaussianBlur(kernel_size, sigma, p, inplace, value_check)(
            imgs
        )

    @pytest.mark.parametrize("p", [0, 0.5, 1.0])
    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self, p: float):
        torch.manual_seed(28)
        x = self.get_uint8_image((4, 3, 4, 4))
        compiled_fn = torch.compile(transforms.BatchRandomGaussianBlur([3, 3], p=p))

        compiled_fn(x)

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
