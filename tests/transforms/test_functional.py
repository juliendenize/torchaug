from __future__ import annotations

from typing import Any

import pytest
import torch
import torchvision.transforms.functional as F_tv

import torchaug.transforms.functional as F


class TestDiv255:
    def test_output_values(self):
        x = torch.randn((3, 2, 2))

        out = F.div_255(x, inplace=False)
        torch.testing.assert_close(out, x / 255)

        F.div_255(x, inplace=True)
        torch.testing.assert_close(x, out)


class TestGaussianBlur:
    @pytest.mark.parametrize(
        "torchaug_kernel,torchaug_sigma,torchvision_kernel,torchvision_sigma",
        [
            ([3, 3], torch.tensor([1, 1]), [3, 3], [1, 1]),
            ([3, 3], torch.tensor([1]), [3, 3], [1, 1]),
            ([3, 3], torch.tensor(1.0), [3, 3], [1, 1]),
            ([3, 3], 1.0, [3, 3], [1, 1]),
            ([3, 3], 1, [3, 3], [1, 1]),
            ([3, 3], (1.0, 1.0), [3, 3], [1, 1]),
            ([3, 3], (1, 1), [3, 3], [1, 1]),
            ([3, 3], [1.0], [3, 3], [1, 1]),
            ([3, 3], [1], [3, 3], [1, 1]),
            ([3, 3], None, [3, 3], None),
            (3, None, [3, 3], None),
        ],
    )
    def test_output_values(
        self,
        torchaug_kernel: int | list[int],
        torchaug_sigma: int | list[int] | float | list[float] | torch.Tensor | None,
        torchvision_kernel: int | list[int],
        torchvision_sigma: int | list[int] | float | list[float] | None,
    ):
        x = torch.randn((3, 2, 2))

        torchaug_out = F.gaussian_blur(x, torchaug_kernel, torchaug_sigma, True)
        torchvision_out = F_tv.gaussian_blur(x, torchvision_kernel, torchvision_sigma)

        torch.testing.assert_close(torchaug_out, torchvision_out)

    def test_cuda_transfer(self):
        if torch.cuda.is_available():
            x = torch.randn((3, 2, 2), device="cuda")
            F.gaussian_blur(x, [3, 3], torch.tensor(1, device="cpu"))

    @pytest.mark.parametrize(
        "kernel,error_type",
        [
            (3.0, TypeError),
            ([3, 3, 3], ValueError),
            ([3, 2], ValueError),
            ([3, -1], ValueError),
        ],
    )
    def test_wrong_kernel(self, kernel: Any, error_type: Exception):
        x = torch.randint(0, 256, (3, 2, 2), dtype=torch.uint8)
        with pytest.raises(error_type):
            F.gaussian_blur(x, kernel, None)

    @pytest.mark.parametrize(
        "sigma,error_type",
        [
            (0, ValueError),
            (-1, ValueError),
            (-1.0, ValueError),
            (torch.tensor([0.5, -1]), ValueError),
            (torch.tensor([0.5, 0.75, 1.0]), ValueError),
            (torch.tensor(-1), ValueError),
            (torch.tensor([[[-1]]]), ValueError),
            ([0.5, 0.5, 0.5], ValueError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_sigma(self, sigma: Any, error_type: Exception):
        x = torch.randint(0, 256, (3, 2, 2), dtype=torch.uint8)
        with pytest.raises(error_type):
            F.gaussian_blur(x, [3, 3], sigma, True)

    @pytest.mark.parametrize(
        "input_tensor,error_type",
        [
            ([2, 0, 3], TypeError),
            (torch.rand((12, 12)), IndexError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_input_tensor(self, input_tensor: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.gaussian_blur(input_tensor, [3, 3], None)


class TestMixup:
    @pytest.mark.parametrize(
        "lam",
        [
            (0.25),
        ],
    )
    def test_output_values(self, lam: float):
        x_1 = torch.randn((3, 2, 2))
        x_2 = torch.randn((3, 2, 2))

        out = F.mixup(x_1, x_2, lam, False)

        expected_out = lam * x_1 + (1 - lam) * x_2

        torch.testing.assert_close(out, expected_out)

    def test_cuda_transfer(self):
        if torch.cuda.is_available():
            x_1 = torch.randn((3, 2, 2), device="cuda")
            x_2 = torch.randn((3, 2, 2), device="cuda")

            F.mixup(x_1, x_2, torch.tensor(0.5, device="cpu"), False)

    @pytest.mark.parametrize(
        "lam",
        [
            (0.25),
        ],
    )
    def test_inplace(self, lam: float):
        x_1 = torch.randn((3, 2, 2))
        x_2 = torch.randn((3, 2, 2))
        clone_x_1 = x_1.clone()
        clone_x_2 = x_2.clone()

        out = F.mixup(clone_x_1, clone_x_2, lam, True)

        lam = torch.tensor(lam).view(-1, 1, 1)
        expected_out = lam * x_1 + (1 - lam) * x_2

        torch.testing.assert_close(out, expected_out)
        torch.testing.assert_close(out, clone_x_1)

    def test_cuda_transfer(self):
        if torch.cuda.is_available():
            x_1 = torch.randn((3, 2, 2), device="cuda")
            x_2 = torch.randn((3, 2, 2), device="cuda")

            F.mixup(x_1, x_2, 0.5, False)

    @pytest.mark.parametrize(
        "lam,error_type",
        [
            ("ahah", TypeError),
        ],
    )
    def test_wrong_lam(self, lam: Any, error_type: Exception):
        x_1 = torch.randn((3, 2, 2))
        x_2 = torch.randn((3, 2, 2))
        with pytest.raises(error_type):
            F.mixup(x_1, x_2, lam, False)

    @pytest.mark.parametrize(
        "x_1,x_2,error_type",
        [
            (
                torch.randn((3, 2, 2)),
                torch.randn((3, 2, 2)).to(torch.int32),
                TypeError,
            ),
            (
                torch.randn((3, 2, 2)).to(torch.int32),
                torch.randn((3, 2, 2)),
                TypeError,
            ),
            ("ahah", torch.randn((3, 2, 2)), TypeError),
            (torch.randn((3, 2, 2)), "ahah", TypeError),
        ],
    )
    def test_wrong_input_tensor(self, x_1: Any, x_2: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.mixup(x_1, x_2, 0.5, False)


class TestMul255:
    def test_output_values(self):
        x = torch.randn((3, 2, 2))

        out = F.mul_255(x, inplace=False)
        torch.testing.assert_close(out, x * 255)

        F.mul_255(x, inplace=True)
        torch.testing.assert_close(x, out)


class TestNormalize:
    @pytest.mark.parametrize(
        "channels,mean,std,cast_dtype,inplace,value_check",
        [
            (1, 0.225, 0.225, None, False, False),
            (3, 0.225, 0.225, None, False, True),
            (3, [0.225], [0.225], None, False, True),
            (3, torch.tensor([0.225]), torch.tensor([0.225]), None, False, True),
            (
                3,
                torch.tensor([0.225, 0.225, 0.225]),
                torch.tensor([0.225, 0.225, 0.225]),
                None,
                False,
                True,
            ),
            (
                3,
                torch.tensor([0.225, 0.225, 0.225]),
                torch.tensor([0.225, 0.225, 0.225]),
                None,
                True,
                True,
            ),
            (
                3,
                torch.tensor([0.225, 0.225, 0.225]),
                torch.tensor([0.225, 0.225, 0.225]),
                torch.float16,
                True,
                True,
            ),
        ],
    )
    def test_output_values(
        self,
        channels: int,
        mean: float | list[float],
        std: float | list[float],
        cast_dtype: torch.dtype | None,
        inplace: bool,
        value_check: bool,
    ):
        x = torch.randn((channels, 2, 2))
        to_normalize_x = x if not inplace else x.clone()

        out = (
            F.normalize(to_normalize_x, mean, std, cast_dtype, inplace, value_check),
        )
        expected_out = (F_tv.normalize(x.to(dtype=cast_dtype), mean, std, False),)

        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "x,error_type",
        [
            (torch.randint(0, 256, (3, 2, 2)), TypeError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_input_tensor(self, x: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.normalize(x, 0.225, 0.225)

    @pytest.mark.parametrize(
        "cast_dtype,error_type",
        [
            (torch.int32, ValueError),
        ],
    )
    def test_wrong_cast_dtype(self, cast_dtype: Any, error_type: Exception):
        x = torch.randint(0, 256, (3, 2, 2))
        with pytest.raises(error_type):
            F.normalize(x, 0.225, 0.225, cast_dtype)

    @pytest.mark.parametrize(
        "std,error_type",
        [
            (0, ValueError),
            ([0, 0.225, 0.225], ValueError),
            (torch.tensor([0, 0.225, 0.225]), ValueError),
        ],
    )
    def test_wrong_std(self, std: Any, error_type: Exception):
        x = torch.randn((3, 2, 2))
        with pytest.raises(error_type):
            F.normalize(x, 0.225, std, value_check=True)


class TestSolarize:
    @pytest.mark.parametrize(
        "x,threshold,value_check",
        [
            (torch.randint(0, 256, (3, 2, 2)), 128, False),
            (torch.randint(0, 256, (3, 2, 2)), torch.tensor(128), True),
            (torch.randn((3, 2, 2)), 0.5, False),
            (torch.randn((3, 2, 2)), torch.tensor(0.5), True),
        ],
    )
    def test_output_values(
        self, x: torch.Tensor, threshold: int | float | torch.Tensor, value_check: bool
    ):
        torch.testing.assert_close(
            F.solarize(x, threshold, value_check), F_tv.solarize(x, float(threshold))
        )

    @pytest.mark.parametrize(
        "x,error_type",
        [
            (torch.randint(0, 256, (2, 2)), TypeError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_input_tensor(self, x: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.solarize(x, 0.5)

    @pytest.mark.parametrize(
        "x,threshold,error_type",
        [
            (torch.randint(0, 256, (3, 2, 2), dtype=torch.uint8), 256, ValueError),
            (torch.randn((3, 2, 2)), 1.5, ValueError),
            (torch.randn((3, 2, 2)), "ahah", TypeError),
        ],
    )
    def test_wrong_threshold(
        self, x: Any, threshold: int | float | torch.Tensor, error_type: Exception
    ):
        with pytest.raises(error_type):
            F.solarize(x, threshold)
