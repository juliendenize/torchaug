from __future__ import annotations

import sys
from typing import Any

import pytest
import torch
import torchvision.transforms.v2.functional as F_tv
from torch import Tensor

import torchaug.transforms.functional as F

from ...base_tester import BaseTesterFunctional


class TestDiv255(BaseTesterFunctional):
    @pytest.mark.parametrize("input_shape", [(3, 2, 2), (4, 2, 2, 2)])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_output_values_float(self, input_shape, inplace):
        x = self.get_float_image(input_shape)
        in_x = x.clone() if inplace else x

        out = F.div_255(in_x, inplace=inplace)
        torch.testing.assert_close(out, x / 255)

        if inplace:
            assert torch.equal(out, in_x)
        else:
            assert not torch.equal(out, in_x)

    def test_output_values_uint8(self):
        pass

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_float_image((3, 2, 2))
        compiled_fn = torch.compile(F.div_255)

        compiled_fn(x)

    @pytest.mark.parametrize(
        "input_tensor,error_type",
        [
            ("ahah", TypeError),
        ],
    )
    def test_wrong_input_tensor(self, input_tensor: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.div_255(input_tensor)


class TestGaussianBlur(BaseTesterFunctional):
    @pytest.mark.parametrize("input_shape", [(3, 4, 4), (2, 3, 4, 4)])
    @pytest.mark.parametrize(
        "torchaug_kernel,torchaug_sigma,torchvision_kernel,torchvision_sigma, value_check",
        [
            ([3, 3], torch.tensor([1, 1]), [3, 3], [1, 1], True),
            ([3, 3], torch.tensor([1]), [3, 3], [1, 1], False),
            (3, torch.tensor(1.0), [3, 3], [1, 1], False),
            ([3, 3], 1.0, [3, 3], [1, 1], False),
            ([3, 3], (1.0, 1.0), [3, 3], [1, 1], False),
            ([3, 3], [1.0], [3, 3], [1, 1], False),
            ([3, 3], None, [3, 3], None, False),
        ],
    )
    def test_output_values_float(
        self,
        input_shape: list[int],
        torchaug_kernel: int | list[int],
        torchaug_sigma: int | list[int] | float | list[float] | Tensor | None,
        torchvision_kernel: int | list[int],
        torchvision_sigma: int | list[int] | float | list[float] | None,
        value_check: bool,
    ):
        x = self.get_float_image(input_shape)

        torchaug_out = F.gaussian_blur(x, torchaug_kernel, torchaug_sigma, value_check)
        torchvision_out = F_tv.gaussian_blur(x, torchvision_kernel, torchvision_sigma)

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize("input_shape", [(3, 4, 4), (2, 3, 4, 4)])
    @pytest.mark.parametrize(
        "torchaug_kernel,torchaug_sigma,torchvision_kernel,torchvision_sigma, value_check",
        [
            ([3, 3], torch.tensor([1, 1]), [3, 3], [1, 1], True),
            ([3, 3], torch.tensor([1]), [3, 3], [1, 1], False),
            (3, torch.tensor(1.0), [3, 3], [1, 1], False),
            ([3, 3], 1.0, [3, 3], [1, 1], False),
            ([3, 3], (1.0, 1.0), [3, 3], [1, 1], False),
            ([3, 3], [1.0], [3, 3], [1, 1], False),
            ([3, 3], None, [3, 3], None, False),
        ],
    )
    def test_output_values_uint8(
        self,
        input_shape: list[int],
        torchaug_kernel: int | list[int],
        torchaug_sigma: int | list[int] | float | list[float] | Tensor | None,
        torchvision_kernel: int | list[int],
        torchvision_sigma: int | list[int] | float | list[float] | None,
        value_check: bool,
    ):
        x = self.get_uint8_image(input_shape)

        torchaug_out = F.gaussian_blur(x, torchaug_kernel, torchaug_sigma, value_check)
        torchvision_out = F_tv.gaussian_blur(x, torchvision_kernel, torchvision_sigma)

        torch.testing.assert_close(torchaug_out, torchvision_out)

    def test_cuda_transfer(self):
        if torch.cuda.is_available():
            x = torch.randn((3, 2, 2), device="cuda")
            F.gaussian_blur(x, [3, 3], torch.tensor(1, device="cpu"))

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_uint8_image((3, 2, 2))
        compiled_fn = torch.compile(F.gaussian_blur)

        compiled_fn(x, [3, 3])

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
            (torch.rand((12, 12)), TypeError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_input_tensor(self, input_tensor: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.gaussian_blur(input_tensor, [3, 3], None)


class TestMixup(BaseTesterFunctional):
    @pytest.mark.parametrize("input_shape", [(3, 4, 4)])
    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_output_values_float(self, input_shape, lam, inplace):
        x1 = self.get_float_image(input_shape)
        x2 = self.get_float_image(input_shape)

        in_x1 = x1.clone() if inplace else x1
        in_x2 = x2.clone() if inplace else x2

        out = F.mixup(in_x1, in_x2, lam, inplace)
        expected_out = lam * x1 + (1 - lam) * x2

        torch.testing.assert_close(out, expected_out)

        if inplace or lam == 1.0:
            assert torch.equal(out, in_x1)
        else:
            assert not torch.equal(out, in_x1)
        return

    def test_output_values_uint8(self):
        pass

    def test_cuda_transfer(self):
        if torch.cuda.is_available():
            x_1 = torch.rand((3, 2, 2), device="cuda")
            x_2 = torch.rand((3, 2, 2), device="cuda")

            F.mixup(x_1, x_2, 0.5, False)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_float_image((3, 2, 2))
        compiled_fn = torch.compile(F.mixup)

        compiled_fn(x, x, 0.5)

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


class TestMul255(BaseTesterFunctional):
    @pytest.mark.parametrize("input_shape", [(3, 2, 2), (4, 2, 2, 2)])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_output_values_float(self, input_shape, inplace):
        x = self.get_float_image(input_shape)
        in_x = x.clone() if inplace else x

        out = F.mul_255(in_x, inplace=inplace)
        torch.testing.assert_close(out, x * 255)

        if inplace:
            assert torch.equal(out, in_x)
        else:
            assert not torch.equal(out, in_x)

    def test_output_values_uint8(self):
        pass

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_float_image((3, 2, 2))
        compiled_fn = torch.compile(F.mul_255)

        compiled_fn(x)

    @pytest.mark.parametrize(
        "input_tensor,error_type",
        [
            ("ahah", TypeError),
        ],
    )
    def test_wrong_input_tensor(self, input_tensor: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.mul_255(input_tensor)


class TestNormalize(BaseTesterFunctional):
    @pytest.mark.parametrize(
        "input_shape,mean,std",
        [
            ((3, 4, 4), 0.25, 0.25),
            ((3, 4, 4), [0.25], [0.25]),
            ((3, 4, 4), [0.25, 0.25, 0.25], [0.25, 0.25, 0.25]),
            ((1, 4, 4), [0.25], [0.25]),
            ((1, 4, 4), 0.25, 0.25),
        ],
    )
    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.parametrize("value_check", [True])
    def test_output_values_float(
        self,
        input_shape: list[int],
        mean: float | list[float],
        std: float | list[float],
        inplace: bool,
        value_check: bool,
    ):
        x = self.get_float_image(input_shape)
        in_x = x.clone() if inplace else x

        out = F.normalize(in_x, mean, std, None, inplace, value_check)
        expected_out = F_tv.normalize(x, mean, std, False)
        torch.testing.assert_close(out, expected_out)

        if inplace and out.dtype == in_x.dtype:
            assert torch.equal(out, in_x)
        else:
            assert not torch.equal(out, in_x)

    @pytest.mark.parametrize(
        "input_shape,mean,std",
        [
            ((3, 4, 4), 0.25, 0.25),
            ((3, 4, 4), [0.25], [0.25]),
            ((3, 4, 4), [0.25, 0.25, 0.25], [0.25, 0.25, 0.25]),
            ((1, 4, 4), [0.25], [0.25]),
            ((1, 4, 4), 0.25, 0.25),
        ],
    )
    @pytest.mark.parametrize("cast_dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.parametrize("value_check", [True])
    def test_output_values_uint8(
        self,
        input_shape: list[int],
        mean: float | list[float],
        std: float | list[float],
        cast_dtype: torch.dtype | None,
        inplace: bool,
        value_check: bool,
    ):
        x = self.get_uint8_image(input_shape)
        in_x = x.clone() if inplace else x

        out = F.normalize(in_x, mean, std, cast_dtype, inplace, value_check)
        expected_out = F_tv.normalize((x / 255).to(dtype=cast_dtype), mean, std, False)
        torch.testing.assert_close(out, expected_out)

        assert not torch.equal(out, in_x)  # cast_dtype prevent inplace.

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_float_image((3, 2, 2))
        compiled_fn = torch.compile(F.normalize)

        compiled_fn(x, 0.5, 0.5)

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
