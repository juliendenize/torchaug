from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

import torchaug.batch_transforms.functional as F
import torchaug.transforms.functional as F_mono

from ...base_tester import BaseTesterFunctional


class TestBatchGaussianBlur(BaseTesterFunctional):
    @pytest.mark.parametrize("input_shape", [(3, 3, 4, 4)])
    @pytest.mark.parametrize(
        "kernel,sigma",
        [
            ([3, 3], torch.tensor([[1, 1], [0.5, 0.75], [2, 2]])),
            ([3, 3], torch.tensor([1, 0.5, 2])),
            ([3, 3], torch.tensor(1.0)),
            ([3, 3], 1.0),
            ([3, 3], (1.0, 1.0)),
            ([3, 3], (1, 1)),
            ([3, 3], [1.0]),
            (3, None),
        ],
    )
    def test_output_values_float(
        self,
        input_shape: list[int],
        kernel: int | list[int],
        sigma: int | list[int] | float | list[float] | Tensor | None,
    ):
        x = self.get_float_image(input_shape)

        out = F.batch_gaussian_blur(x, kernel, sigma, True)
        expected_out = torch.stack(
            [
                F_mono.gaussian_blur(
                    x[i],
                    kernel,
                    sigma[i]
                    if sigma is not None
                    and not isinstance(sigma, (tuple, list, float))
                    and sigma.ndim > 0
                    and len(sigma) == x.shape[0]
                    else sigma,
                )
                for i in range(x.shape[0])
            ]
        )

        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize("input_shape", [(3, 3, 4, 4)])
    @pytest.mark.parametrize(
        "kernel,sigma",
        [
            ([3, 3], torch.tensor([[1, 1], [0.5, 0.75], [2, 2]])),
            ([3, 3], torch.tensor([1, 0.5, 2])),
            ([3, 3], torch.tensor(1.0)),
            ([3, 3], 1.0),
            ([3, 3], (1.0, 1.0)),
            ([3, 3], [1.0]),
            (3, None),
        ],
    )
    def test_output_values_uint8(
        self,
        input_shape: list[int],
        kernel: int | list[int],
        sigma: int | list[int] | float | list[float] | Tensor | None,
    ):
        x = self.get_uint8_image(input_shape)

        out = F.batch_gaussian_blur(x, kernel, sigma, True)
        expected_out = torch.stack(
            [
                F_mono.gaussian_blur(
                    x[i],
                    kernel,
                    sigma[i]
                    if sigma is not None
                    and not isinstance(sigma, (tuple, list, float))
                    and sigma.ndim > 0
                    and len(sigma) == x.shape[0]
                    else sigma,
                )
                for i in range(x.shape[0])
            ]
        )

        torch.testing.assert_close(out, expected_out)

    def test_cuda_transfer(self):
        if torch.cuda.is_available():
            x = torch.randn((3, 3, 2, 2), device="cuda")
            F.batch_gaussian_blur(x, [3, 3], torch.tensor(1, device="cpu"))

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
        x = torch.randint(0, 256, (3, 3, 2, 2), dtype=torch.uint8)
        with pytest.raises(error_type):
            F.batch_gaussian_blur(x, kernel, None)

    @pytest.mark.parametrize(
        "sigma,error_type",
        [
            (0, ValueError),
            (-1, ValueError),
            (-1.0, ValueError),
            (torch.tensor([0.5, -1, 0.25]), ValueError),
            (torch.tensor([0.5, 0.25, 0.25, 0.25]), ValueError),
            (torch.tensor([[0.5, 0.5, 0.5]]), ValueError),
            (torch.tensor(-1), ValueError),
            (torch.tensor([[[-1]]]), ValueError),
            ([0.5, 0.5, 0.5], ValueError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_sigma(self, sigma: Any, error_type: Exception):
        x = torch.randint(0, 256, (3, 3, 2, 2), dtype=torch.uint8)
        with pytest.raises(error_type):
            F.batch_gaussian_blur(x, [3, 3], sigma, True)

    @pytest.mark.parametrize(
        "input_tensor,error_type",
        [
            ([2, 0, 3], TypeError),
            (torch.rand((4, 12, 12)), TypeError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_input_tensor(self, input_tensor: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.batch_gaussian_blur(input_tensor, [3, 3], None)


class TestBatchMixup(BaseTesterFunctional):
    @pytest.mark.parametrize("input_shape", [(3, 3, 2, 2)])
    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.parametrize(
        "lam,stack_lam",
        [
            (0.5, [0.5, 0.5, 0.5]),
            (torch.tensor(0.5), [0.5, 0.5, 0.5]),
            (torch.tensor([0.5]), [0.5, 0.5, 0.5]),
            (torch.tensor([[0.5]]), [0.5, 0.5, 0.5]),
            (torch.tensor([0.25, 0.75, 1.0]), [0.25, 0.75, 1.0]),
            (torch.tensor([[0.25], [0.75], [1.0]]), [0.25, 0.75, 1.0]),
        ],
    )
    def test_output_values_float(
        self,
        input_shape: list[int],
        inplace: bool,
        lam: float | Tensor,
        stack_lam: list[float],
    ):
        x1 = self.get_float_image(input_shape)
        x2 = self.get_float_image(input_shape)
        in_x1 = x1.clone() if inplace else x1
        in_x2 = x2.clone() if inplace else x2

        stack_lam = torch.tensor(stack_lam).view(-1, 1, 1, 1)

        out = F.batch_mixup(in_x1, in_x2, lam, inplace)
        expected_out = stack_lam * x1 + (1 - stack_lam) * x2

        torch.testing.assert_close(out, expected_out)

        if inplace:
            assert torch.equal(out, in_x1)
        else:
            assert not torch.equal(out, in_x1)

    def test_output_values_uint8(self):
        pass

    def test_cuda_transfer(self):
        if torch.cuda.is_available():
            x_1 = torch.rand((3, 3, 2, 2), device="cuda")
            x_2 = torch.rand((3, 3, 2, 2), device="cuda")

            F.batch_mixup(x_1, x_2, torch.tensor(0.5, device="cpu"), False)

    def test_cuda_transfer(self):
        if torch.cuda.is_available():
            x_1 = torch.randn((3, 3, 2, 2), device="cuda")
            x_2 = torch.randn((3, 3, 2, 2), device="cuda")

            F.batch_mixup(x_1, x_2, torch.tensor(0.5, device="cpu"), False)

    @pytest.mark.parametrize(
        "lam,error_type",
        [
            (torch.tensor([[[0.5]], [[0.75]], [[0.5]]]), ValueError),
            (torch.tensor([0.5, 0.75]), ValueError),
            (torch.tensor([[0.5, 0.75]]), ValueError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_lam(self, lam: Any, error_type: Exception):
        x_1 = torch.randn((3, 3, 2, 2))
        x_2 = torch.randn((3, 3, 2, 2))
        with pytest.raises(error_type):
            F.batch_mixup(x_1, x_2, lam, False)

    @pytest.mark.parametrize(
        "x_1,x_2,error_type",
        [
            (
                torch.randn((3, 3, 2, 2)),
                torch.randn((3, 3, 2, 2)).to(torch.int32),
                TypeError,
            ),
            (
                torch.randn((3, 3, 2, 2)).to(torch.int32),
                torch.randn((3, 3, 2, 2)),
                TypeError,
            ),
            ("ahah", torch.randn((3, 3, 2, 2)), TypeError),
            (torch.randn((3, 3, 2, 2)), "ahah", TypeError),
        ],
    )
    def test_wrong_input_tensor(self, x_1: Any, x_2: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.batch_mixup(x_1, x_2, 0.5, False)
