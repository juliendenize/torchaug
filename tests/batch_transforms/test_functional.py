from __future__ import annotations

from typing import Any

import pytest
import torch
import torchvision.transforms.functional as F_tv
from torch import Tensor

import torchaug.batch_transforms.functional as F


class TestBatchAdjustBrightness:
    @pytest.mark.parametrize(
        "brightness_factor,brightness_torchvision_factor",
        [
            (torch.tensor((1, 0.5, 2)), (1, 0.5, 2)),
            (torch.tensor(0.5), (0.5, 0.5, 0.5)),
            (torch.tensor([0.5]), (0.5, 0.5, 0.5)),
            (0.5, (0.5, 0.5, 0.5)),
        ],
    )
    def test_output_values(
        self,
        brightness_factor: float | Tensor,
        brightness_torchvision_factor: tuple[float],
    ):
        x = torch.randint(0, 256, (3, 3, 2, 2), dtype=torch.uint8)

        torchaug_out = F.batch_adjust_brightness(x, brightness_factor, True)
        torchvision_out = torch.stack(
            [
                F_tv.adjust_brightness(x[i], t_f)
                for i, t_f in enumerate(brightness_torchvision_factor)
            ]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize(
        "brightness_factor,error_type",
        [
            (-0.5, ValueError),
            (torch.tensor(-0.5), ValueError),
            (torch.tensor([1.0, -0.5, 2.0]), ValueError),
            (torch.tensor([1.0, 2.0]), ValueError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_factor(self, brightness_factor: Any, error_type: Exception):
        x = torch.randint(0, 256, (3, 3, 2, 2), dtype=torch.uint8)
        with pytest.raises(error_type):
            F.batch_adjust_brightness(x, brightness_factor, True)

    @pytest.mark.parametrize(
        "input_tensor,error_type",
        [
            ([2, 0, 3], TypeError),
            (torch.rand((4, 12, 12)), TypeError),
            (torch.rand((4, 4, 12, 12)), TypeError),
        ],
    )
    def test_wrong_input_tensor(self, input_tensor: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.batch_adjust_brightness(input_tensor, torch.tensor([1.0, 0.5, 2.0]), True)


class TestBatchAdjustContrast:
    @pytest.mark.parametrize(
        "contrast_factor,contrast_torchvision_factor",
        [
            (torch.tensor((1, 0.5, 2)), (1, 0.5, 2)),
            (torch.tensor(0.5), (0.5, 0.5, 0.5)),
            (torch.tensor([0.5]), (0.5, 0.5, 0.5)),
            (0.5, (0.5, 0.5, 0.5)),
        ],
    )
    def test_output_values(
        self,
        contrast_factor: float | Tensor,
        contrast_torchvision_factor: tuple[float],
    ):
        x = torch.randint(0, 256, (3, 3, 2, 2), dtype=torch.uint8)

        torchaug_out = F.batch_adjust_contrast(x, contrast_factor, True)
        torchvision_out = torch.stack(
            [
                F_tv.adjust_contrast(x[i], t_f)
                for i, t_f in enumerate(contrast_torchvision_factor)
            ]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize(
        "contrast_factor,contrast_torchvision_factor",
        [
            (torch.tensor((1, 0.5, 2)), (1, 0.5, 2)),
        ],
    )
    def test_output_values_one_channel(
        self,
        contrast_factor: float | Tensor,
        contrast_torchvision_factor: tuple[float],
    ):
        x = torch.randint(0, 256, (3, 1, 2, 2), dtype=torch.uint8)

        torchaug_out = F.batch_adjust_contrast(x, contrast_factor, True)
        torchvision_out = torch.stack(
            [
                F_tv.adjust_contrast(x[i], t_f)
                for i, t_f in enumerate(contrast_torchvision_factor)
            ]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize(
        "contrast_factor,error_type",
        [
            (-0.5, ValueError),
            (torch.tensor(-0.5), ValueError),
            (torch.tensor([1.0, -0.5, 2.0]), ValueError),
            (torch.tensor([1.0, 2.0]), ValueError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_factor(self, contrast_factor: Any, error_type: Exception):
        x = torch.randint(0, 256, (3, 3, 2, 2), dtype=torch.uint8)
        with pytest.raises(error_type):
            F.batch_adjust_contrast(x, contrast_factor, True)

    @pytest.mark.parametrize(
        "input_tensor,error_type",
        [
            ([2, 0, 3], TypeError),
            (torch.rand((4, 12, 12)), TypeError),
            (torch.rand((4, 4, 12, 12)), TypeError),
        ],
    )
    def test_wrong_input_tensor(self, input_tensor: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.batch_adjust_contrast(input_tensor, torch.tensor([1.0, 0.5, 2.0]), True)


class TestBatchAdjustHue:
    @pytest.mark.parametrize(
        "hue_factor,hue_torchvision_factor",
        [
            (torch.tensor((0, 0.25, -0.25)), (0, 0.25, -0.25)),
            (torch.tensor(0.5), (0.5, 0.5, 0.5)),
            (torch.tensor([0.5]), (0.5, 0.5, 0.5)),
            (0.5, (0.5, 0.5, 0.5)),
        ],
    )
    def test_output_values(
        self,
        hue_factor: float | Tensor,
        hue_torchvision_factor: tuple[float],
    ):
        x = torch.randint(0, 256, (3, 3, 2, 2), dtype=torch.uint8)

        torchaug_out = F.batch_adjust_hue(x, hue_factor, True)
        torchvision_out = torch.stack(
            [F_tv.adjust_hue(x[i], t_f) for i, t_f in enumerate(hue_torchvision_factor)]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize(
        "hue_factor,hue_torchvision_factor",
        [
            (torch.tensor((0, 0.25, -0.25)), (0, 0.25, -0.25)),
        ],
    )
    def test_output_values_one_channel(
        self,
        hue_factor: float | Tensor,
        hue_torchvision_factor: tuple[float],
    ):
        x = torch.randint(0, 256, (3, 1, 2, 2), dtype=torch.uint8)

        torchaug_out = F.batch_adjust_hue(x, hue_factor, True)
        torchvision_out = torch.stack(
            [F_tv.adjust_hue(x[i], t_f) for i, t_f in enumerate(hue_torchvision_factor)]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize(
        "hue_factor,error_type",
        [
            (-1.0, ValueError),
            (torch.tensor(-1), ValueError),
            (torch.tensor([0.25, -1.5, 0.5]), ValueError),
            (torch.tensor([0.25, 0.25]), ValueError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_factor(self, hue_factor: Any, error_type: Exception):
        x = torch.randint(0, 256, (3, 3, 2, 2), dtype=torch.uint8)
        with pytest.raises(error_type):
            F.batch_adjust_hue(x, hue_factor, True)

    @pytest.mark.parametrize(
        "input_tensor,error_type",
        [
            ([2, 0, 3], TypeError),
            (torch.rand((4, 12, 12)), TypeError),
            (torch.rand((4, 4, 12, 12)), TypeError),
        ],
    )
    def test_wrong_input_tensor(self, input_tensor: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.batch_adjust_hue(input_tensor, torch.tensor([1.0, 0.5, 2.0]), True)


class TestBatchAdjustSaturation:
    @pytest.mark.parametrize(
        "saturation_factor,saturation_torchvision_factor",
        [
            (torch.tensor((1, 0.5, 2)), (1, 0.5, 2)),
            (torch.tensor(0.5), (0.5, 0.5, 0.5)),
            (torch.tensor([0.5]), (0.5, 0.5, 0.5)),
            (0.5, (0.5, 0.5, 0.5)),
        ],
    )
    def test_output_values(
        self,
        saturation_factor: float | Tensor,
        saturation_torchvision_factor: tuple[float],
    ):
        x = torch.randint(0, 256, (3, 3, 2, 2), dtype=torch.uint8)

        torchaug_out = F.batch_adjust_saturation(x, saturation_factor, True)
        torchvision_out = torch.stack(
            [
                F_tv.adjust_saturation(x[i], t_f)
                for i, t_f in enumerate(saturation_torchvision_factor)
            ]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize(
        "saturation_factor,saturation_torchvision_factor",
        [
            (torch.tensor((1, 0.5, 2)), (1, 0.5, 2)),
        ],
    )
    def test_output_values_one_channel(
        self,
        saturation_factor: float | Tensor,
        saturation_torchvision_factor: tuple[float],
    ):
        x = torch.randint(0, 256, (3, 1, 2, 2), dtype=torch.uint8)

        torchaug_out = F.batch_adjust_saturation(x, saturation_factor, True)
        torchvision_out = torch.stack(
            [
                F_tv.adjust_saturation(x[i], t_f)
                for i, t_f in enumerate(saturation_torchvision_factor)
            ]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize(
        "saturation_factor,error_type",
        [
            (-0.5, ValueError),
            (torch.tensor(-0.5), ValueError),
            (torch.tensor([1.0, -0.5, 2.0]), ValueError),
            (torch.tensor([1.0, 2.0]), ValueError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_factor(self, saturation_factor: Any, error_type: Exception):
        x = torch.randint(0, 256, (3, 3, 2, 2), dtype=torch.uint8)
        with pytest.raises(error_type):
            F.batch_adjust_saturation(x, saturation_factor, True)

    @pytest.mark.parametrize(
        "input_tensor,error_type",
        [
            ([2, 0, 3], TypeError),
            (torch.rand((4, 12, 12)), TypeError),
            (torch.rand((4, 4, 12, 12)), TypeError),
        ],
    )
    def test_wrong_input_tensor(self, input_tensor: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.batch_adjust_saturation(input_tensor, torch.tensor([1.0, 0.5, 2.0]), True)


class TestBatchGaussianBlur:
    @pytest.mark.parametrize(
        "torchaug_kernel,torchaug_sigma,torchvision_kernel,torchvision_sigma",
        [
            (
                [3, 3],
                torch.tensor([[1, 1], [0.5, 0.75], [2, 2]]),
                [3, 3],
                [[1, 1], [0.5, 0.75], [2, 2]],
            ),
            ([3, 3], torch.tensor([1, 0.5, 2]), [3, 3], [1, 0.5, 2]),
            ([3, 3], torch.tensor(1.0), [3, 3], [1, 1, 1]),
            ([3, 3], 1.0, [3, 3], [1, 1, 1]),
            ([3, 3], 1, [3, 3], [1, 1, 1]),
            ([3, 3], (1.0, 1.0), [3, 3], [1, 1, 1]),
            ([3, 3], (1, 1), [3, 3], [1, 1, 1]),
            ([3, 3], [1.0], [3, 3], [1, 1, 1]),
            ([3, 3], [1], [3, 3], [1, 1, 1]),
            ([3, 3], None, [3, 3], [None, None, None]),
            (3, None, [3, 3], [None, None, None]),
        ],
    )
    def test_output_values(
        self,
        torchaug_kernel: int | list[int],
        torchaug_sigma: int | list[int] | float | list[float] | Tensor | None,
        torchvision_kernel: int | list[int],
        torchvision_sigma: int | list[int] | float | list[float] | None,
    ):
        x = torch.randn((3, 3, 2, 2))

        torchaug_out = F.batch_gaussian_blur(x, torchaug_kernel, torchaug_sigma, True)
        torchvision_out = torch.stack(
            [
                F_tv.gaussian_blur(x[i], torchvision_kernel, t_s)
                for i, t_s in enumerate(torchvision_sigma)
            ]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

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


class TestBatchMixup:
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
    def test_output_values(self, lam: float | Tensor, stack_lam: list[float]):
        x_1 = torch.randn((3, 3, 2, 2))
        x_2 = torch.randn((3, 3, 2, 2))
        stack_lam = torch.tensor(stack_lam).view(-1, 1, 1, 1)

        out = F.batch_mixup(x_1, x_2, lam, False)
        expected_out = stack_lam * x_1 + (1 - stack_lam) * x_2

        torch.testing.assert_close(out, expected_out)

    def test_cuda_transfer(self):
        if torch.cuda.is_available():
            x_1 = torch.randn((3, 3, 2, 2), device="cuda")
            x_2 = torch.randn((3, 3, 2, 2), device="cuda")

            F.batch_mixup(x_1, x_2, torch.tensor(0.5, device="cpu"), False)

    @pytest.mark.parametrize(
        "lam,stack_lam",
        [
            (0.5, [0.5, 0.5, 0.5]),
        ],
    )
    def test_inplace(self, lam: float | Tensor, stack_lam: list[float]):
        x_1 = torch.randn((3, 3, 2, 2))
        x_2 = torch.randn((3, 3, 2, 2))
        clone_x_1 = x_1.clone()
        clone_x_2 = x_2.clone()
        stack_lam = torch.tensor(stack_lam).view(-1, 1, 1, 1)

        out = F.batch_mixup(clone_x_1, clone_x_2, lam, True)
        expected_out = stack_lam * x_1 + (1 - stack_lam) * x_2

        torch.testing.assert_close(out, expected_out)
        torch.testing.assert_close(out, clone_x_1)

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
