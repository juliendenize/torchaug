from __future__ import annotations

import sys
from typing import Any

import pytest
import torch
import torchvision.transforms.functional as F_tv
from torch import Tensor

import torchaug.transforms.functional as F

from ...base_tester import BaseTesterFunctional


class TestAdjustHue(BaseTesterFunctional):
    @pytest.mark.parametrize("input_shape", [(3, 2, 2), (1, 2, 2), (2, 3, 2, 2)])
    @pytest.mark.parametrize(
        "hue_factor,hue_torchvision_factor",
        [
            (0, 0.0),
            (0.25, 0.25),
            (-0.25, -0.25),
        ],
    )
    def test_output_values_float(
        self,
        input_shape: list[int],
        hue_factor: float | Tensor,
        hue_torchvision_factor: tuple[float],
    ):
        x = self.get_float_image(input_shape)

        torchaug_out = F.adjust_hue(x, hue_factor)
        torchvision_out = F_tv.adjust_hue(x, hue_torchvision_factor)

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize("input_shape", [(3, 2, 2), (1, 2, 2), (2, 3, 2, 2)])
    @pytest.mark.parametrize(
        "hue_factor,hue_torchvision_factor",
        [
            (0, 0.0),
            (0.25, 0.25),
            (-0.25, -0.25),
        ],
    )
    def test_output_values_uint8(
        self,
        input_shape: list[int],
        hue_factor: float | Tensor,
        hue_torchvision_factor: tuple[float],
    ):
        x = self.get_uint8_image(input_shape)

        torchaug_out = F.adjust_hue(x, hue_factor)
        torchvision_out = F_tv.adjust_hue(x, hue_torchvision_factor)

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize(
        "hue_factor,hue_torchvision_factor",
        [
            (0.0, 0.0),
            (0.25, 0.25),
            (-0.25, -0.25),
        ],
    )
    def test_output_values_one_channel(
        self,
        hue_factor: float | Tensor,
        hue_torchvision_factor: tuple[float],
    ):
        x = torch.randint(0, 256, (1, 2, 2), dtype=torch.uint8)

        torchaug_out = F.adjust_hue(x, hue_factor)
        torchvision_out = F_tv.adjust_hue(x, hue_torchvision_factor)

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_uint8_image((3, 2, 2))
        compiled_fn = torch.compile(F.adjust_hue)

        compiled_fn(x, 0.1)

    @pytest.mark.parametrize(
        "hue_factor,error_type",
        [
            (-1.0, ValueError),
            (1.0, ValueError),
            ("ahah", TypeError),
        ],
    )
    def test_wrong_factor(self, hue_factor: Any, error_type: Exception):
        x = torch.randint(0, 256, (3, 3, 2, 2), dtype=torch.uint8)
        with pytest.raises(error_type):
            F.adjust_hue(x, hue_factor)

    @pytest.mark.parametrize(
        "input_tensor,error_type",
        [
            ([2, 0, 3], TypeError),
            (torch.rand((4, 12, 12)), TypeError),
        ],
    )
    def test_wrong_input_tensor(self, input_tensor: Any, error_type: Exception):
        with pytest.raises(error_type):
            F.adjust_hue(input_tensor, torch.tensor([1.0, 0.5, 2.0]))


class TestSolarize(BaseTesterFunctional):
    @pytest.mark.parametrize("input_shape", [(3, 2, 2), (2, 3, 2, 2)])
    @pytest.mark.parametrize(
        "threshold,value_check",
        [
            (0.5, False),
            (torch.tensor(0.5), True),
        ],
    )
    def test_output_values_float(
        self, input_shape: list[int], threshold: int | float | Tensor, value_check: bool
    ):
        x = self.get_float_image(input_shape)
        torch.testing.assert_close(
            F.solarize(x, threshold, value_check), F_tv.solarize(x, float(threshold))
        )

    @pytest.mark.parametrize("input_shape", [(3, 2, 2), (2, 3, 2, 2)])
    @pytest.mark.parametrize(
        "threshold,value_check",
        [
            (128, False),
            (torch.tensor(128), True),
        ],
    )
    def test_output_values_uint8(
        self, input_shape: list[int], threshold: int | float | Tensor, value_check: bool
    ):
        x = self.get_uint8_image(input_shape)
        torch.testing.assert_close(
            F.solarize(x, threshold, value_check), F_tv.solarize(x, float(threshold))
        )

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_uint8_image((3, 2, 2))
        compiled_fn = torch.compile(F.solarize)

        compiled_fn(x, 0.5)

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
        self, x: Any, threshold: int | float | Tensor, error_type: Exception
    ):
        with pytest.raises(error_type):
            F.solarize(x, threshold)
