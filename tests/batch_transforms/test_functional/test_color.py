from __future__ import annotations

import sys
from typing import Any

import pytest
import torch
import torchvision.transforms.v2.functional as F_tv
from torch import Tensor

import torchaug.batch_transforms.functional as F

from ...base_tester import BaseTesterFunctional


class TestBatchAdjustBrightness(BaseTesterFunctional):
    @pytest.mark.parametrize("input_shape", [(3, 3, 2, 2), (3, 1, 2, 2)])
    @pytest.mark.parametrize(
        "brightness_factor,brightness_torchvision_factor",
        [
            (torch.tensor((1, 0.5, 2)), (1, 0.5, 2)),
            (torch.tensor(0.5), (0.5, 0.5, 0.5)),
            (torch.tensor([0.5]), (0.5, 0.5, 0.5)),
            (0.5, (0.5, 0.5, 0.5)),
        ],
    )
    def test_output_values_float(
        self,
        input_shape: list[int],
        brightness_factor: float | Tensor,
        brightness_torchvision_factor: tuple[float],
    ):
        x = self.get_float_image(input_shape)

        torchaug_out = F.batch_adjust_brightness(x, brightness_factor, True)
        torchvision_out = torch.stack(
            [
                F_tv.adjust_brightness(x[i], t_f)
                for i, t_f in enumerate(brightness_torchvision_factor)
            ]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize("input_shape", [(3, 3, 2, 2), (3, 1, 2, 2)])
    @pytest.mark.parametrize(
        "brightness_factor,brightness_torchvision_factor",
        [
            (torch.tensor((1, 0.5, 2)), (1.0, 0.5, 2.0)),
            (torch.tensor(0.5), (0.5, 0.5, 0.5)),
            (torch.tensor([0.5]), (0.5, 0.5, 0.5)),
            (0.5, (0.5, 0.5, 0.5)),
        ],
    )
    def test_output_values_uint8(
        self,
        input_shape: list[int],
        brightness_factor: float | Tensor,
        brightness_torchvision_factor: tuple[float],
    ):
        x = self.get_uint8_image(input_shape)

        torchaug_out = F.batch_adjust_brightness(x, brightness_factor, True)
        torchvision_out = torch.stack(
            [
                F_tv.adjust_brightness(x[i], t_f)
                for i, t_f in enumerate(brightness_torchvision_factor)
            ]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        x = self.get_uint8_image((4, 3, 2, 2))
        compiled_fn = torch.compile(F.batch_adjust_brightness)

        compiled_fn(x, 0.5, False)

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


class TestBatchAdjustContrast(BaseTesterFunctional):
    @pytest.mark.parametrize("input_shape", [(3, 3, 2, 2), (3, 1, 2, 2)])
    @pytest.mark.parametrize(
        "contrast_factor,contrast_torchvision_factor",
        [
            (torch.tensor((1, 0.5, 2)), (1, 0.5, 2)),
            (torch.tensor(0.5), (0.5, 0.5, 0.5)),
            (torch.tensor([0.5]), (0.5, 0.5, 0.5)),
            (0.5, (0.5, 0.5, 0.5)),
        ],
    )
    def test_output_values_float(
        self,
        input_shape: list[int],
        contrast_factor: float | Tensor,
        contrast_torchvision_factor: tuple[float],
    ):
        x = self.get_float_image(input_shape)

        torchaug_out = F.batch_adjust_contrast(x, contrast_factor, True)
        torchvision_out = torch.stack(
            [
                F_tv.adjust_contrast(x[i], t_f)
                for i, t_f in enumerate(contrast_torchvision_factor)
            ]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize("input_shape", [(3, 3, 2, 2), (3, 1, 2, 2)])
    @pytest.mark.parametrize(
        "contrast_factor,contrast_torchvision_factor",
        [
            (torch.tensor((1, 0.5, 2)), (1, 0.5, 2)),
            (torch.tensor(0.5), (0.5, 0.5, 0.5)),
            (torch.tensor([0.5]), (0.5, 0.5, 0.5)),
            (0.5, (0.5, 0.5, 0.5)),
        ],
    )
    def test_output_values_uint8(
        self,
        input_shape: list[int],
        contrast_factor: float | Tensor,
        contrast_torchvision_factor: tuple[float],
    ):
        x = self.get_uint8_image(input_shape)

        torchaug_out = F.batch_adjust_contrast(x, contrast_factor, True)
        torchvision_out = torch.stack(
            [
                F_tv.adjust_contrast(x[i], t_f)
                for i, t_f in enumerate(contrast_torchvision_factor)
            ]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        x = self.get_uint8_image((4, 3, 2, 2))
        compiled_fn = torch.compile(F.batch_adjust_contrast)

        compiled_fn(x, 0.5, False)

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


class TestBatchAdjustHue(BaseTesterFunctional):
    @pytest.mark.parametrize("input_shape", [(3, 3, 2, 2), (3, 1, 2, 2)])
    @pytest.mark.parametrize(
        "hue_factor,hue_torchvision_factor",
        [
            (torch.tensor((0, 0.25, -0.25)), (0, 0.25, -0.25)),
            (torch.tensor(0.5), (0.5, 0.5, 0.5)),
            (torch.tensor([0.5]), (0.5, 0.5, 0.5)),
            (0.5, (0.5, 0.5, 0.5)),
        ],
    )
    def test_output_values_float(
        self,
        input_shape: list[int],
        hue_factor: float | Tensor,
        hue_torchvision_factor: tuple[float],
    ):
        x = self.get_float_image(input_shape)

        torchaug_out = F.batch_adjust_hue(x, hue_factor, True)
        torchvision_out = torch.stack(
            [F_tv.adjust_hue(x[i], t_f) for i, t_f in enumerate(hue_torchvision_factor)]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize("input_shape", [(3, 3, 2, 2), (3, 1, 2, 2)])
    @pytest.mark.parametrize(
        "hue_factor,hue_torchvision_factor",
        [
            (torch.tensor((0, 0.25, -0.25)), (0, 0.25, -0.25)),
            (torch.tensor(0.5), (0.5, 0.5, 0.5)),
            (torch.tensor([0.5]), (0.5, 0.5, 0.5)),
            (0.5, (0.5, 0.5, 0.5)),
        ],
    )
    def test_output_values_uint8(
        self,
        input_shape: list[int],
        hue_factor: float | Tensor,
        hue_torchvision_factor: tuple[float],
    ):
        x = self.get_uint8_image(input_shape)

        torchaug_out = F.batch_adjust_hue(x, hue_factor, True)
        torchvision_out = torch.stack(
            [F_tv.adjust_hue(x[i], t_f) for i, t_f in enumerate(hue_torchvision_factor)]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        x = self.get_uint8_image((4, 3, 2, 2))
        compiled_fn = torch.compile(F.batch_adjust_hue)

        compiled_fn(x, 0.5, False)

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


class TestBatchAdjustSaturation(BaseTesterFunctional):
    @pytest.mark.parametrize("input_shape", [(3, 3, 2, 2), (3, 1, 2, 2)])
    @pytest.mark.parametrize(
        "saturation_factor,saturation_torchvision_factor",
        [
            (torch.tensor((1, 0.5, 2)), (1, 0.5, 2)),
            (torch.tensor(0.5), (0.5, 0.5, 0.5)),
            (torch.tensor([0.5]), (0.5, 0.5, 0.5)),
            (0.5, (0.5, 0.5, 0.5)),
        ],
    )
    def test_output_values_float(
        self,
        input_shape: list[int],
        saturation_factor: float | Tensor,
        saturation_torchvision_factor: tuple[float],
    ):
        x = self.get_float_image(input_shape)

        torchaug_out = F.batch_adjust_saturation(x, saturation_factor, True)
        torchvision_out = torch.stack(
            [
                F_tv.adjust_saturation(x[i], t_f)
                for i, t_f in enumerate(saturation_torchvision_factor)
            ]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.parametrize("input_shape", [(3, 3, 2, 2), (3, 1, 2, 2)])
    @pytest.mark.parametrize(
        "saturation_factor,saturation_torchvision_factor",
        [
            (torch.tensor((1, 0.5, 2)), (1, 0.5, 2)),
            (torch.tensor(0.5), (0.5, 0.5, 0.5)),
            (torch.tensor([0.5]), (0.5, 0.5, 0.5)),
            (0.5, (0.5, 0.5, 0.5)),
        ],
    )
    def test_output_values_uint8(
        self,
        input_shape: list[int],
        saturation_factor: float | Tensor,
        saturation_torchvision_factor: tuple[float],
    ):
        x = self.get_uint8_image(input_shape)

        torchaug_out = F.batch_adjust_saturation(x, saturation_factor, True)
        torchvision_out = torch.stack(
            [
                F_tv.adjust_saturation(x[i], t_f)
                for i, t_f in enumerate(saturation_torchvision_factor)
            ]
        )

        torch.testing.assert_close(torchaug_out, torchvision_out)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        x = self.get_uint8_image((4, 3, 2, 2))
        compiled_fn = torch.compile(F.batch_adjust_saturation)

        compiled_fn(x, 0.5, False)

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
