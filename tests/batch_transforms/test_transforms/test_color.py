from __future__ import annotations

import re
import sys
from typing import Any

import pytest
import torch
import torchvision.transforms.v2.functional as F_tv

import torchaug.batch_transforms as transforms
import torchaug.batch_transforms.functional as F_b
import torchaug.transforms.functional as F

from ...base_tester import BaseTesterTransform


class TestBatchRandomColorJitter(BaseTesterTransform):
    def test_output_values_float(self):
        torch.manual_seed(28)

        imgs = self.get_float_image((8, 3, 8, 8))
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
    def test_functional_float(
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
        imgs = self.get_float_image((8, 3, 8, 8))
        out = transforms.BatchRandomColorJitter(
            brightness,
            contrast,
            saturation,
            hue,
            p,
            num_rand_calls,
            inplace,
            value_check,
        )(imgs)

        if inplace or p == 0 or num_rand_calls == 0:
            assert torch.equal(imgs, out)
        else:
            assert not torch.equal(imgs, out)

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
    def test_functional_uint8(
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
        imgs = self.get_uint8_image((8, 3, 8, 8))
        out = transforms.BatchRandomColorJitter(
            brightness,
            contrast,
            saturation,
            hue,
            p,
            num_rand_calls,
            inplace,
            value_check,
        )(imgs)

        if inplace or p == 0 or num_rand_calls == 0:
            assert torch.equal(imgs, out)
        else:
            assert not torch.equal(imgs, out)

    @pytest.mark.parametrize("p", [0, 0.5, 1.0])
    @pytest.mark.skip(reason="Compile error from Dynamo.")
    def test_compile(self, p: float):
        torch.manual_seed(28)
        x = self.get_uint8_image((4, 3, 2, 2))
        compiled_fn = torch.compile(
            transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, p=p)
        )

        compiled_fn(x)

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


class TestBatchRandomGrayscale(BaseTesterTransform):
    def test_output_values_float(self):
        torch.manual_seed(28)
        imgs = self.get_float_image((4, 3, 8, 8))
        indices_to_apply = torch.tensor([2, 1])

        expected_out = imgs.clone()
        expected_out[indices_to_apply] = F_tv.rgb_to_grayscale(
            imgs[indices_to_apply], 3
        )

        out = transforms.BatchRandomGrayscale(0.5)(imgs)
        torch.testing.assert_close(out, expected_out)

    def test_output_values_uint8(self):
        torch.manual_seed(28)
        imgs = self.get_uint8_image((4, 3, 8, 8))
        indices_to_apply = torch.tensor([2, 1])

        expected_out = imgs.clone()
        expected_out[indices_to_apply] = F_tv.rgb_to_grayscale(
            imgs[indices_to_apply], 1
        )

        out = transforms.BatchRandomGrayscale(0.5, 1)(imgs)
        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "p,inplace", [(0.5, False), (1.0, False), (0.0, False), (0.5, True)]
    )
    def test_functional_float(self, p: float, inplace: bool):
        imgs = self.get_float_image((4, 3, 8, 8))
        out = transforms.BatchRandomGrayscale(p, 3, inplace)(imgs)

        if inplace or p == 0:
            assert torch.equal(imgs, out)
        else:
            assert not torch.equal(imgs, out)

    @pytest.mark.parametrize(
        "p,inplace", [(0.5, False), (1.0, False), (0.0, False), (0.5, True)]
    )
    def test_functional_uint8(self, p: float, inplace: bool):
        imgs = self.get_uint8_image((4, 3, 8, 8))
        out = transforms.BatchRandomGrayscale(p, 1, inplace)(imgs)

        if inplace or p == 0:
            assert torch.equal(imgs, out)
        else:
            assert not torch.equal(imgs, out)

    @pytest.mark.parametrize("p", [0, 0.1, 0.2, 0.3, 0.5, 1.0])
    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self, p: float):
        torch.manual_seed(28)
        x = self.get_uint8_image((4, 3, 2, 2))
        compiled_fn = torch.compile(transforms.BatchRandomGrayscale(p=p))

        compiled_fn(x)

    @pytest.mark.parametrize(
        "p,num_output_channels,inplace,repr",
        [
            (
                0.5,
                3,
                False,
                "BatchRandomGrayscale(p=0.5, num_output_channels=3, inplace=False)",
            ),
            (
                1.0,
                1,
                False,
                "BatchRandomGrayscale(p=1.0, num_output_channels=1, inplace=False)",
            ),
            (
                0.0,
                3,
                False,
                "BatchRandomGrayscale(p=0.0, num_output_channels=3, inplace=False)",
            ),
            (
                0.5,
                1,
                True,
                "BatchRandomGrayscale(p=0.5, num_output_channels=1, inplace=True)",
            ),
        ],
    )
    def test_repr(self, p: float, num_output_channels: int, inplace: bool, repr: str):
        assert (
            transforms.BatchRandomGrayscale(p, num_output_channels, inplace).__repr__()
            == repr
        )

    def test_wrong_num_output_channels(self):
        with pytest.raises(ValueError):
            transforms.BatchRandomGrayscale(1, 2)


class TestBatchRandomSolarize(BaseTesterTransform):
    def test_output_values_float(self):
        torch.manual_seed(28)
        imgs = self.get_float_image((4, 3, 8, 8))
        indices_to_apply = torch.tensor([2, 1])

        expected_out = imgs.clone()
        expected_out[indices_to_apply] = F.solarize(imgs[indices_to_apply], 0.5)

        out = transforms.BatchRandomSolarize(0.5, 0.5)(imgs)
        torch.testing.assert_close(out, expected_out)

    def test_output_values_uint8(self):
        torch.manual_seed(28)
        imgs = self.get_uint8_image((4, 3, 8, 8))
        indices_to_apply = torch.tensor([2, 1])

        expected_out = imgs.clone()
        expected_out[indices_to_apply] = F.solarize(imgs[indices_to_apply], 128)

        out = transforms.BatchRandomSolarize(128, 0.5)(imgs)
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
    def test_functional_float(
        self, threshold: float, p: float, inplace: bool, value_check: bool
    ):
        imgs = self.get_float_image((4, 3, 8, 8))
        out = transforms.BatchRandomSolarize(threshold, p, inplace, value_check)(imgs)

        if inplace or p == 0:
            assert torch.equal(imgs, out)
        else:
            assert not torch.equal(imgs, out)

    @pytest.mark.parametrize(
        "threshold,p,inplace,value_check",
        [
            (0.5, 0.5, False, False),
            (0.5, 1.0, False, False),
            (0.5, 0.0, False, False),
            (0.5, 0.5, True, True),
        ],
    )
    def test_functional_uint8(
        self, threshold: float, p: float, inplace: bool, value_check: bool
    ):
        imgs = self.get_uint8_image((4, 3, 8, 8))
        out = transforms.BatchRandomSolarize(threshold, p, inplace, value_check)(imgs)

        if inplace or p == 0:
            assert torch.equal(imgs, out)
        else:
            assert not torch.equal(imgs, out)

    @pytest.mark.parametrize("p", [0, 0.5, 1.0])
    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self, p: float):
        torch.manual_seed(28)
        x = self.get_uint8_image((4, 3, 2, 2))
        compiled_fn = torch.compile(transforms.BatchRandomSolarize(0.5, p=p))

        compiled_fn(x)

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
