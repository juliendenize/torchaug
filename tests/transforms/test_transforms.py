import re
from typing import Any, Sequence

import pytest
import torch
import torchvision.transforms.functional as F_tv
from torchvision.transforms.functional import convert_image_dtype

import torchaug.transforms as transforms
import torchaug.transforms.functional as F


class TestDiv255:
    def test_output_values(self):
        tensor = torch.rand((3, 16, 16))
        expected_out = F.div_255(tensor, False)

        torch.testing.assert_close(
            transforms.Div255(inplace=False)(tensor), expected_out
        )
        transforms.Div255(inplace=True)(tensor)

        torch.testing.assert_close(tensor, expected_out)

    def test_repr(self):
        assert transforms.Div255(False).__repr__() == "Div255(inplace=False)"
        assert transforms.Div255(True).__repr__() == "Div255(inplace=True)"


class TestMixup:
    def test_output_values(self):
        torch.manual_seed(28)

        imgs = torch.randn(4, 3, 8, 8)
        labels = torch.randint(0, 5, (4, 1)).to(torch.float)
        inpt_imgs = imgs.clone()
        inpt_labels = labels.clone()

        # Test with labels
        out_imgs, out_labels, out_lam = transforms.MixUp(0.5, inplace=True)(
            inpt_imgs, inpt_labels
        )

        expected_lam = 0.8844035863876343
        expected_out_labels = expected_lam * labels + (1 - expected_lam) * labels.roll(
            1, 0
        )
        expected_out_imgs = expected_lam * imgs + (1 - expected_lam) * imgs.roll(1, 0)

        torch.testing.assert_close(torch.tensor(out_lam), torch.tensor(expected_lam))
        torch.testing.assert_close(out_labels, expected_out_labels)
        torch.testing.assert_close(out_imgs, expected_out_imgs)
        torch.testing.assert_close(out_imgs, inpt_imgs)
        torch.testing.assert_close(out_labels, inpt_labels)

        # Test without labels
        out_imgs, out_labels, out_lam = transforms.MixUp(0.5, inplace=True)(
            inpt_imgs, None
        )
        assert out_labels is None

        out_imgs, out_labels, out_lam = transforms.MixUp(0.5, inplace=False)(imgs, None)
        assert out_labels is None

        expected_lam = 0.007286167237907648
        torch.testing.assert_close(torch.tensor(out_lam), torch.tensor(expected_lam))
        expected_out_imgs = expected_lam * imgs + (1 - expected_lam) * imgs.roll(1, 0)
        torch.testing.assert_close(out_imgs, expected_out_imgs)

    @pytest.mark.parametrize(
        "alpha,inplace,repr",
        [
            (0.1, False, "MixUp(alpha=0.1, inplace=False)"),
            (0.5, True, "MixUp(alpha=0.5, inplace=True)"),
        ],
    )
    def test_repr(self, alpha: float, inplace: bool, repr: str):
        assert transforms.MixUp(alpha, inplace).__repr__() == repr


class TestMul255:
    def test_output_values(self):
        tensor = torch.rand((3, 16, 16))
        expected_out = F.mul_255(tensor, False)

        torch.testing.assert_close(
            transforms.Mul255(inplace=False)(tensor), expected_out
        )
        transforms.Mul255(inplace=True)(tensor)

        torch.testing.assert_close(tensor, expected_out)

    def test_repr(self):
        assert transforms.Mul255(False).__repr__() == "Mul255(inplace=False)"
        assert transforms.Mul255(True).__repr__() == "Mul255(inplace=True)"


class TestNormalize:
    @pytest.mark.parametrize(
        "x,mean,std,cast_dtype,inplace,value_check",
        [
            (torch.rand((3, 16, 16)), (0.5,), (0.5,), None, False, False),
            (torch.rand((3, 16, 16)), (0.5,), (0.5,), None, True, False),
            (torch.rand((3, 16, 16)), (0.5,), (0.5,), None, False, True),
            (
                torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8),
                (0.5,),
                (0.5,),
                torch.float32,
                False,
                True,
            ),
        ],
    )
    def test_output_values(
        self,
        x: torch.Tensor,
        mean: float | list[float],
        std: float | list[float],
        cast_dtype: torch.device | None,
        inplace: bool,
        value_check: bool,
    ):
        to_normalize_x = x if not inplace else x.clone()
        out = transforms.Normalize(mean, std, cast_dtype, inplace, value_check)(
            to_normalize_x
        )
        expected_out = F.normalize(x, mean, std, cast_dtype=cast_dtype, inplace=False)

        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "mean,std,cast_dtype,inplace,value_check,repr",
        [
            (
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                None,
                False,
                True,
                "Normalize(mean=[[[0.5]], [[0.5]], [[0.5]]], std=[[[0.5]], [[0.5]], [[0.5]]], cast_dtype=None, inplace=False, value_check=True)",
            ),
            (
                0.5,
                0.5,
                torch.float32,
                False,
                True,
                "Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=torch.float32, inplace=False, value_check=True)",
            ),
        ],
    )
    def test_repr(
        self,
        mean: float | list[float],
        std: float | list[float],
        cast_dtype: torch.device | None,
        inplace: bool,
        value_check: bool,
        repr: str,
    ):
        assert (
            transforms.Normalize(mean, std, cast_dtype, inplace, value_check).__repr__()
            == repr
        )


class TestRandomApply:
    @pytest.mark.parametrize(
        "p,apply_transforms",
        [
            (
                0,
                [
                    transforms.Normalize((0.5,), (0.5,)),
                    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0)),
                ],
            ),
            (
                0.5,
                [
                    transforms.Normalize((0.5,), (0.5,)),
                    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0)),
                ],
            ),
            (
                1.0,
                [
                    transforms.Normalize((0.5,), (0.5,)),
                    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0)),
                ],
            ),
            (1.0, transforms.Normalize((0.5,), (0.5,))),
        ],
    )
    def test_functional(
        self, p: float, apply_transforms: torch.nn.Module | list[torch.nn.Module]
    ):
        tensor = torch.rand((1, 16, 16))
        transforms.RandomApply(
            apply_transforms,
            p=p,
        )(tensor)

    @pytest.mark.parametrize(
        "p,apply_transforms,repr",
        [
            (
                1.0,
                transforms.Normalize((0.5,), (0.5,)),
                "RandomApply(\n    p=1.0,\n    Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=False, value_check=False)\n)",
            ),
        ],
    )
    def test_repr(
        self,
        apply_transforms: torch.nn.Module | list[torch.nn.Module],
        p: float,
        repr: str,
    ):
        assert transforms.RandomApply(apply_transforms, p).__repr__() == repr

    @pytest.mark.parametrize("p, error_type", [(-0.1, ValueError), (1.1, ValueError)])
    def test_wrong_p(self, p: float, error_type: Exception):
        with pytest.raises(error_type):
            transforms.RandomApply(
                [transforms.Normalize([225, 225, 225], [0.25, 0.25, 0.25])],
                p,
            )


class TestRandomColorJitter:
    def test_functional(self):
        torch.manual_seed(28)
        tensor = torch.rand((3, 16, 16))

        out_jittered = transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 1.0)(tensor)
        transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.5)(tensor)
        out_not_jittered = transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.0)(tensor)

        brightness, contrast, saturation, hue = (
            torch.tensor(0.8414264917373657),
            torch.tensor(0.9322187900543213),
            torch.tensor(0.9402793049812317),
            torch.tensor(0.1355568766593933),
        )

        torchvision_out = F_tv.adjust_brightness(
            F_tv.adjust_hue(
                F_tv.adjust_contrast(
                    F_tv.adjust_saturation(tensor, saturation), contrast
                ),
                hue,
            ),
            brightness,
        )

        torch.testing.assert_close(out_jittered, torchvision_out)
        torch.testing.assert_close(out_not_jittered, tensor)

    @pytest.mark.parametrize(
        "brightness,contrast,saturation,hue,p,repr",
        [
            (
                0.5,
                0.1,
                0.2,
                0.0,
                0.5,
                "RandomColorJitter(brightness=(0.5, 1.5), contrast=(0.9, 1.1), saturation=(0.8, 1.2), hue=None, p=0.5)",
            ),
            (
                0.0,
                0.1,
                0.2,
                0.1,
                1.0,
                "RandomColorJitter(brightness=None, contrast=(0.9, 1.1), saturation=(0.8, 1.2), hue=(-0.1, 0.1), p=1.0)",
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
        repr: str,
    ):
        assert (
            transforms.RandomColorJitter(
                brightness, contrast, saturation, hue, p
            ).__repr__()
            == repr
        )


class TestRandomGaussianBlur:
    def test_functional(self):
        torch.manual_seed(28)
        tensor = torch.rand((3, 16, 16))

        out_not_blurred = transforms.RandomGaussianBlur(3, 0.1, 0.0)(tensor)
        transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.5)(tensor)
        out_blurred = transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 1.0)(tensor)

        torchvision_out = F_tv.gaussian_blur(tensor, (3, 3), 0.762560)

        torch.testing.assert_close(out_blurred, torchvision_out)
        torch.testing.assert_close(out_not_blurred, tensor)

    @pytest.mark.parametrize(
        "kernel_size,sigma,p,value_check,repr",
        [
            (
                3,
                0.1,
                0.5,
                False,
                r"RandomGaussianBlur\(kernel_size=\(3, 3\), sigma=\[0\.\d+, 0\.\d+\], p=0\.5, value_check=False\)",
            ),
            (
                [3, 9],
                [0.1, 0.9],
                0.5,
                True,
                r"RandomGaussianBlur\(kernel_size=\[3, 9\], sigma=\[0\.\d+, 0\.\d+\], p=0\.5, value_check=True\)",
            ),
        ],
    )
    def test_repr(
        self,
        kernel_size: int | tuple[int],
        sigma: float | tuple[float],
        p: float,
        value_check: bool,
        repr: str,
    ):
        assert re.match(
            repr,
            transforms.RandomGaussianBlur(
                kernel_size, sigma, p, value_check
            ).__repr__(),
        )

    @pytest.mark.parametrize(
        "kernel_size,sigma,error_type",
        [
            ((3, 3, 3), 0.1, ValueError),
            ((3, 2), (0.1, 2.0), ValueError),
            ((3, 0), (0.1, 2.0), ValueError),
            ((3, 3), 0, ValueError),
            ((3, 3), -1, ValueError),
            ((3, 3), (2.0, 1.0), ValueError),
            ((3, 3), (0.0, 1.0), ValueError),
            ((3, 3), (0.1, 2.0, 3.0), ValueError),
            ((3, 3), "ahah", ValueError),
        ],
    )
    def test_wrong_parameters(
        self, kernel_size: Any, sigma: Any, error_type: Exception
    ):
        with pytest.raises(error_type):
            transforms.RandomGaussianBlur(kernel_size, sigma)


class TestRandomSolarize:
    def test_functional(self):
        torch.manual_seed(28)

        tensor = torch.rand((3, 16, 16))

        out_not_solarized = transforms.RandomSolarize(0.5, 0.0)(tensor)
        transforms.RandomSolarize(0.5, 0.5)(tensor)
        out_solarized = transforms.RandomSolarize(0.5, 1.0)(tensor)

        expected_out = F.solarize(tensor, 0.5)

        torch.testing.assert_close(out_solarized, expected_out)
        torch.testing.assert_close(out_not_solarized, tensor)

    def test_repr(self):
        assert re.match(
            r"RandomSolarize\(threshold=0.\d+, p=0.5, value_check=False\)",
            transforms.RandomSolarize(0.5, 0.5, False).__repr__(),
        )
        assert re.match(
            r"RandomSolarize\(threshold=0.\d+, p=0.5, value_check=True\)",
            transforms.RandomSolarize(0.5, 0.5, True).__repr__(),
        )


class TestVideoNormalize:
    @pytest.mark.parametrize(
        "x,mean,std,cast_dtype,inplace,value_check,video_format",
        [
            (torch.randn(2, 3, 224, 224), 0.5, 0.5, None, False, False, "TCHW"),
            (torch.randn(3, 2, 224, 224), 0.5, 0.5, None, False, False, "CTHW"),
            (torch.randn(2, 3, 2, 224, 224), 0.5, 0.5, None, False, False, "CTHW"),
            (torch.randn(2, 3, 2, 224, 224), 0.5, 0.5, None, False, True, "CTHW"),
            (torch.randn(3, 2, 224, 224), 0.5, 0.5, None, True, False, "CTHW"),
            (
                torch.randint(0, 256, (3, 2, 224, 224)),
                0.5,
                0.5,
                torch.float32,
                False,
                False,
                "CTHW",
            ),
        ],
    )
    def test_functional(
        self,
        x: torch.Tensor,
        mean: Sequence[float],
        std: Sequence[float],
        cast_dtype: torch.dtype | None,
        inplace: bool,
        value_check: bool,
        video_format: str,
    ):
        to_normalize_x = x.clone() if inplace else x
        out = transforms.VideoNormalize(
            mean, std, cast_dtype, inplace, value_check, video_format
        )(to_normalize_x)

        if video_format == "CTHW":
            dims = [0, 2, 1, 3, 4] if x.ndim == 5 else [1, 0, 2, 3]
            expected_x = x.permute(dims)
        else:
            expected_x = x

        if cast_dtype is not None:
            expected_x = convert_image_dtype(expected_x, cast_dtype)

        expected_out = F.normalize(expected_x, mean, std, cast_dtype, False, False)
        if video_format == "CTHW":
            expected_out = expected_out.permute(dims)

        torch.testing.assert_close(out, expected_out)
        if inplace:
            torch.testing.assert_close(to_normalize_x, expected_out)

    @pytest.mark.parametrize(
        "mean,std,cast_dtype,inplace,value_check,video_format,repr",
        [
            (
                0.5,
                0.5,
                None,
                False,
                False,
                "TCHW",
                "VideoNormalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=False, value_check=False, video_format=TCHW)",
            ),
            (
                0.5,
                0.5,
                None,
                False,
                False,
                "CTHW",
                "VideoNormalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=False, value_check=False, video_format=CTHW)",
            ),
            (
                0.5,
                0.5,
                None,
                False,
                False,
                "CTHW",
                "VideoNormalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=False, value_check=False, video_format=CTHW)",
            ),
            (
                0.5,
                0.5,
                None,
                False,
                True,
                "CTHW",
                "VideoNormalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=False, value_check=True, video_format=CTHW)",
            ),
            (
                0.5,
                0.5,
                None,
                True,
                False,
                "CTHW",
                "VideoNormalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=False, video_format=CTHW)",
            ),
            (
                0.5,
                0.5,
                torch.float32,
                False,
                False,
                "CTHW",
                "VideoNormalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=torch.float32, inplace=False, value_check=False, video_format=CTHW)",
            ),
        ],
    )
    def test_repr(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        cast_dtype: torch.dtype | None,
        inplace: bool,
        value_check: bool,
        video_format: str,
        repr: str,
    ):
        assert (
            transforms.VideoNormalize(
                mean, std, cast_dtype, inplace, value_check, video_format
            ).__repr__()
            == repr
        )

    def test_wrong_format(self):
        with pytest.raises(ValueError):
            transforms.VideoNormalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True, video_format="ahah"
            )

    def test_wrong_tensor(self):
        tensor = torch.rand((6, 3, 2, 3, 16, 16))
        with pytest.raises(TypeError):
            transforms.VideoNormalize(
                (0.5,), (0.5, 0.1, 0), inplace=True, video_format="CTHW"
            )(tensor)


class TestVideoWrapper:
    def test_functional(self):
        transform = transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        # test CTHW format
        tensor = torch.rand((3, 2, 16, 16))
        expected_out = F.normalize(tensor, (0.5,), (0.5,), inplace=False)
        out = transforms.VideoWrapper(transform=transform)(tensor)
        torch.testing.assert_close(out, expected_out)

        # test TCHW format
        expected_out = F.normalize(tensor, (0.5,), (0.5,), inplace=False)
        out = transforms.VideoWrapper(transform=transform, video_format="TCHW")(
            tensor.permute(1, 0, 2, 3)
        )
        torch.testing.assert_close(out.permute(1, 0, 2, 3), expected_out)

    def test_repr(self):
        transform = transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        assert (
            transforms.VideoWrapper(transform=transform).__repr__()
            == "VideoWrapper(\n    transform=Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=False, value_check=True),\n    video_format=CTHW\n)"
        )

    def test_wrong_video_format(self):
        transform = transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        with pytest.raises(ValueError):
            transforms.VideoWrapper(transform=transform, video_format="ahah")

    def test_wrong_tensor(self):
        transform = transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        tensor = torch.rand((6, 3, 2, 3, 16, 16))
        with pytest.raises(TypeError):
            transforms.VideoWrapper(transform=transform)(tensor)
