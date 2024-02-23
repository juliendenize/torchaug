from __future__ import annotations

import re
import sys
from typing import Any, Sequence

import pytest
import torch

import torchaug.transforms as transforms
import torchaug.transforms.functional as F
import torchvision.transforms.v2.functional as F_tv

from ...base_tester import BaseTesterTransform


class TestDiv255(BaseTesterTransform):
    @pytest.mark.parametrize("input_shape", [(3, 2, 2)])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_functional_float(self, input_shape, inplace):
        tensor = self.get_float_image(input_shape)

        in_tensor = tensor.clone() if inplace else tensor

        out = transforms.Div255(inplace=inplace)(in_tensor)
        expected_out = F.div_255(tensor, False)

        torch.testing.assert_close(out, expected_out)

        if inplace:
            assert torch.equal(out, in_tensor)
        else:
            assert not torch.equal(out, in_tensor)

    def test_functional_uint8(self):
        pass

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_float_image((3, 2, 2))
        compiled_fn = torch.compile(transforms.Div255())

        compiled_fn(x)

    def test_repr(self):
        assert transforms.Div255(False).__repr__() == "Div255(inplace=False)"
        assert transforms.Div255(True).__repr__() == "Div255(inplace=True)"


class TestMixup(BaseTesterTransform):
    @pytest.mark.parametrize("input_shape", [(3, 2, 2)])
    @pytest.mark.parametrize("use_labels", [False, True])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_functional_float(self, input_shape, use_labels, inplace):
        torch.manual_seed(28)

        imgs = self.get_float_image(input_shape)
        labels = torch.randint(0, 5, (input_shape[0], 1)).to(torch.float)

        in_imgs = imgs.clone() if inplace else imgs
        in_labels = labels.clone() if inplace else labels

        out_imgs, out_labels, out_lam = transforms.MixUp(0.5, inplace=inplace)(
            in_imgs, in_labels if use_labels else None
        )

        expected_lam = 0.4325355887413025
        expected_out_imgs = expected_lam * imgs + (1 - expected_lam) * imgs.roll(1, 0)

        torch.testing.assert_close(torch.tensor(out_lam), torch.tensor(expected_lam))
        if use_labels:
            expected_out_labels = expected_lam * labels + (
                1 - expected_lam
            ) * labels.roll(1, 0)
            torch.testing.assert_close(out_labels, expected_out_labels)
        torch.testing.assert_close(out_imgs, expected_out_imgs)
        if inplace:
            assert torch.equal(out_imgs, in_imgs)
            if use_labels:
                assert torch.equal(out_labels, in_labels)
            else:
                assert out_labels is None
        else:
            assert not torch.equal(out_imgs, in_imgs)
            if use_labels:
                assert not torch.equal(out_labels, in_labels)
            else:
                assert out_labels is None

    def test_functional_uint8(self):
        pass

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_float_image((3, 2, 2))
        compiled_fn = torch.compile(transforms.MixUp(0.5))

        compiled_fn(x)

    @pytest.mark.parametrize(
        "alpha,inplace,repr",
        [
            (0.1, False, "MixUp(alpha=0.1, inplace=False)"),
            (0.5, True, "MixUp(alpha=0.5, inplace=True)"),
        ],
    )
    def test_repr(self, alpha: float, inplace: bool, repr: str):
        assert transforms.MixUp(alpha, inplace).__repr__() == repr


class TestMul255(BaseTesterTransform):
    @pytest.mark.parametrize("input_shape", [(3, 2, 2)])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_functional_float(self, input_shape, inplace):
        tensor = self.get_float_image(input_shape)

        in_tensor = tensor.clone() if inplace else tensor

        out = transforms.Mul255(inplace=inplace)(in_tensor)
        expected_out = F.mul_255(tensor, False)

        torch.testing.assert_close(out, expected_out)

        if inplace:
            assert torch.equal(out, in_tensor)
        else:
            assert not torch.equal(out, in_tensor)

    def test_functional_uint8(self):
        pass

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_float_image((3, 2, 2))
        compiled_fn = torch.compile(transforms.Mul255())

        compiled_fn(x)

    def test_repr(self):
        assert transforms.Mul255(False).__repr__() == "Mul255(inplace=False)"
        assert transforms.Mul255(True).__repr__() == "Mul255(inplace=True)"


class TestNormalize(BaseTesterTransform):
    @pytest.mark.parametrize("input_shape", [(3, 2, 2)])
    @pytest.mark.parametrize(
        "mean,std", [((0.5), 0.5), ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    @pytest.mark.parametrize("cast_dtype", [torch.float32])
    @pytest.mark.parametrize("inplace,value_check", [(False, False), (True, True)])
    def test_functional_float(
        self, input_shape, mean, std, cast_dtype, inplace, value_check
    ):
        x = self.get_float_image(input_shape)
        in_x = x.clone() if inplace else x
        out = transforms.Normalize(mean, std, cast_dtype, inplace, value_check)(in_x)
        expected_out = F.normalize(x, mean, std, cast_dtype, False, value_check)

        torch.testing.assert_close(out, expected_out)

        if inplace:
            assert torch.equal(out, in_x)
        else:
            assert not torch.equal(out, in_x)

    @pytest.mark.parametrize("input_shape", [(3, 2, 2)])
    @pytest.mark.parametrize(
        "mean,std", [((0.5), 0.5), ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    @pytest.mark.parametrize("cast_dtype", [torch.float32])
    @pytest.mark.parametrize("inplace,value_check", [(False, False), (True, True)])
    def test_functional_uint8(
        self, input_shape, mean, std, cast_dtype, inplace, value_check
    ):
        x = self.get_uint8_image(input_shape)
        in_x = x.clone() if inplace else x
        out = transforms.Normalize(mean, std, cast_dtype, inplace, value_check)(in_x)
        expected_out = F.normalize(x, mean, std, cast_dtype, False, value_check)

        torch.testing.assert_close(out, expected_out)

        assert not torch.equal(out, in_x)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_float_image((3, 2, 2))
        compiled_fn = torch.compile(transforms.Normalize(0.5, 0.5))

        compiled_fn(x)

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


class TestRandomApply(BaseTesterTransform):
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
    def test_functional_float(
        self, p: float, apply_transforms: torch.nn.Module | list[torch.nn.Module]
    ):
        tensor = self.get_float_image((1, 16, 16))
        transforms.RandomApply(
            apply_transforms,
            p=p,
        )(tensor)

    @pytest.mark.parametrize(
        "p,apply_transforms",
        [
            (
                0,
                [
                    transforms.Normalize((0.5,), (0.5,), cast_dtype=torch.float32),
                    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0)),
                ],
            ),
            (
                0.5,
                [
                    transforms.Normalize((0.5,), (0.5,), cast_dtype=torch.float32),
                    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0)),
                ],
            ),
            (
                1.0,
                [
                    transforms.Normalize((0.5,), (0.5,), cast_dtype=torch.float32),
                    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0)),
                ],
            ),
            (1.0, transforms.Normalize((0.5,), (0.5,), cast_dtype=torch.float32)),
        ],
    )
    def test_functional_uint8(
        self, p: float, apply_transforms: torch.nn.Module | list[torch.nn.Module]
    ):
        tensor = self.get_uint8_image((1, 16, 16))
        transforms.RandomApply(
            apply_transforms,
            p=p,
        )(tensor)

    @pytest.mark.parametrize("p", [0, 0.5, 1.0])
    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self, p: float):
        torch.manual_seed(28)

        x = self.get_uint8_image((3, 2, 2))
        compiled_fn = torch.compile(
            transforms.RandomApply(
                transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.0), p=p
            )
        )

        compiled_fn(x)

    @pytest.mark.parametrize(
        "p,apply_transforms,repr",
        [
            (
                1.0,
                transforms.Normalize((0.5,), (0.5,)),
                "RandomApply(\n    p=1.0,\n    transforms=ModuleList(\n      (0): Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=False, value_check=False)\n    )\n)",
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


class TestRandomGaussianBlur(BaseTesterTransform):
    def test_functional_float(self):
        torch.manual_seed(28)
        tensor = self.get_float_image((3, 16, 16))

        out_not_blurred = transforms.RandomGaussianBlur(3, 0.1, 0.0)(tensor)
        transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.5)(tensor)
        out_blurred = transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 1.0)(tensor)

        torchvision_out = F_tv.gaussian_blur(tensor, (3, 3), 0.762560)

        torch.testing.assert_close(out_blurred, torchvision_out)
        torch.testing.assert_close(out_not_blurred, tensor)

    def test_functional_uint8(self):
        torch.manual_seed(28)
        tensor = self.get_uint8_image((3, 16, 16))

        out_not_blurred = transforms.RandomGaussianBlur(3, 0.1, 0.0)(tensor)
        transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.5)(tensor)
        out_blurred = transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 1.0)(tensor)

        torchvision_out = F_tv.gaussian_blur(tensor, (3, 3), 0.762560)

        torch.testing.assert_close(out_blurred, torchvision_out)
        torch.testing.assert_close(out_not_blurred, tensor)

    @pytest.mark.parametrize("p", [0, 0.5, 1.0])
    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self, p: float):
        torch.manual_seed(28)

        x = self.get_uint8_image((3, 2, 2))
        compiled_fn = torch.compile(
            transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), p=p)
        )

        compiled_fn(x)

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
        "kernel_size,error_type",
        [
            ((3, 3, 3), ValueError),
            ((3, 2), ValueError),
            ((3, 0), ValueError),
        ],
    )
    def test_wrong_parameters_kernel(self, kernel_size: Any, error_type: Exception):
        with pytest.raises(error_type):
            transforms.RandomGaussianBlur(kernel_size, [0.1, 2.0])

    @pytest.mark.parametrize(
        "sigma,error_type",
        [
            (0, ValueError),
            (-1, ValueError),
            ((2.0, 1.0), ValueError),
            ((0.0, 1.0), ValueError),
            ((0.1, 2.0, 3.0), ValueError),
            ("ahah", ValueError),
        ],
    )
    def test_wrong_parameters_sigma(self, sigma: Any, error_type: Exception):
        with pytest.raises(error_type):
            transforms.RandomGaussianBlur(3, sigma)


class TestVideoNormalize(BaseTesterTransform):
    @pytest.mark.parametrize(
        "input_shape,video_format", [((4, 3, 2, 2), "TCHW"), ((3, 4, 2, 2), "CTHW")]
    )
    @pytest.mark.parametrize(
        "mean,std", [(0.5, 0.5), ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    @pytest.mark.parametrize("inplace,value_check", [(True, True), (False, False)])
    @pytest.mark.parametrize("cast_dtype", [None])
    def test_functional_float(
        self,
        input_shape,
        mean: Sequence[float],
        std: Sequence[float],
        cast_dtype: torch.dtype | None,
        inplace: bool,
        value_check: bool,
        video_format: str,
    ):
        x = self.get_float_image(input_shape)
        in_x = x.clone() if inplace else x
        out = transforms.VideoNormalize(
            mean, std, cast_dtype, inplace, value_check, video_format
        )(in_x)

        if video_format == "CTHW":
            dims = [0, 2, 1, 3, 4] if x.ndim == 5 else [1, 0, 2, 3]
            expected_x = x.permute(dims)
        else:
            expected_x = x

        expected_out = F.normalize(expected_x, mean, std, cast_dtype, False, False)
        if video_format == "CTHW":
            expected_out = expected_out.permute(dims)

        torch.testing.assert_close(out, expected_out)
        if inplace:
            assert torch.equal(in_x, expected_out)
        else:
            assert not torch.equal(in_x, expected_out)

    @pytest.mark.parametrize(
        "input_shape,video_format", [((4, 3, 2, 2), "TCHW"), ((3, 4, 2, 2), "CTHW")]
    )
    @pytest.mark.parametrize(
        "mean,std", [(0.5, 0.5), ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    @pytest.mark.parametrize("inplace,value_check", [(True, True), (False, False)])
    @pytest.mark.parametrize("cast_dtype", [torch.float32])
    def test_functional_uint8(
        self,
        input_shape,
        mean: Sequence[float],
        std: Sequence[float],
        cast_dtype: torch.dtype | None,
        inplace: bool,
        value_check: bool,
        video_format: str,
    ):
        x = self.get_uint8_image(input_shape)
        in_x = x.clone() if inplace else x
        out = transforms.VideoNormalize(
            mean, std, cast_dtype, inplace, value_check, video_format
        )(in_x)

        if video_format == "CTHW":
            dims = [0, 2, 1, 3, 4] if x.ndim == 5 else [1, 0, 2, 3]
            expected_x = x.permute(dims)
        else:
            expected_x = x

        expected_out = F.normalize(expected_x, mean, std, cast_dtype, False, False)
        if video_format == "CTHW":
            expected_out = expected_out.permute(dims)

        torch.testing.assert_close(out, expected_out)
        assert not torch.equal(in_x, expected_out)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_float_image((3, 2, 2, 2))
        compiled_fn = torch.compile(transforms.VideoNormalize(0.5, 0.5))

        compiled_fn(x)

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
