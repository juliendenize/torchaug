from __future__ import annotations

import sys
from typing import Sequence

import pytest
import torch
from torch import nn

import torchaug.transforms as transforms
from torchaug.utils import VideoFormat

from ...base_tester import BaseTesterTransform


class TestImageWrapper(BaseTesterTransform):
    @pytest.mark.parametrize("input_shape", [(3, 2, 2)])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_functional_float(self, input_shape, inplace):
        torch.manual_seed(28)

        list_transforms = (
            transforms.Normalize((0.5,), (0.5,), inplace=False, value_check=True),
            transforms.Mul255(),
        )

        tensor = self.get_float_image(input_shape)
        in_tensor = tensor.clone() if inplace else tensor

        out = transforms.ImageWrapper(list_transforms, inplace=inplace)(in_tensor)

        expected_out = (
            transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor) * 255.0
        )
        torch.testing.assert_close(out, expected_out)

        if inplace:
            assert torch.equal(out, in_tensor)
        else:
            assert not torch.equal(out, in_tensor)

        for transform in list_transforms:
            if hasattr(transform, "inplace"):
                assert transform.inplace

    @pytest.mark.parametrize("input_shape", [(3, 2, 2)])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_functional_uint8(self, input_shape, inplace):
        torch.manual_seed(28)

        transform = (
            transforms.Normalize(
                (0.5,),
                (0.5,),
                cast_dtype=torch.float32,
                inplace=False,
                value_check=True,
            ),
        )

        tensor = self.get_uint8_image(input_shape)
        in_tensor = tensor.clone() if inplace else tensor

        out = transforms.ImageWrapper(transform, inplace=inplace)(in_tensor)

        expected_out = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), cast_dtype=torch.float32, inplace=False
        )(tensor)
        torch.testing.assert_close(out, expected_out)

        assert not torch.equal(out, in_tensor)

        if hasattr(transform, "inplace"):
            assert transform.inplace

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_uint8_image((3, 2, 2))

        transform = (
            transforms.Normalize(
                (0.5,),
                (0.5,),
                cast_dtype=torch.float32,
                inplace=False,
                value_check=True,
            ),
        )

        compiled_fn = torch.compile(transforms.ImageWrapper(transform))

        compiled_fn(x)

    @pytest.mark.parametrize(
        "list_transforms,inplace,repr",
        [
            (
                transforms.Normalize((0.5,), (0.5,), inplace=False, value_check=True),
                False,
                "ImageWrapper(\n    inplace=False,\n    transforms=ModuleList(\n      (0): Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True)\n    )\n)",
            ),
            (
                [
                    transforms.Normalize(
                        (0.5,), (0.5,), inplace=False, value_check=True
                    ),
                    transforms.Div255(inplace=False),
                ],
                True,
                "ImageWrapper(\n    inplace=True,\n    transforms=ModuleList(\n      (0): Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True)\n      (1): Div255(inplace=True)\n    )\n)",
            ),
        ],
    )
    def test_repr(
        self, list_transforms: nn.Module | Sequence[nn.Module], inplace: bool, repr: str
    ):
        assert transforms.ImageWrapper(list_transforms, inplace).__repr__() == repr

    def test_wrong_tensor(self):
        transform = transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        tensor = torch.rand(16)
        with pytest.raises(TypeError):
            transforms.ImageWrapper(transform)(tensor)

    def test_wrong_transforms(self):
        with pytest.raises(TypeError):
            transforms.ImageWrapper("ahah")


class TestVideoWrapper(BaseTesterTransform):
    @pytest.mark.parametrize(
        "input_shape,video_format", [((3, 4, 2, 2), "CTHW"), ((4, 3, 2, 2), "TCHW")]
    )
    @pytest.mark.parametrize("inplace", [True, False])
    def test_functional_float(self, input_shape, inplace, video_format):
        torch.manual_seed(28)

        list_transforms = (
            transforms.VideoNormalize(
                (0.5,),
                (0.5,),
                video_format=video_format,
                inplace=False,
                value_check=True,
            ),
        )

        tensor = self.get_float_image(input_shape)
        in_tensor = tensor.clone() if inplace else tensor

        expected_out = transforms.VideoNormalize(
            (0.5,), (0.5,), video_format=video_format, inplace=False
        )(tensor)
        out = transforms.VideoWrapper(
            list_transforms, inplace=inplace, video_format=video_format
        )(in_tensor)
        torch.testing.assert_close(out, expected_out)

        for transform in list_transforms:
            if hasattr(transform, "inplace"):
                assert transform.inplace
            if hasattr(transform, "video_format"):
                assert transform.video_format == VideoFormat.TCHW

        if inplace and video_format == "TCHW":
            assert torch.equal(out, in_tensor)
        else:
            assert not torch.equal(out, in_tensor)

    @pytest.mark.parametrize(
        "input_shape,video_format", [((3, 4, 2, 2), "CTHW"), ((4, 3, 2, 2), "TCHW")]
    )
    @pytest.mark.parametrize("inplace", [True, False])
    def test_functional_uint8(self, input_shape, inplace, video_format):
        torch.manual_seed(28)

        list_transforms = (
            transforms.VideoNormalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
                cast_dtype=torch.float32,
                inplace=False,
                value_check=True,
            ),
            transforms.Mul255(),
        )

        tensor = self.get_uint8_image(input_shape)
        in_tensor = tensor.clone() if inplace else tensor

        expected_out = (
            transforms.VideoNormalize(
                (0.5,), (0.5,), cast_dtype=torch.float32, inplace=False
            )(tensor)
            * 255.0
        )
        out = transforms.VideoWrapper(
            list_transforms, inplace=inplace, video_format=video_format
        )(in_tensor)
        torch.testing.assert_close(out, expected_out)

        for transform in list_transforms:
            if hasattr(transform, "inplace"):
                assert transform.inplace
            if hasattr(transform, "video_format"):
                assert transform.video_format == VideoFormat.TCHW

        assert not torch.equal(out, in_tensor)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_uint8_image((3, 2, 2, 2))

        transform = (
            transforms.VideoNormalize(
                (0.5,),
                (0.5,),
                cast_dtype=torch.float32,
                inplace=False,
                value_check=True,
            ),
        )

        compiled_fn = torch.compile(transforms.VideoWrapper(transform))

        compiled_fn(x)

    @pytest.mark.parametrize(
        "list_transforms,inplace,video_format,repr",
        [
            (
                transforms.VideoNormalize(
                    (0.5,), (0.5,), inplace=False, value_check=True, video_format="CTHW"
                ),
                False,
                "TCHW",
                "VideoWrapper(\n    inplace=False,\n    video_format=TCHW,\n    transforms=ModuleList(\n      (0): VideoNormalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True, video_format=TCHW)\n    )\n)",
            ),
            (
                [
                    transforms.VideoNormalize(
                        (0.5,),
                        (0.5,),
                        inplace=False,
                        value_check=True,
                        video_format="TCHW",
                    ),
                    transforms.Div255(inplace=False),
                ],
                False,
                "CTHW",
                "VideoWrapper(\n    inplace=False,\n    video_format=CTHW,\n    transforms=ModuleList(\n      (0): VideoNormalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True, video_format=TCHW)\n      (1): Div255(inplace=True)\n    )\n)",
            ),
        ],
    )
    def test_repr(
        self,
        list_transforms: nn.Module | Sequence[nn.Module],
        inplace: bool,
        video_format: str,
        repr: str,
    ):
        assert (
            transforms.VideoWrapper(list_transforms, inplace, video_format).__repr__()
            == repr
        )

    def test_wrong_tensor(self):
        transform = transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        tensor = torch.rand((6, 3, 2, 3, 16, 16))
        with pytest.raises(TypeError):
            transforms.VideoWrapper(transform)(tensor)

    def test_wrong_transforms(self):
        with pytest.raises(TypeError):
            transforms.VideoWrapper("ahah")


class TestWrapper(BaseTesterTransform):
    @pytest.mark.parametrize("input_shape", [(3, 2, 2)])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_functional_float(self, input_shape, inplace):
        torch.manual_seed(28)

        list_transforms = (
            transforms.Normalize((0.5,), (0.5,), inplace=False, value_check=True),
            transforms.Mul255(),
        )

        tensor = self.get_float_image(input_shape)
        in_tensor = tensor.clone() if inplace else tensor

        out = transforms.Wrapper(list_transforms, inplace=inplace)(in_tensor)

        expected_out = (
            transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor) * 255.0
        )
        torch.testing.assert_close(out, expected_out)

        if inplace:
            assert torch.equal(out, in_tensor)
        else:
            assert not torch.equal(out, in_tensor)

        for transform in list_transforms:
            if hasattr(transform, "inplace"):
                assert transform.inplace

    @pytest.mark.parametrize("input_shape", [(3, 2, 2)])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_functional_uint8(self, input_shape, inplace):
        torch.manual_seed(28)

        transform = (
            transforms.Normalize(
                (0.5,),
                (0.5,),
                cast_dtype=torch.float32,
                inplace=False,
                value_check=True,
            ),
        )

        tensor = self.get_uint8_image(input_shape)
        in_tensor = tensor.clone() if inplace else tensor

        out = transforms.Wrapper(transform, inplace=inplace)(in_tensor)

        expected_out = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), cast_dtype=torch.float32, inplace=False
        )(tensor)
        torch.testing.assert_close(out, expected_out)

        assert not torch.equal(out, in_tensor)

        if hasattr(transform, "inplace"):
            assert transform.inplace

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_uint8_image((3, 2, 2))

        transform = (
            transforms.Normalize(
                (0.5,),
                (0.5,),
                cast_dtype=torch.float32,
                inplace=False,
                value_check=True,
            ),
        )

        compiled_fn = torch.compile(transforms.ImageWrapper(transform))

        compiled_fn(x)

    @pytest.mark.parametrize(
        "list_transforms,inplace,repr",
        [
            (
                transforms.Normalize((0.5,), (0.5,), inplace=False, value_check=True),
                False,
                "Wrapper(\n    inplace=False,\n    transforms=ModuleList(\n      (0): Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True)\n    )\n)",
            ),
            (
                [
                    transforms.Normalize(
                        (0.5,), (0.5,), inplace=False, value_check=True
                    ),
                    transforms.Div255(inplace=False),
                ],
                True,
                "Wrapper(\n    inplace=True,\n    transforms=ModuleList(\n      (0): Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True)\n      (1): Div255(inplace=True)\n    )\n)",
            ),
        ],
    )
    def test_repr(
        self, list_transforms: nn.Module | Sequence[nn.Module], inplace: bool, repr: str
    ):
        assert transforms.Wrapper(list_transforms, inplace).__repr__() == repr

    def test_wrong_input(self):
        transform = transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        tensor = "ahah"
        with pytest.raises(TypeError):
            transforms.Wrapper(transform)(tensor)

    def test_wrong_transforms(self):
        with pytest.raises(TypeError):
            transforms.Wrapper("ahah")

    def test_wrong_video_format(self):
        transform = transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        with pytest.raises(ValueError):
            transforms.VideoWrapper(transform, video_format="ahah")
