from __future__ import annotations

import re
import sys
from typing import Any, Sequence

import pytest
import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as F_tv
from torch import nn
from torchvision.transforms.functional import InterpolationMode

import torchaug.batch_transforms as transforms
import torchaug.batch_transforms.functional as F_b
import torchaug.transforms as mono_transforms
import torchaug.transforms.functional as F
from torchaug.utils import VideoFormat

from ...base_tester import BaseTesterTransform


class TestBatchImageWrapper(BaseTesterTransform):
    def test_output_values_float(self):
        torch.manual_seed(28)

        transform = (
            mono_transforms.Normalize((0.5,), (0.5,), inplace=False, value_check=True),
            mono_transforms.Mul255(),
        )

        tensor = self.get_float_image((3, 3, 16, 16))
        expected_out = (
            mono_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor) * 255.0
        )
        out = transforms.BatchImageWrapper(transform)(tensor)
        torch.testing.assert_close(out, expected_out)

    def test_output_values_uint8(self):
        torch.manual_seed(28)

        transform = (mono_transforms.RandomSolarize(128, 1.0),)

        tensor = self.get_uint8_image((3, 3, 16, 16))
        expected_out = mono_transforms.RandomSolarize(128, 1.0)(tensor)
        out = transforms.BatchImageWrapper(transform)(tensor)
        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "list_transforms,inplace",
        [
            (
                mono_transforms.Normalize(
                    (0.5,), (0.5,), inplace=False, value_check=True
                ),
                False,
            ),
            (
                [
                    mono_transforms.Normalize(
                        (0.5,), (0.5,), inplace=False, value_check=True
                    ),
                    mono_transforms.Div255(inplace=False),
                ],
                True,
            ),
        ],
    )
    def test_functional_float(
        self, list_transforms: nn.Module | Sequence[nn.Module], inplace: bool
    ):
        torch.manual_seed(28)

        tensor = self.get_float_image((2, 3, 16, 16))

        input_tensor = tensor.clone() if inplace else tensor

        wrapper = transforms.BatchImageWrapper(list_transforms, inplace)

        if not isinstance(list_transforms, Sequence):
            list_transforms = [list_transforms]

        for transform in list_transforms:
            if hasattr(transform, "inplace"):
                assert transform.inplace

        output = wrapper(input_tensor)

        if inplace:
            assert torch.equal(output, input_tensor)
        else:
            assert not torch.equal(output, input_tensor)

    @pytest.mark.parametrize(
        "list_transforms,inplace",
        [
            (
                transforms.BatchRandomSolarize(
                    128, 0.5, inplace=False, value_check=True
                ),
                False,
            ),
            (
                [
                    transforms.BatchRandomSolarize(
                        128, 0.5, inplace=False, value_check=True
                    ),
                    transforms.BatchRandomColorJitter(0.5, inplace=False),
                ],
                True,
            ),
        ],
    )
    def test_functional_uint8(
        self, list_transforms: nn.Module | Sequence[nn.Module], inplace: bool
    ):
        torch.manual_seed(28)

        tensor = self.get_uint8_image((2, 3, 16, 16))

        input_tensor = tensor.clone() if inplace else tensor

        wrapper = transforms.BatchImageWrapper(list_transforms, inplace)

        if not isinstance(list_transforms, Sequence):
            list_transforms = [list_transforms]

        for transform in list_transforms:
            if hasattr(transform, "inplace"):
                assert transform.inplace

        output = wrapper(input_tensor)

        if inplace:
            assert torch.equal(output, input_tensor)
        else:
            assert not torch.equal(output, input_tensor)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_uint8_image((4, 3, 2, 2))
        compiled_fn = torch.compile(
            transforms.BatchImageWrapper(
                transforms.BatchRandomSolarize(
                    128, 0.5, inplace=False, value_check=True
                )
            )
        )

        compiled_fn(x)

    @pytest.mark.parametrize(
        "list_transforms,inplace,repr",
        [
            (
                mono_transforms.Normalize(
                    (0.5,), (0.5,), inplace=False, value_check=True
                ),
                False,
                "BatchImageWrapper(\n    inplace=False,\n    transforms=ModuleList(\n      (0): Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True)\n    )\n)",
            ),
            (
                [
                    mono_transforms.Normalize(
                        (0.5,), (0.5,), inplace=False, value_check=True
                    ),
                    mono_transforms.Div255(inplace=False),
                ],
                True,
                "BatchImageWrapper(\n    inplace=True,\n    transforms=ModuleList(\n      (0): Normalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True)\n      (1): Div255(inplace=True)\n    )\n)",
            ),
        ],
    )
    def test_repr(
        self, list_transforms: nn.Module | Sequence[nn.Module], inplace: bool, repr: str
    ):
        assert transforms.BatchImageWrapper(list_transforms, inplace).__repr__() == repr

    def test_wrong_tensor(self):
        transform = mono_transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        tensor = torch.rand((3, 16, 16))
        with pytest.raises(TypeError):
            transforms.BatchVideoWrapper(transform)(tensor)

    def test_wrong_transforms(self):
        with pytest.raises(TypeError):
            transforms.BatchImageWrapper("ahah")


class TestBatchVideoWrapper(BaseTesterTransform):
    def test_output_values_float(self):
        torch.manual_seed(28)

        transform = (
            mono_transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False, value_check=True
            ),
            mono_transforms.Mul255(),
        )

        # test CTHW format
        tensor = self.get_float_image((2, 3, 2, 16, 16))
        expected_out = (
            mono_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor) * 255.0
        )
        out = transforms.BatchVideoWrapper(transform)(tensor)
        torch.testing.assert_close(out, expected_out)

        # test TCHW format
        expected_out = (
            mono_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor) * 255.0
        )
        out = transforms.BatchVideoWrapper(transform, video_format="TCHW")(
            tensor.permute(0, 2, 1, 3, 4)
        )
        torch.testing.assert_close(out.permute(0, 2, 1, 3, 4), expected_out)

        # test same_on_frames True
        image = self.get_float_image((3, 224, 224))
        video = torch.stack([image, image])
        batch_video = torch.stack([video, video])
        out = transforms.BatchVideoWrapper(
            transforms.BatchRandomColorJitter(0.5, p=1.0),
            video_format="TCHW",
            same_on_frames=True,
        )(batch_video)
        torch.testing.assert_close(out[:, 0], out[:, 1])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out[0], out[1])

        # test same_on_frames False
        out = transforms.BatchVideoWrapper(
            transforms.BatchRandomColorJitter(0.5, p=1.0),
            video_format="TCHW",
            same_on_frames=False,
        )(batch_video)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out[:, 0], out[:, 1])

    def test_output_values_uint8(self):
        torch.manual_seed(28)

        transform = [
            transforms.BatchRandomSolarize(
                128,
                p=1,
                inplace=False,
                value_check=True,
            )
        ]

        # test CTHW format
        tensor = self.get_uint8_image((2, 3, 3, 16, 16))
        expected_out = transforms.BatchRandomSolarize(
            128,
            p=1,
            inplace=False,
            value_check=True,
        )(tensor)
        out = transforms.BatchVideoWrapper(transform)(tensor)
        torch.testing.assert_close(out, expected_out)

        # test TCHW format
        expected_out = transforms.BatchRandomSolarize(
            128,
            p=1,
            inplace=False,
            value_check=True,
        )(tensor)
        out = transforms.BatchVideoWrapper(transform, video_format="TCHW")(
            tensor.permute(0, 2, 1, 3, 4)
        )
        torch.testing.assert_close(out.permute(0, 2, 1, 3, 4), expected_out)

        # test same_on_frames True
        image = self.get_uint8_image((3, 224, 224))
        video = torch.stack([image, image])
        batch_video = torch.stack([video, video])
        out = transforms.BatchVideoWrapper(
            transforms.BatchRandomColorJitter(
                0.5,
                p=1,
                inplace=False,
                value_check=True,
            ),
            video_format="TCHW",
        )(batch_video)
        torch.testing.assert_close(out[:, 0], out[:, 1])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out[0], out[1])

        # test same_on_frames False
        out = transforms.BatchVideoWrapper(
            transforms.BatchRandomColorJitter(
                0.5,
                p=1,
                inplace=False,
                value_check=True,
            ),
            same_on_frames=False,
            video_format="TCHW",
        )(batch_video)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out[:, 0], out[:, 1])

    @pytest.mark.parametrize(
        "list_transforms,inplace,same_on_frames,video_format",
        [
            (
                mono_transforms.VideoNormalize(
                    (0.5, 0.5, 0.5),
                    (0.5,),
                    inplace=False,
                    value_check=True,
                    video_format="CTHW",
                ),
                False,
                False,
                "CTHW",
            ),
            (
                mono_transforms.VideoNormalize(
                    (0.5, 0.5, 0.5),
                    (0.5,),
                    inplace=False,
                    value_check=True,
                    video_format="TCHW",
                ),
                True,
                False,
                "CTHW",
            ),
            (
                [
                    mono_transforms.VideoNormalize(
                        (0.5, 0.5, 0.5),
                        (0.5,),
                        inplace=False,
                        value_check=True,
                        video_format="CTHW",
                    ),
                    mono_transforms.Div255(inplace=False),
                ],
                False,
                True,
                "CTHW",
            ),
            (
                [
                    mono_transforms.VideoNormalize(
                        (0.5, 0.5, 0.5),
                        (0.5,),
                        inplace=True,
                        value_check=True,
                        video_format="CTHW",
                    ),
                    mono_transforms.Div255(inplace=False),
                ],
                True,
                True,
                "TCHW",
            ),
        ],
    )
    def test_functional_float(
        self,
        list_transforms: nn.Module | Sequence[nn.Module],
        inplace: bool,
        same_on_frames: bool,
        video_format: str,
    ):
        torch.manual_seed(28)

        if video_format == "CTHW":
            tensor = self.get_float_image((2, 3, 5, 16, 16))
        else:
            tensor = self.get_float_image((2, 5, 3, 16, 16))

        input_tensor = tensor.clone() if inplace else tensor

        wrapper = transforms.BatchVideoWrapper(
            list_transforms, inplace, same_on_frames, video_format
        )

        if not isinstance(list_transforms, Sequence):
            list_transforms = [list_transforms]

        for transform in list_transforms:
            if hasattr(transform, "inplace"):
                assert transform.inplace
            if hasattr(transform, "video_format"):
                assert transform.video_format == VideoFormat.TCHW

        output = wrapper(input_tensor)

        if inplace and not video_format == "CTHW":
            assert torch.equal(output, input_tensor)
        else:
            assert not torch.equal(output, input_tensor)

    @pytest.mark.parametrize(
        "list_transforms,inplace,same_on_frames,video_format",
        [
            (
                transforms.BatchRandomColorJitter(
                    0.5,
                    inplace=False,
                    value_check=True,
                ),
                False,
                False,
                "CTHW",
            ),
            (
                [
                    transforms.BatchRandomColorJitter(
                        0.5,
                        inplace=False,
                        value_check=True,
                    ),
                    transforms.BatchRandomColorJitter(
                        0.5,
                        inplace=False,
                        value_check=True,
                    ),
                ],
                True,
                True,
                "TCHW",
            ),
        ],
    )
    def test_functional_uint8(
        self,
        list_transforms: nn.Module | Sequence[nn.Module],
        inplace: bool,
        same_on_frames: bool,
        video_format: str,
    ):
        torch.manual_seed(28)

        if video_format == "CTHW":
            tensor = self.get_uint8_image((2, 3, 5, 16, 16))
        else:
            tensor = self.get_uint8_image((2, 5, 3, 16, 16))

        input_tensor = tensor.clone() if inplace else tensor

        wrapper = transforms.BatchVideoWrapper(
            list_transforms, inplace, same_on_frames, video_format
        )

        if not isinstance(list_transforms, Sequence):
            list_transforms = [list_transforms]

        for transform in list_transforms:
            if hasattr(transform, "inplace"):
                assert transform.inplace
            if hasattr(transform, "video_format"):
                assert transform.video_format == "TCHW"

        output = wrapper(input_tensor)

        if inplace and not video_format == "CTHW":
            assert torch.equal(output, input_tensor)
        else:
            assert not torch.equal(output, input_tensor)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)

        x = self.get_uint8_image((4, 3, 2, 2, 2))
        compiled_fn = torch.compile(
            transforms.BatchVideoWrapper(
                transforms.BatchRandomSolarize(
                    128, 0.5, inplace=False, value_check=True
                )
            )
        )

        compiled_fn(x)

    @pytest.mark.parametrize(
        "list_transforms,inplace,same_on_frames,video_format,repr",
        [
            (
                mono_transforms.VideoNormalize(
                    (0.5,), (0.5,), inplace=False, value_check=True, video_format="CTHW"
                ),
                False,
                False,
                "TCHW",
                "BatchVideoWrapper(\n    inplace=False,\n    same_on_frames=False,\n    video_format=TCHW,\n    transforms=ModuleList(\n      (0): VideoNormalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True, video_format=TCHW)\n    )\n)",
            ),
            (
                [
                    mono_transforms.VideoNormalize(
                        (0.5,),
                        (0.5,),
                        inplace=False,
                        value_check=True,
                        video_format="TCHW",
                    ),
                    mono_transforms.Div255(inplace=False),
                ],
                False,
                True,
                "CTHW",
                "BatchVideoWrapper(\n    inplace=False,\n    same_on_frames=True,\n    video_format=CTHW,\n    transforms=ModuleList(\n      (0): VideoNormalize(mean=[[[0.5]]], std=[[[0.5]]], cast_dtype=None, inplace=True, value_check=True, video_format=TCHW)\n      (1): Div255(inplace=True)\n    )\n)",
            ),
        ],
    )
    def test_repr(
        self,
        list_transforms: nn.Module | Sequence[nn.Module],
        inplace: bool,
        same_on_frames: bool,
        video_format: str,
        repr: str,
    ):
        assert (
            transforms.BatchVideoWrapper(
                list_transforms, inplace, same_on_frames, video_format
            ).__repr__()
            == repr
        )

    def test_wrong_video_format(self):
        transform = mono_transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        with pytest.raises(ValueError):
            transforms.BatchVideoWrapper(transform, video_format="ahah")

    def test_wrong_tensor(self):
        transform = mono_transforms.Normalize(
            (0.5,), (0.5,), inplace=False, value_check=True
        )
        tensor = torch.rand((6, 3, 2, 3, 16, 16))
        with pytest.raises(TypeError):
            transforms.BatchVideoWrapper(transform)(tensor)

    def test_wrong_transforms(self):
        with pytest.raises(TypeError):
            transforms.BatchVideoWrapper("ahah")


class TestBatchVideoResize:
    def test_functional(self):
        # test CTHW format
        tensor = torch.rand((2, 3, 8, 16, 16))
        expected_out = (
            tv_transforms.Resize(2, antialias=True)(
                tensor.permute(0, 2, 1, 3, 4).reshape(2 * 8, 3, 16, 16)
            )
            .reshape(2, 8, 3, 2, 2)
            .permute(0, 2, 1, 3, 4)
        )
        out = transforms.BatchVideoResize(size=2, video_format="CTHW")(tensor)
        torch.testing.assert_close(out, expected_out)

        # test TCHW format
        tensor = torch.rand((2, 8, 3, 16, 16))
        expected_out = tv_transforms.Resize(2, antialias=True)(
            tensor.reshape(2 * 8, 3, 16, 16)
        ).reshape(2, 8, 3, 2, 2)
        out = transforms.BatchVideoResize(size=2, video_format="TCHW")(tensor)
        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "size,interpolation,max_size,antialias,video_format,repr",
        [
            (
                2,
                InterpolationMode.BILINEAR,
                None,
                True,
                "CTHW",
                "BatchVideoResize(size=2, interpolation=bilinear, max_size=None, antialias=True, video_format=CTHW)",
            ),
            (
                [2, 2],
                InterpolationMode.BICUBIC,
                224,
                True,
                "TCHW",
                "BatchVideoResize(size=[2, 2], interpolation=bicubic, max_size=224, antialias=True, video_format=TCHW)",
            ),
            (
                [2, 2],
                3,
                224,
                True,
                "TCHW",
                "BatchVideoResize(size=[2, 2], interpolation=bicubic, max_size=224, antialias=True, video_format=TCHW)",
            ),
        ],
    )
    def test_repr(
        self,
        size: int | list[int],
        interpolation: InterpolationMode,
        max_size: int | None,
        antialias: bool,
        video_format: str,
        repr: str,
    ):
        assert (
            transforms.BatchVideoResize(
                size, interpolation, max_size, antialias, video_format
            ).__repr__()
            == repr
        )

    @pytest.mark.parametrize(
        "size,error_type", [(None, TypeError), ([2, 3, 4], ValueError)]
    )
    def test_wrong_size(self, size: Any, error_type: Exception):
        with pytest.raises(error_type):
            transforms.BatchVideoResize(size)

    def test_wrong_video_format(self):
        with pytest.raises(ValueError):
            transforms.BatchVideoResize(2, video_format="ahah")

    def test_wrong_tensor(self):
        tensor = torch.rand((6, 3, 2, 3, 16, 16))
        with pytest.raises(TypeError):
            transforms.BatchVideoResize(2)(tensor)
