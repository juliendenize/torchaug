from __future__ import annotations

import sys
from typing import Any, Sequence

import pytest
import torch
import torchvision.transforms.v2 as tv_transforms
import torchvision.transforms.v2.functional as F_tv
from torchvision.transforms.v2.functional import InterpolationMode

import torchaug.batch_transforms as transforms

from ...base_tester import BaseTesterTransform


class TestBatchRandomHorizontalFlip(BaseTesterTransform):
    def test_output_values_float(self):
        torch.manual_seed(28)
        imgs = self.get_float_image((4, 3, 8, 8))
        indices_to_apply = torch.tensor([2, 1])

        expected_out = imgs.clone()
        expected_out[indices_to_apply] = F_tv.hflip(imgs[indices_to_apply])

        out = transforms.BatchRandomHorizontalFlip(0.5)(imgs)
        torch.testing.assert_close(out, expected_out)

    def test_output_values_uint8(self):
        torch.manual_seed(28)
        imgs = self.get_uint8_image((4, 3, 8, 8))
        indices_to_apply = torch.tensor([2, 1])

        expected_out = imgs.clone()
        expected_out[indices_to_apply] = F_tv.hflip(imgs[indices_to_apply])

        out = transforms.BatchRandomHorizontalFlip(0.5)(imgs)
        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "p,inplace", [(0.5, False), (1.0, False), (0.0, False), (0.5, True)]
    )
    def test_functional_float(self, p: float, inplace: bool):
        imgs = self.get_float_image((4, 3, 8, 8))
        out = transforms.BatchRandomHorizontalFlip(p, inplace)(imgs)

        if inplace or p == 0:
            assert torch.equal(imgs, out)
        else:
            assert not torch.equal(imgs, out)

    @pytest.mark.parametrize(
        "p,inplace", [(0.5, False), (1.0, False), (0.0, False), (0.5, True)]
    )
    def test_functional_uint8(self, p: float, inplace: bool):
        imgs = self.get_uint8_image((4, 3, 8, 8))
        out = transforms.BatchRandomHorizontalFlip(p, inplace)(imgs)

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
        compiled_fn = torch.compile(transforms.BatchRandomHorizontalFlip(p=p))

        compiled_fn(x)

    @pytest.mark.parametrize(
        "p,inplace,repr",
        [
            (0.5, False, "BatchRandomHorizontalFlip(p=0.5, inplace=False)"),
            (1.0, False, "BatchRandomHorizontalFlip(p=1.0, inplace=False)"),
            (0.0, False, "BatchRandomHorizontalFlip(p=0.0, inplace=False)"),
            (0.5, True, "BatchRandomHorizontalFlip(p=0.5, inplace=True)"),
        ],
    )
    def test_repr(self, p: float, inplace: bool, repr: str):
        assert transforms.BatchRandomHorizontalFlip(p, inplace).__repr__() == repr


class TestBatchRandomResizedCrop(BaseTesterTransform):
    def test_output_values_float(self):
        torch.manual_seed(28)

        imgs = self.get_float_image((8, 3, 8, 8))
        indices_to_apply = torch.tensor([0, 1, 7, 5, 6, 4, 2, 3])

        i_s = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
        j_s = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2])
        h_s = torch.tensor([7, 7, 7, 7, 4, 4, 4, 4])
        w_s = torch.tensor([8, 8, 8, 8, 4, 4, 4, 4])

        expected_out = torch.empty((8, 3, 4, 4), dtype=imgs.dtype)

        for idx, i, j, h, w in zip(indices_to_apply, i_s, j_s, h_s, w_s):
            expected_out[idx] = F_tv.resized_crop(
                imgs[idx],
                i,
                j,
                h,
                w,
                [4, 4],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

        out = transforms.BatchRandomResizedCrop(
            4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=2
        )(imgs)

        torch.testing.assert_close(out, expected_out)

    def test_output_values_uint8(self):
        torch.manual_seed(28)

        imgs = self.get_uint8_image((8, 3, 8, 8))
        indices_to_apply = torch.tensor([0, 1, 7, 5, 6, 4, 2, 3])

        i_s = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
        j_s = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2])
        h_s = torch.tensor([7, 7, 7, 7, 4, 4, 4, 4])
        w_s = torch.tensor([8, 8, 8, 8, 4, 4, 4, 4])

        expected_out = torch.empty((8, 3, 4, 4), dtype=imgs.dtype)

        for idx, i, j, h, w in zip(indices_to_apply, i_s, j_s, h_s, w_s):
            expected_out[idx] = F_tv.resized_crop(
                imgs[idx],
                i,
                j,
                h,
                w,
                [4, 4],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

        out = transforms.BatchRandomResizedCrop(
            4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=2
        )(imgs)

        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "size,scale,ratio,interpolation,antialias,num_rand_calls",
        [
            (4, (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BILINEAR, True, -1),
            (4, (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BILINEAR, True, 0),
            (4, (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BILINEAR, True, 16),
            ([4], (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BILINEAR, True, 16),
            ([4], (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BICUBIC, False, 16),
        ],
    )
    def test_functional_float(
        self,
        size: int | Sequence[int],
        scale: Sequence[float],
        ratio: Sequence[float],
        interpolation: InterpolationMode,
        antialias: bool,
        num_rand_calls: int,
    ):
        imgs = self.get_float_image((8, 3, 8, 8))

        transforms.BatchRandomResizedCrop(
            size, scale, ratio, interpolation, antialias, num_rand_calls
        )(imgs)

    @pytest.mark.parametrize(
        "size,scale,ratio,interpolation,antialias,num_rand_calls",
        [
            (4, (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BILINEAR, True, -1),
            (4, (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BILINEAR, True, 0),
            (4, (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BILINEAR, True, 16),
            ([4], (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BILINEAR, True, 16),
            ([4], (0.08, 1), (3 / 4, 4 / 3), InterpolationMode.BICUBIC, False, 16),
        ],
    )
    def test_functional_uint8(
        self,
        size: int | Sequence[int],
        scale: Sequence[float],
        ratio: Sequence[float],
        interpolation: InterpolationMode,
        antialias: bool,
        num_rand_calls: int,
    ):
        imgs = self.get_uint8_image((8, 3, 8, 8))

        transforms.BatchRandomResizedCrop(
            size, scale, ratio, interpolation, antialias, num_rand_calls
        )(imgs)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)
        x = self.get_uint8_image((4, 3, 2, 2))
        compiled_fn = torch.compile(transforms.BatchRandomResizedCrop([2, 2]))

        compiled_fn(x)

    @pytest.mark.parametrize(
        "size,scale,ratio,interpolation,antialias,num_rand_calls,repr",
        [
            (
                4,
                (0.08, 1),
                (3 / 4, 4 / 3),
                InterpolationMode.BILINEAR,
                True,
                -1,
                "BatchRandomResizedCrop(size=(4, 4), scale=(0.08, 1), ratio=(0.75, 1.3333), interpolation=bilinear, antialias=True, num_rand_calls=-1)",
            ),
            (
                4,
                (0.08, 1),
                (3 / 4, 4 / 3),
                InterpolationMode.BILINEAR,
                True,
                0,
                "BatchRandomResizedCrop(size=(4, 4), scale=(0.08, 1), ratio=(0.75, 1.3333), interpolation=bilinear, antialias=True, num_rand_calls=0)",
            ),
            (
                4,
                (0.08, 1),
                (3 / 4, 4 / 3),
                InterpolationMode.BILINEAR,
                True,
                16,
                "BatchRandomResizedCrop(size=(4, 4), scale=(0.08, 1), ratio=(0.75, 1.3333), interpolation=bilinear, antialias=True, num_rand_calls=16)",
            ),
            (
                [4],
                (0.08, 1),
                (3 / 4, 4 / 3),
                InterpolationMode.BILINEAR,
                True,
                16,
                "BatchRandomResizedCrop(size=(4, 4), scale=(0.08, 1), ratio=(0.75, 1.3333), interpolation=bilinear, antialias=True, num_rand_calls=16)",
            ),
            (
                [4],
                (0.08, 1),
                (3 / 4, 4 / 3),
                InterpolationMode.BICUBIC,
                False,
                16,
                "BatchRandomResizedCrop(size=(4, 4), scale=(0.08, 1), ratio=(0.75, 1.3333), interpolation=bicubic, antialias=False, num_rand_calls=16)",
            ),
        ],
    )
    def test_repr(
        self,
        size: int | Sequence[int],
        scale: Sequence[float],
        ratio: Sequence[float],
        interpolation: InterpolationMode,
        antialias: bool,
        num_rand_calls,
        repr: str,
    ):
        assert (
            transforms.BatchRandomResizedCrop(
                size, scale, ratio, interpolation, antialias, num_rand_calls
            ).__repr__()
            == repr
        )

    @pytest.mark.parametrize(
        "num_rand_calls,error_type", [("ahah", ValueError), (-2, ValueError)]
    )
    def test_wrong_num_rand_calls(self, num_rand_calls: Any, error_type: Exception):
        with pytest.raises(error_type):
            transforms.BatchRandomResizedCrop(
                4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=num_rand_calls
            )

    @pytest.mark.parametrize("size,error_type", [(4.0, TypeError), ("ahah", TypeError)])
    def test_wrong_size(self, size: Any, error_type: Exception):
        with pytest.raises(error_type):
            transforms.BatchRandomResizedCrop(size, (0.08, 1), (3 / 4, 4 / 3))


class TestBatchVideoResize(BaseTesterTransform):
    def test_output_values_float(self):
        # test CTHW format
        tensor = self.get_float_image((2, 3, 8, 16, 16))
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
        tensor = self.get_float_image((2, 8, 3, 16, 16))
        expected_out = tv_transforms.Resize(2, antialias=True)(
            tensor.reshape(2 * 8, 3, 16, 16)
        ).reshape(2, 8, 3, 2, 2)
        out = transforms.BatchVideoResize(size=2, video_format="TCHW")(tensor)
        torch.testing.assert_close(out, expected_out)

    def test_output_values_uint8(self):
        # test CTHW format
        tensor = self.get_uint8_image((2, 3, 8, 16, 16))
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
        tensor = self.get_uint8_image((2, 8, 3, 16, 16))
        expected_out = tv_transforms.Resize(2, antialias=True)(
            tensor.reshape(2 * 8, 3, 16, 16)
        ).reshape(2, 8, 3, 2, 2)
        out = transforms.BatchVideoResize(size=2, video_format="TCHW")(tensor)
        torch.testing.assert_close(out, expected_out)

    @pytest.mark.parametrize(
        "size,interpolation,max_size,antialias,video_format",
        [
            (2, InterpolationMode.BILINEAR, 4, False, "CTHW"),
            ([2, 8], InterpolationMode.BICUBIC, None, True, "TCHW"),
        ],
    )
    def test_functional_float(
        self,
        size: int | list[int],
        interpolation: InterpolationMode,
        max_size: int | None,
        antialias: bool,
        video_format: str,
    ):
        tensor = (
            self.get_float_image((2, 8, 3, 16, 16))
            if video_format == "TCHW"
            else self.get_float_image((2, 3, 8, 16, 16))
        )
        transforms.BatchVideoResize(
            size=size,
            interpolation=interpolation,
            max_size=max_size,
            antialias=antialias,
            video_format=video_format,
        )(tensor)

    @pytest.mark.parametrize(
        "size,interpolation,max_size,antialias,video_format",
        [
            (2, InterpolationMode.BILINEAR, 4, False, "CTHW"),
            ([2, 8], InterpolationMode.BICUBIC, None, True, "TCHW"),
        ],
    )
    def test_functional_uint8(
        self,
        size: int | list[int],
        interpolation: InterpolationMode,
        max_size: int | None,
        antialias: bool,
        video_format: str,
    ):
        tensor = (
            self.get_uint8_image((2, 8, 3, 16, 16))
            if video_format == "TCHW"
            else self.get_uint8_image((2, 3, 8, 16, 16))
        )
        transforms.BatchVideoResize(
            size=size,
            interpolation=interpolation,
            max_size=max_size,
            antialias=antialias,
            video_format=video_format,
        )(tensor)

    @pytest.mark.skipif(
        sys.version_info >= (3, 11), reason="requires python3.10 or lower."
    )
    def test_compile(self):
        torch.manual_seed(28)
        x = self.get_uint8_image((4, 3, 2, 2, 2))
        compiled_fn = torch.compile(transforms.BatchVideoResize([2, 2]))

        compiled_fn(x)

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
