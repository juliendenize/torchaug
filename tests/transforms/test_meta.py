import functools
import itertools

import pytest
import torch
import torchvision.transforms.v2.functional as TVF

import torchaug.transforms as transforms
import torchaug.transforms.functional as F
from torchaug import ta_tensors
from torchaug.ta_tensors._bounding_boxes import _convert_ta_format_to_tv_format

from ..utils import (
    BOUNDING_BOXES_MAKERS,
    assert_equal,
    check_functional,
    check_kernel,
    check_transform,
    cpu_and_cuda,
)


class TestConvertBoundingBoxFormat:
    old_new_formats = list(itertools.permutations(iter(ta_tensors.BoundingBoxFormat), 2))

    @pytest.mark.parametrize(("old_format", "new_format"), old_new_formats)
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_kernel(self, old_format, new_format, make_input):
        check_kernel(
            F.convert_bounding_box_format,
            make_input(format=old_format),
            new_format=new_format,
            old_format=old_format,
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_kernel_noop(self, format, inplace, make_input):
        input = make_input(format=format).as_subclass(torch.Tensor)
        input_version = input._version

        output = F.convert_bounding_box_format(input, old_format=format, new_format=format, inplace=inplace)

        assert output is input
        assert output.data_ptr() == input.data_ptr()
        assert output._version == input_version

    @pytest.mark.parametrize(("old_format", "new_format"), old_new_formats)
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_kernel_inplace(self, old_format, new_format, make_input):
        input = make_input(format=old_format).as_subclass(torch.Tensor)
        input_version = input._version

        output_out_of_place = F.convert_bounding_box_format(input, old_format=old_format, new_format=new_format)
        assert output_out_of_place.data_ptr() != input.data_ptr()
        assert output_out_of_place is not input

        output_inplace = F.convert_bounding_box_format(
            input, old_format=old_format, new_format=new_format, inplace=True
        )
        assert output_inplace.data_ptr() == input.data_ptr()
        assert output_inplace._version > input_version
        assert output_inplace is input

        assert_equal(output_inplace, output_out_of_place)

    @pytest.mark.parametrize(("old_format", "new_format"), old_new_formats)
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_functional(self, old_format, new_format, make_input):
        check_functional(
            F.convert_bounding_box_format,
            make_input(format=old_format),
            new_format=new_format,
        )

    @pytest.mark.parametrize(("old_format", "new_format"), old_new_formats)
    @pytest.mark.parametrize("format_type", ["enum", "str"])
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_transform(self, old_format, new_format, format_type, make_input):
        check_transform(
            transforms.ConvertBoundingBoxFormat(new_format.name if format_type == "str" else new_format),
            make_input(format=old_format),
        )

    @pytest.mark.parametrize(("old_format", "new_format"), old_new_formats)
    @pytest.mark.parametrize("batch", [True, False])
    def test_strings(self, old_format, new_format, batch):
        # Non-regression test for https://github.com/pytorch/vision/issues/8258
        input = (
            ta_tensors.BoundingBoxes(
                torch.tensor([[10, 10, 20, 20]]),
                format=old_format,
                canvas_size=(50, 50),
            )
            if not batch
            else (
                ta_tensors.BatchBoundingBoxes(
                    torch.tensor([[10, 10, 20, 20], [10, 10, 20, 20]]),
                    format=old_format,
                    canvas_size=(50, 50),
                    range_samples=[(0, 1), (1, 2)],
                )
            )
        )

        expected = TVF.convert_bounding_box_format(
            input,
            old_format=_convert_ta_format_to_tv_format(old_format),
            new_format=_convert_ta_format_to_tv_format(new_format),
        )

        old_format = old_format.name
        new_format = new_format.name

        out_functional = F.convert_bounding_box_format(input, new_format=new_format)
        out_functional_tensor = F.convert_bounding_box_format(
            input.as_subclass(torch.Tensor),
            old_format=old_format,
            new_format=new_format,
        )
        out_transform = transforms.ConvertBoundingBoxFormat(new_format)(input)
        for out in (out_functional, out_functional_tensor, out_transform):
            assert_equal(out, expected)

    @pytest.mark.parametrize(("old_format", "new_format"), old_new_formats)
    @pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("fn_type", ["functional", "transform"])
    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_correctness(self, old_format, new_format, dtype, device, fn_type, make_input):
        bounding_boxes = make_input(format=old_format, dtype=dtype, device=device)

        if fn_type == "functional":
            fn = functools.partial(F.convert_bounding_box_format, new_format=new_format)
        else:
            fn = transforms.ConvertBoundingBoxFormat(format=new_format)

        actual = fn(bounding_boxes)
        expected = TVF.convert_bounding_box_format(
            bounding_boxes,
            old_format=_convert_ta_format_to_tv_format(old_format),
            new_format=_convert_ta_format_to_tv_format(new_format),
        )

        assert_equal(actual, expected)

    @pytest.mark.parametrize("make_input", BOUNDING_BOXES_MAKERS)
    def test_errors(self, make_input):
        input_ta_tensor = make_input()
        input_pure_tensor = input_ta_tensor.as_subclass(torch.Tensor)

        for input in [input_ta_tensor, input_pure_tensor]:
            with pytest.raises(TypeError, match="missing 1 required argument: 'new_format'"):
                F.convert_bounding_box_format(input)

        with pytest.raises(ValueError, match="`old_format` has to be passed"):
            F.convert_bounding_box_format(input_pure_tensor, new_format=input_ta_tensor.format)

        with pytest.raises(ValueError, match="`old_format` must not be passed"):
            F.convert_bounding_box_format(
                input_ta_tensor,
                old_format=input_ta_tensor.format,
                new_format=input_ta_tensor.format,
            )


class TestClampBoundingBoxes:
    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_kernel(self, format, dtype, device, make_bounding_boxes):
        bounding_boxes = make_bounding_boxes(format=format, dtype=dtype, device=device)
        check_kernel(
            F.clamp_bounding_boxes,
            bounding_boxes,
            format=bounding_boxes.format,
            canvas_size=bounding_boxes.canvas_size,
        )

    @pytest.mark.parametrize("format", list(ta_tensors.BoundingBoxFormat))
    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_functional(self, format, make_bounding_boxes):
        check_functional(F.clamp_bounding_boxes, make_bounding_boxes(format=format))

    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_errors(self, make_bounding_boxes):
        input_ta_tensor = make_bounding_boxes()
        input_pure_tensor = input_ta_tensor.as_subclass(torch.Tensor)
        format, canvas_size = input_ta_tensor.format, input_ta_tensor.canvas_size

        for format_, canvas_size_ in [
            (None, None),
            (format, None),
            (None, canvas_size),
        ]:
            with pytest.raises(
                ValueError,
                match="For pure tensor inputs, `format` and `canvas_size` have to be passed.",
            ):
                F.clamp_bounding_boxes(input_pure_tensor, format=format_, canvas_size=canvas_size_)

        for format_, canvas_size_ in [
            (format, canvas_size),
            (format, None),
            (None, canvas_size),
        ]:
            with pytest.raises(
                ValueError,
                match="For bounding box ta_tensor inputs, `format` and `canvas_size` must not be passed.",
            ):
                F.clamp_bounding_boxes(input_ta_tensor, format=format_, canvas_size=canvas_size_)

    @pytest.mark.parametrize("make_bounding_boxes", BOUNDING_BOXES_MAKERS)
    def test_transform(self, make_bounding_boxes):
        check_transform(transforms.ClampBoundingBoxes(), make_bounding_boxes())
