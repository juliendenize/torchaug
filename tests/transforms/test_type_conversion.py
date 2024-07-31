import numpy as np
import pytest
import torch
from torchvision.transforms import ToPILImage

import torchaug.transforms as transforms
import torchaug.transforms.functional as F
from torchaug import ta_tensors
from torchaug.ta_tensors import (
    BatchBoundingBoxes,
    BoundingBoxesNestedTensors,
    ImageNestedTensors,
    LabelsNestedTensors,
    MaskNestedTensors,
    NestedTensors,
    VideoNestedTensors,
    convert_bboxes_to_batch_bboxes,
    convert_labels_to_batch_labels,
    convert_masks_to_batch_masks,
)

from ..utils import (
    assert_equal,
    make_batch_images,
    make_bounding_boxes,
    make_detection_masks,
    make_image,
    make_image_tensor,
    make_labels,
    make_video,
    transform_cls_to_functional,
)


class TestToImage:
    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_batch_images])
    @pytest.mark.parametrize("fn", [F.to_image, transform_cls_to_functional(transforms.ToImage)])
    def test_functional_and_transform(self, make_input, fn):
        input = make_input()
        output = fn(input)

        assert isinstance(output, ta_tensors.Image)

        input_size = list(input.shape[:2]) if isinstance(input, np.ndarray) else F.get_size(input)
        assert F.get_size(output) == input_size

        if isinstance(input, torch.Tensor):
            assert output.data_ptr() == input.data_ptr()

    @pytest.mark.parametrize("fn", [F.to_image, transform_cls_to_functional(transforms.ToImage)])
    def test_image_functional_and_transform(self, fn):
        input = ToPILImage()(torch.ones(3, 10, 10, dtype=torch.uint8))
        output = fn(input)

        assert isinstance(output, ta_tensors.Image)
        assert F.get_size(output) == [10, 10]

    def test_2d_np_array(self):
        # Non-regression test for https://github.com/pytorch/vision/issues/8255
        input = np.random.rand(10, 10)
        assert F.to_image(input).shape == (1, 10, 10)

    def test_functional_error(self):
        with pytest.raises(
            TypeError,
            match="Input can either be a pure Tensor, a numpy array",
        ):
            F.to_image(object())


class TestToBatchImages:
    @pytest.mark.parametrize("make_input", [make_image, make_image_tensor])
    @pytest.mark.parametrize("fn", [F.to_batch_images, transform_cls_to_functional(transforms.ToBatchImages)])
    def test_functional_and_transform(self, make_input, fn):
        input = make_input(batch_dims=(2,))
        output = fn(input)

        assert isinstance(output, ta_tensors.BatchImages)

        input_size = F.get_size(input)
        assert F.get_size(output) == input_size

        assert output.data_ptr() == input.data_ptr()

    def test_functional_error(self):
        with pytest.raises(
            TypeError,
            match="Input should be a Tensor",
        ):
            F.to_batch_images(object())


class TestNestedToList:
    @pytest.mark.parametrize(
        "make_input,type_nested",
        [
            (make_image, ImageNestedTensors),
            (make_video, VideoNestedTensors),
            (make_bounding_boxes, BoundingBoxesNestedTensors),
            (make_detection_masks, MaskNestedTensors),
            (make_labels, LabelsNestedTensors),
            (make_image_tensor, NestedTensors),
        ],
    )
    def test_functional(self, make_input, type_nested):
        tensor = make_input()
        tensors = [tensor] * 5
        nested_tensors = type_nested(tensors)

        output = F.nested_to_list(nested_tensors)
        assert isinstance(output, list)
        assert len(output) == 5
        for t, o in zip(tensors, output):
            assert t is o

    @pytest.mark.parametrize(
        "make_input,type_nested",
        [
            (make_image, ImageNestedTensors),
            (make_video, VideoNestedTensors),
            (make_bounding_boxes, BoundingBoxesNestedTensors),
            (make_detection_masks, MaskNestedTensors),
            (make_labels, LabelsNestedTensors),
            (make_image_tensor, NestedTensors),
        ],
    )
    def test_transform(self, make_input, type_nested):
        tensor = make_input()
        tensors = [tensor] * 5
        nested_tensors = type_nested(tensors)

        transform = transforms.NestedToList()
        output = transform(nested_tensors)
        assert isinstance(output, list)
        assert len(output) == 5
        for t, o in zip(tensors, output):
            assert t is o


class TestNestedToBatch:
    @pytest.mark.parametrize(
        "make_input,type_nested,fn",
        [
            (make_image, ImageNestedTensors, None),
            (make_video, VideoNestedTensors, None),
            (make_bounding_boxes, BoundingBoxesNestedTensors, convert_bboxes_to_batch_bboxes),
            (make_detection_masks, MaskNestedTensors, convert_masks_to_batch_masks),
            (make_labels, LabelsNestedTensors, convert_labels_to_batch_labels),
            (make_image_tensor, NestedTensors, None),
        ],
    )
    def test_functional(self, make_input, type_nested, fn):
        tensors = [make_input() for _ in range(5)]
        nested_tensors = type_nested(tensors)

        result = F.nested_to_batch(nested_tensors)
        assert isinstance(result, type_nested.batch_tensors_type)
        if fn is not None:
            result_fn = fn(tensors)
            assert_equal(result.data, result_fn.data)
            assert result.samples_ranges == result_fn.samples_ranges
            if isinstance(result, BatchBoundingBoxes):
                assert result.canvas_size == result_fn.canvas_size
                assert result.format == result_fn.format

    @pytest.mark.parametrize(
        "make_input,type_nested,fn",
        [
            (make_image, ImageNestedTensors, None),
            (make_video, VideoNestedTensors, None),
            (make_bounding_boxes, BoundingBoxesNestedTensors, convert_bboxes_to_batch_bboxes),
            (make_detection_masks, MaskNestedTensors, convert_masks_to_batch_masks),
            (make_labels, LabelsNestedTensors, convert_labels_to_batch_labels),
            (make_image_tensor, NestedTensors, None),
        ],
    )
    def test_transform(self, make_input, type_nested, fn):
        tensors = [make_input() for _ in range(5)]
        nested_tensors = type_nested(tensors)

        transform = transforms.NestedToBatch()
        result = transform(nested_tensors)
        assert isinstance(result, type_nested.batch_tensors_type)
        if fn is not None:
            result_fn = fn(tensors)
            assert_equal(result.data, result_fn.data)
            assert result.samples_ranges == result_fn.samples_ranges
            if isinstance(result, BatchBoundingBoxes):
                assert result.canvas_size == result_fn.canvas_size
                assert result.format == result_fn.format
