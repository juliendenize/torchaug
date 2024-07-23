import pytest
import torch

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
    set_return_type,
)

from ..utils import (
    assert_equal,
    cpu_and_cuda,
    make_bounding_boxes,
    make_detection_masks,
    make_image,
    make_image_tensor,
    make_labels,
    make_video,
)


class TestTANestedTensors:
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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_init(self, make_input, type_nested, device):
        tensor = make_input(device=device)
        tensors = [tensor] * 5
        nested_tensors = type_nested(tensors)
        assert len(nested_tensors) == 5
        assert nested_tensors.shapes == [tensor.shape] * 5
        assert nested_tensors.device == tensors[0].device
        assert nested_tensors.dtype == tensors[0].dtype

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_add(self, make_input, type_nested, device):
        # Test 1: Add nested tensors
        tensors1 = [make_input(device=device) for _ in range(5)]
        tensors2 = [make_input(device=device) for _ in range(5)]
        nested_tensors1 = type_nested(tensors1)
        nested_tensors2 = type_nested(tensors2)
        print(nested_tensors1)
        result = nested_tensors1 + nested_tensors2
        assert isinstance(result, type_nested)
        assert len(result) == 5
        for tensor1, tensor2, tensor in zip(tensors1, tensors2, result.tensors):
            assert_equal(tensor, tensor1 + tensor2)

        # Test 2: Add nested tensors with tensor
        tensor = make_input(device=device)
        result = nested_tensors1 + tensor
        assert isinstance(result, type_nested)
        assert len(result) == 5
        for tensor1, result_tensor in zip(tensors1, result.tensors):
            assert_equal(result_tensor, tensor1 + tensor)

        # Test 3: Add nested tensors with number
        scalar_result = nested_tensors1 + 1.0
        assert isinstance(scalar_result, type_nested)
        assert len(scalar_result) == 5
        for result_tensor, tensor1 in zip(scalar_result.tensors, tensors1):
            assert_equal(result_tensor, tensor1 + 1.0)

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_add_(self, make_input, type_nested, device):
        # Test 1: Add nested tensors
        tensors1 = [make_input(device=device) for _ in range(5)]
        tensors2 = [make_input(device=device) for _ in range(5)]
        nested_tensors1 = type_nested(tensors1)
        nested_tensors2 = type_nested(tensors2)
        results = nested_tensors1.clone()
        results.add_(nested_tensors2)
        assert len(nested_tensors1) == 5
        for tensor1, tensor2, tensor in zip(tensors1, tensors2, results.tensors):
            assert_equal(tensor, tensor1 + tensor2)

        # Test 2: Add nested tensors with tensor
        tensor = make_input(device=device)
        results = nested_tensors1.clone()
        results.add_(tensor)
        assert len(nested_tensors1) == 5
        for tensor1, result_tensor in zip(tensors1, results.tensors):
            assert_equal(result_tensor, tensor1 + tensor)

        # Test 3: Add nested tensors with number
        results = nested_tensors1.clone()
        if tensors1[0].is_floating_point():
            number = 1.0
        elif tensors1[0].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            number = 1
        elif tensors1[0].dtype == torch.bool:
            number = True
        else:
            raise ValueError(f"Unsupported dtype {tensors1[0].dtype}")
        results.add_(number)
        assert len(nested_tensors1) == 5
        for result_tensor, tensor1 in zip(results.tensors, tensors1):
            assert_equal(result_tensor, tensor1 + number)

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_mul(self, make_input, type_nested, device):
        # Test 1: Multiply nested tensors
        tensors1 = [make_input(device=device) for _ in range(5)]
        tensors2 = [make_input(device=device) for _ in range(5)]
        nested_tensors1 = type_nested(tensors1)
        nested_tensors2 = type_nested(tensors2)
        result = nested_tensors1 * nested_tensors2
        assert isinstance(result, type_nested)
        assert len(result) == 5
        for tensor1, tensor2, tensor in zip(tensors1, tensors2, result.tensors):
            assert_equal(tensor, tensor1 * tensor2)

        # Test 2: Multiply nested tensors with tensor
        tensor = make_input(device=device)
        result = nested_tensors1 * tensor
        assert isinstance(result, type_nested)
        assert len(result) == 5
        for tensor1, result_tensor in zip(tensors1, result.tensors):
            assert_equal(result_tensor, tensor1 * tensor)

        # Test 3: Multiply nested tensors with number
        scalar_result = nested_tensors1 * 2.0
        assert isinstance(scalar_result, type_nested)
        assert len(scalar_result) == 5
        for result_tensor, tensor1 in zip(scalar_result.tensors, tensors1):
            assert_equal(result_tensor, tensor1 * 2.0)

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_mul_(self, make_input, type_nested, device):
        # Test 1: Multiply nested tensors
        tensors1 = [make_input(device=device) for _ in range(5)]
        tensors2 = [make_input(device=device) for _ in range(5)]
        nested_tensors1 = type_nested(tensors1)
        nested_tensors2 = type_nested(tensors2)
        results = nested_tensors1.clone()
        results.mul_(nested_tensors2)
        assert len(nested_tensors1) == 5
        for tensor1, tensor2, tensor in zip(tensors1, tensors2, results.tensors):
            assert_equal(tensor, tensor1 * tensor2)

        # Test 2: Multiply nested tensors with tensor
        tensor = make_input(device=device)
        results = nested_tensors1.clone()
        results.mul_(tensor)
        assert len(nested_tensors1) == 5
        for tensor1, result_tensor in zip(tensors1, results.tensors):
            assert_equal(result_tensor, tensor1 * tensor)

        # Test 3: Multiply nested tensors with number
        results = nested_tensors1.clone()
        if tensors1[0].is_floating_point():
            number = 2.0
        elif tensors1[0].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            number = 2
        elif tensors1[0].dtype == torch.bool:
            number = True
        else:
            raise ValueError(f"Unsupported dtype {tensors1[0].dtype}")
        results.mul_(number)
        assert len(nested_tensors1) == 5
        for result_tensor, tensor1 in zip(results.tensors, tensors1):
            assert_equal(result_tensor, tensor1 * number)

    @pytest.mark.parametrize(
        "make_input,type_nested",
        [
            (make_image, ImageNestedTensors),
            (make_video, VideoNestedTensors),
            (make_bounding_boxes, BoundingBoxesNestedTensors),
            (make_labels, LabelsNestedTensors),
            (make_image_tensor, NestedTensors),
        ],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_sub(self, make_input, type_nested, device):
        # Test 1: Subtract nested tensors
        tensors1 = [make_input(device=device) for _ in range(5)]
        tensors2 = [make_input(device=device) for _ in range(5)]
        nested_tensors1 = type_nested(tensors1)
        nested_tensors2 = type_nested(tensors2)
        result = nested_tensors1 - nested_tensors2
        assert isinstance(result, type_nested)
        assert len(result) == 5
        for tensor1, tensor2, tensor in zip(tensors1, tensors2, result.tensors):
            assert_equal(tensor, tensor1 - tensor2)

        # Test 2: Subtract nested tensors with tensor
        tensor = make_input(device=device)
        result = nested_tensors1 - tensor
        assert isinstance(result, type_nested)
        assert len(result) == 5
        for tensor1, result_tensor in zip(tensors1, result.tensors):
            assert_equal(result_tensor, tensor1 - tensor)

        # Test 3: Subtract nested tensors with number
        scalar_result = nested_tensors1 - 1.0
        assert isinstance(scalar_result, type_nested)
        assert len(scalar_result) == 5
        for result_tensor, tensor1 in zip(scalar_result.tensors, tensors1):
            assert_equal(result_tensor, tensor1 - 1.0)

    @pytest.mark.parametrize(
        "make_input,type_nested",
        [
            (make_image, ImageNestedTensors),
            (make_video, VideoNestedTensors),
            (make_bounding_boxes, BoundingBoxesNestedTensors),
            (make_labels, LabelsNestedTensors),
            (make_image_tensor, NestedTensors),
        ],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_sub_(self, make_input, type_nested, device):
        # Test 1: Subtract nested tensors
        tensors1 = [make_input(device=device) for _ in range(5)]
        tensors2 = [make_input(device=device) for _ in range(5)]
        nested_tensors1 = type_nested(tensors1)
        nested_tensors2 = type_nested(tensors2)
        results = nested_tensors1.clone()
        results.sub_(nested_tensors2)
        assert len(nested_tensors1) == 5
        for tensor1, tensor2, tensor in zip(tensors1, tensors2, results.tensors):
            assert_equal(tensor, tensor1 - tensor2)

        # Test 2: Subtract nested tensors with tensor
        tensor = make_input(device=device)
        results = nested_tensors1.clone()
        results.sub_(tensor)
        assert len(nested_tensors1) == 5
        for tensor1, result_tensor in zip(tensors1, results.tensors):
            assert_equal(result_tensor, tensor1 - tensor)

        # Test 3: Subtract nested tensors with number
        results = nested_tensors1.clone()
        if tensors1[0].is_floating_point():
            number = 1.0
        elif tensors1[0].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            number = 1
        elif tensors1[0].dtype == torch.bool:
            number = True
        else:
            raise ValueError(f"Unsupported dtype {tensors1[0].dtype}")
        results.sub_(number)
        assert len(nested_tensors1) == 5
        for result_tensor, tensor1 in zip(results.tensors, tensors1):
            assert_equal(result_tensor, tensor1 - number)

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_div(self, make_input, type_nested, device):
        # Test 1: Divide nested tensors
        with set_return_type("TATensor"):
            tensors1 = [make_input(device=device) + 1 for _ in range(5)]
            tensors2 = [make_input(device=device) + 1 for _ in range(5)]
        nested_tensors1 = type_nested(tensors1)
        nested_tensors2 = type_nested(tensors2)
        result = nested_tensors1 / nested_tensors2
        assert isinstance(result, type_nested)
        assert len(result) == 5
        for tensor1, tensor2, tensor in zip(tensors1, tensors2, result.tensors):
            assert_equal(tensor, tensor1 / tensor2)

        # Test 2: Divide nested tensors with tensor
        with set_return_type("TATensor"):
            tensor = make_input(device=device) + 1
        result = nested_tensors1 / tensor
        assert isinstance(result, type_nested)
        assert len(result) == 5
        for tensor1, result_tensor in zip(tensors1, result.tensors):
            assert_equal(result_tensor, tensor1 / tensor)

        # Test 3: Divide nested tensors with number
        scalar_result = nested_tensors1 / 2.0
        assert isinstance(scalar_result, type_nested)
        assert len(scalar_result) == 5
        for result_tensor, tensor1 in zip(scalar_result.tensors, tensors1):
            assert_equal(result_tensor, tensor1 / 2.0)

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_div_(self, make_input, type_nested, device):
        # Test 1: Divide nested tensors
        with set_return_type("TATensor"):
            tensors1 = [make_input(device=device) + 1 for _ in range(5)]
            tensors2 = [make_input(device=device) + 1 for _ in range(5)]
        nested_tensors1 = type_nested(tensors1).to(dtype=torch.float32)
        nested_tensors2 = type_nested(tensors2).to(dtype=torch.float32)
        results = nested_tensors1.clone()
        results.div_(nested_tensors2)
        assert len(nested_tensors1) == 5
        for tensor1, tensor2, tensor in zip(tensors1, tensors2, results.tensors):
            assert_equal(tensor, tensor1 / tensor2)

        # Test 2: Divide nested tensors with tensor
        with set_return_type("TATensor"):
            tensor = make_input(device=device) + 1
        results = nested_tensors1.clone()
        results.div_(tensor)
        assert len(nested_tensors1) == 5
        for tensor1, result_tensor in zip(tensors1, results.tensors):
            assert_equal(result_tensor, tensor1 / tensor)

        # Test 3: Divide nested tensors with number
        results = nested_tensors1.clone()
        if tensors1[0].is_floating_point():
            number = 2.0
        elif tensors1[0].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            number = 2
        elif tensors1[0].dtype == torch.bool:
            number = True
        else:
            raise ValueError(f"Unsupported dtype {tensors1[0].dtype}")
        results.div_(number)
        assert len(nested_tensors1) == 5
        for result_tensor, tensor1 in zip(results.tensors, tensors1):
            assert_equal(result_tensor, tensor1 / number)

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_floor_divide(self, make_input, type_nested, device):
        # Test 1: Floor divide nested tensors
        with set_return_type("TATensor"):
            tensors1 = [make_input(device=device) + 1 for _ in range(5)]
            tensors2 = [make_input(device=device) + 1 for _ in range(5)]
        nested_tensors1 = type_nested(tensors1)
        nested_tensors2 = type_nested(tensors2)
        result = nested_tensors1 // nested_tensors2
        assert isinstance(result, type_nested)
        assert len(result) == 5
        for tensor1, tensor2, tensor in zip(tensors1, tensors2, result.tensors):
            assert_equal(tensor, tensor1 // tensor2)

        # Test 2: Floor divide nested tensors with tensor
        with set_return_type("TATensor"):
            tensor = make_input(device=device) + 1
        result = nested_tensors1 // tensor
        assert isinstance(result, type_nested)
        assert len(result) == 5
        for tensor1, result_tensor in zip(tensors1, result.tensors):
            assert_equal(result_tensor, tensor1 // tensor)

        # Test 3: Floor divide nested tensors with number
        scalar_result = nested_tensors1 // 2.0
        assert isinstance(scalar_result, type_nested)
        assert len(scalar_result) == 5
        for result_tensor, tensor1 in zip(scalar_result.tensors, tensors1):
            assert_equal(result_tensor, tensor1 // 2.0)

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_floor_divide_(self, make_input, type_nested, device):
        # Test 1: Floor divide nested tensors
        with set_return_type("TATensor"):
            tensors1 = [make_input(device=device) + 1 for _ in range(5)]
            tensors2 = [make_input(device=device) + 1 for _ in range(5)]
        nested_tensors1 = type_nested(tensors1)
        nested_tensors2 = type_nested(tensors2)
        results = nested_tensors1.clone()
        results.floor_divide_(nested_tensors2)
        assert len(nested_tensors1) == 5
        for tensor1, tensor2, tensor in zip(tensors1, tensors2, results.tensors):
            assert_equal(tensor, tensor1 // tensor2)

        # Test 2: Floor divide nested tensors with tensor
        with set_return_type("TATensor"):
            tensor = make_input(device=device) + 1
        results = nested_tensors1.clone()
        results.floor_divide_(tensor)
        assert len(nested_tensors1) == 5
        for tensor1, result_tensor in zip(tensors1, results.tensors):
            assert_equal(result_tensor, tensor1 // tensor)

        # Test 3: Floor divide nested tensors with number
        results = nested_tensors1.clone()
        if tensors1[0].is_floating_point():
            number = 2.0
        elif tensors1[0].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            number = 2
        elif tensors1[0].dtype == torch.bool:
            number = True
        else:
            raise ValueError(f"Unsupported dtype {tensors1[0].dtype}")
        results.floor_divide_(number)
        assert len(nested_tensors1) == 5
        for result_tensor, tensor1 in zip(results.tensors, tensors1):
            assert_equal(result_tensor, tensor1 // number)

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_remainder(self, make_input, type_nested, device):
        # Test 1: Remainder nested tensors
        with set_return_type("TATensor"):
            tensors1 = [make_input(device=device) + 1 for _ in range(5)]
            tensors2 = [make_input(device=device) + 1 for _ in range(5)]
        nested_tensors1 = type_nested(tensors1)
        nested_tensors2 = type_nested(tensors2)
        result = nested_tensors1 % nested_tensors2
        assert isinstance(result, type_nested)
        assert len(result) == 5
        for tensor1, tensor2, tensor in zip(tensors1, tensors2, result.tensors):
            assert_equal(tensor, tensor1 % tensor2)

        # Test 2: Remainder nested tensors with tensor
        tensor = make_input(device=device)
        with set_return_type("TATensor"):
            tensor = make_input(device=device) + 1
        result = nested_tensors1 % tensor
        assert isinstance(result, type_nested)
        assert len(result) == 5
        for tensor1, result_tensor in zip(tensors1, result.tensors):
            assert_equal(result_tensor, tensor1 % tensor)

        # Test 3: Remainder nested tensors with number
        scalar_result = nested_tensors1 % 2.0
        assert isinstance(scalar_result, type_nested)
        assert len(scalar_result) == 5
        for result_tensor, tensor1 in zip(scalar_result.tensors, tensors1):
            assert_equal(result_tensor, tensor1 % 2.0)

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_remainder_(self, make_input, type_nested, device):
        # Test 1: Remainder nested tensors
        with set_return_type("TATensor"):
            tensors1 = [make_input(device=device) + 1 for _ in range(5)]
            tensors2 = [make_input(device=device) + 1 for _ in range(5)]
        nested_tensors1 = type_nested(tensors1)
        nested_tensors2 = type_nested(tensors2)
        results = nested_tensors1.clone()
        results.remainder_(nested_tensors2)
        assert len(nested_tensors1) == 5
        for tensor1, tensor2, tensor in zip(tensors1, tensors2, results.tensors):
            assert_equal(tensor, tensor1 % tensor2)

        # Test 2: Remainder nested tensors with tensor
        with set_return_type("TATensor"):
            tensor = make_input(device=device) + 1
        results = nested_tensors1.clone()
        results.remainder_(tensor)
        assert len(nested_tensors1) == 5
        for tensor1, result_tensor in zip(tensors1, results.tensors):
            assert_equal(result_tensor, tensor1 % tensor)

        # Test 3: Remainder nested tensors with number
        results = nested_tensors1.clone()
        if tensors1[0].is_floating_point():
            number = 2.0
        elif tensors1[0].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            number = 2
        elif tensors1[0].dtype == torch.bool:
            number = True
        else:
            raise ValueError(f"Unsupported dtype {tensors1[0].dtype}")
        results.remainder_(number)
        assert len(nested_tensors1) == 5
        for result_tensor, tensor1 in zip(results.tensors, tensors1):
            assert_equal(result_tensor, tensor1 % number)

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_size(self, make_input, type_nested, device):
        tensors = [make_input(device=device) for _ in range(5)]
        nested_tensors = type_nested(tensors)

        sizes = nested_tensors.size()
        assert isinstance(sizes, list)
        assert len(sizes) == 5
        assert sizes == [tensors[0].size()] * 5

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_to(self, make_input, type_nested, device, dtype):
        tensors = [make_input(device="cpu") for _ in range(5)]
        nested_tensors = type_nested(tensors)
        result = nested_tensors.to(device=device, dtype=dtype)
        assert isinstance(result, type_nested)
        assert len(result) == 5
        assert result.device.type == device
        assert result.dtype == dtype

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_to_batch(self, make_input, type_nested, fn, device):
        tensors = [make_input(device=device) for _ in range(5)]
        nested_tensors = type_nested(tensors)

        result = nested_tensors.to_batch()
        assert isinstance(result, type_nested.batch_tensors_type)
        if fn is not None:
            result_fn = fn(tensors)
            assert_equal(result.data, result_fn.data)
            assert result.samples_ranges == result_fn.samples_ranges
            if isinstance(result, BatchBoundingBoxes):
                assert result.canvas_size == result_fn.canvas_size
                assert result.format == result_fn.format

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_to_list(self, make_input, type_nested, device):
        tensors = [make_input(device=device) for _ in range(5)]
        nested_tensors = type_nested(tensors)
        result = nested_tensors.to_list()

        for tensor, result_tensor in zip(tensors, result):
            assert tensor is result_tensor
