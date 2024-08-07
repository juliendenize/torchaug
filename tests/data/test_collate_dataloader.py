import pytest
import torch

from torchaug.data.dataloader._collate import default_collate
from torchaug.ta_tensors import (
    BatchBoundingBoxes,
    BatchImages,
    BatchLabels,
    BatchMasks,
    BatchVideos,
    Image,
    Labels,
    Video,
    convert_bboxes_to_batch_bboxes,
    convert_labels_to_batch_labels,
    convert_masks_to_batch_masks,
)

from ..utils import (
    assert_equal,
    cpu_and_cuda,
    make_batch_bounding_boxes,
    make_batch_detection_masks,
    make_batch_images,
    make_batch_labels,
    make_batch_segmentation_masks,
    make_batch_videos,
    make_bounding_boxes,
    make_detection_masks,
    make_image,
    make_labels,
    make_segmentation_mask,
    make_video,
)


class TestDefaultCollate:
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    def test_image(self, device, dtype):
        images = [make_image((32, 32), dtype=dtype, device=device) for _ in range(4)]
        actual = default_collate(images)
        expected_output_data = torch.stack([image.data for image in images], 0)

        assert_equal(actual, expected_output_data)
        assert isinstance(actual, BatchImages)
        assert actual.device.type == device
        assert actual.dtype == dtype

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    def test_batch_images(self, device, dtype):
        images = [make_batch_images((32, 32), dtype=dtype, device=device) for _ in range(4)]
        actual = default_collate(images)
        expected_output_data = torch.cat([image.data for image in images], 0)

        assert_equal(actual, expected_output_data)
        assert isinstance(actual, BatchImages)
        assert actual.device.type == device
        assert actual.dtype == dtype

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    def test_video(self, device, dtype):
        videos = [make_video((32, 32), dtype=dtype, device=device) for _ in range(4)]
        actual = default_collate(videos)
        expected_output_data = torch.stack([video.data for video in videos], 0)

        assert_equal(actual, expected_output_data)
        assert isinstance(actual, BatchVideos)
        assert actual.device.type == device
        assert actual.dtype == dtype

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    def test_batch_videos(self, device, dtype):
        videos = [make_batch_videos((32, 32), dtype=dtype, device=device) for _ in range(4)]
        actual = default_collate(videos)
        expected_output_data = torch.cat([video.data for video in videos], 0)

        assert_equal(actual, expected_output_data)
        assert isinstance(actual, BatchVideos)
        assert actual.device.type == device
        assert actual.dtype == dtype

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_bounding_boxes", [make_bounding_boxes])
    def test_bounding_boxes(self, device, make_bounding_boxes):
        bounding_boxes = [
            make_bounding_boxes(device=device),
            make_bounding_boxes(device=device),
        ]

        actual = default_collate(bounding_boxes)
        expected_output = convert_bboxes_to_batch_bboxes(bounding_boxes)

        assert_equal(actual, expected_output)
        assert actual.device.type == device
        assert actual.canvas_size == expected_output.canvas_size
        assert actual.format == expected_output.format
        assert all(
            actual.samples_ranges[i] == expected_output.samples_ranges[i] for i in range(len(actual.samples_ranges))
        )
        assert isinstance(actual, BatchBoundingBoxes)

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_mask", [make_detection_masks, make_segmentation_mask])
    def test_mask(self, device, make_mask):
        masks = [
            make_mask(device=device),
            make_mask(device=device),
        ]

        actual = default_collate(masks)
        expected_output = convert_masks_to_batch_masks(masks)

        assert_equal(actual, expected_output)
        assert actual.device.type == device
        assert all(
            actual.samples_ranges[i] == expected_output.samples_ranges[i] for i in range(len(actual.samples_ranges))
        )
        assert isinstance(actual, BatchMasks)

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_labels(self, device):
        labels = [
            make_labels(device=device),
            make_labels(device=device),
        ]

        actual = default_collate(labels)
        expected_output = convert_labels_to_batch_labels(labels)

        assert_equal(actual, expected_output)
        assert actual.device.type == device
        assert all(
            actual.samples_ranges[i] == expected_output.samples_ranges[i] for i in range(len(actual.samples_ranges))
        )
        assert isinstance(actual, BatchLabels)

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_batch_masks", [make_batch_detection_masks, make_batch_segmentation_masks])
    def test_batch_masks(self, device, make_batch_masks):
        masks = [
            make_batch_masks(device=device),
            make_batch_masks(device=device),
        ]

        actual = default_collate(masks)
        expected_output_data = BatchMasks.cat(masks).data

        assert_equal(actual, expected_output_data)
        assert actual.device.type == device
        assert actual.samples_ranges == masks[0].samples_ranges + [
            (idx_start + masks[0].num_data, idx_end + masks[0].num_data)
            for idx_start, idx_end in masks[1].samples_ranges
        ]
        assert isinstance(actual, BatchMasks)

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_batch_labels(self, device):
        labels = [
            make_batch_labels(device=device),
            make_batch_labels(device=device),
        ]

        actual = default_collate(labels)
        expected_output_data = BatchLabels.cat(labels).data

        assert_equal(actual, expected_output_data)
        assert actual.device.type == device
        assert actual.samples_ranges == labels[0].samples_ranges + [
            (idx_start + labels[0].num_data, idx_end + labels[0].num_data)
            for idx_start, idx_end in labels[1].samples_ranges
        ]
        assert isinstance(actual, BatchLabels)

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_batch_bounding_boxes", [make_batch_bounding_boxes])
    def test_batch_bounding_boxes(self, device, make_batch_bounding_boxes):
        bounding_boxes_batches = [
            make_batch_bounding_boxes(device=device),
            make_batch_bounding_boxes(device=device),
        ]

        actual = default_collate(bounding_boxes_batches)
        expected_output_data = BatchBoundingBoxes.cat(bounding_boxes_batches).data

        assert_equal(actual, expected_output_data)
        assert actual.device.type == device
        assert actual.canvas_size == bounding_boxes_batches[0].canvas_size
        assert actual.format == bounding_boxes_batches[0].format
        assert actual.samples_ranges == bounding_boxes_batches[0].samples_ranges + [
            (idx_start + bounding_boxes_batches[0].num_data, idx_end + bounding_boxes_batches[0].num_data)
            for idx_start, idx_end in bounding_boxes_batches[1].samples_ranges
        ]
        assert isinstance(actual, BatchBoundingBoxes)

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_batch_list(self, device):
        sample = [
            Image(make_image(), device=device),
            Video(make_video(), device=device),
            make_bounding_boxes(device=device),
            make_detection_masks(device=device),
            make_segmentation_mask(device=device),
            "yolo",
            torch.randn(3, 32, 32, device=device),
        ]

        batch = [sample] * 4

        collated_batch = default_collate(batch)

        assert isinstance(collated_batch, list)
        assert len(collated_batch) == len(sample)
        assert type(collated_batch[0]) is BatchImages
        assert type(collated_batch[1]) is BatchVideos
        assert type(collated_batch[2]) is BatchBoundingBoxes
        assert type(collated_batch[3]) is BatchMasks
        assert type(collated_batch[4]) is BatchMasks
        assert type(collated_batch[5]) is tuple
        assert type(collated_batch[6]) is torch.Tensor

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_batch_dict(self, device):
        sample = {
            "image": Image(make_image(), device=device),
            "video": Video(make_video(), device=device),
            "bbox": make_bounding_boxes(device=device),
            "mask1": make_detection_masks(device=device),
            "mask2": make_segmentation_mask(device=device),
            "string": "yolo",
            "tensor": torch.randn(3, 32, 32, device=device),
        }

        batch = [sample] * 4

        collated_batch = default_collate(batch)

        assert isinstance(collated_batch, dict)
        assert len(collated_batch) == len(sample)
        assert type(collated_batch["image"]) is BatchImages
        assert type(collated_batch["video"]) is BatchVideos
        assert type(collated_batch["bbox"]) is BatchBoundingBoxes
        assert type(collated_batch["mask1"]) is BatchMasks
        assert type(collated_batch["mask2"]) is BatchMasks
        assert isinstance(collated_batch["string"], list)
        assert isinstance(collated_batch["tensor"], torch.Tensor)
