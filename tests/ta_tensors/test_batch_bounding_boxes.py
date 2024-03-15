import pytest
import torch

from torchaug.ta_tensors import (
    BatchBoundingBoxes,
    BoundingBoxes,
    BoundingBoxFormat,
    convert_batch_bboxes_to_bboxes,
    convert_bboxes_to_batch_bboxes,
)


class TestBatchBoundingBoxes:
    def test_batch_size(self):
        bbox = BatchBoundingBoxes(
            torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]),
            format=BoundingBoxFormat.XYWH,
            canvas_size=(100, 100),
            samples_ranges=[(0, 1), (1, 2), (2, 3)],
        )

        assert bbox.batch_size == 3

    def test_num_data(self):
        bbox = BatchBoundingBoxes(
            torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]),
            format=BoundingBoxFormat.XYWH,
            canvas_size=(100, 100),
            samples_ranges=[(0, 1), (1, 2), (2, 3)],
        )

        assert bbox.num_data == 3

    def test_cat(self):
        bbox1 = BatchBoundingBoxes(
            torch.tensor([[0, 0, 10, 10]]), format=BoundingBoxFormat.XYWH, canvas_size=(100, 100), samples_ranges=[(0, 1)]
        )
        bbox2 = BatchBoundingBoxes(
            torch.tensor([[20, 20, 30, 30]]),
            format=BoundingBoxFormat.XYWH,
            canvas_size=(100, 100),
            samples_ranges=[(0, 1)],
        )
        bbox3 = BatchBoundingBoxes(
            torch.tensor([[40, 40, 50, 50]]),
            format=BoundingBoxFormat.XYWH,
            canvas_size=(100, 100),
            samples_ranges=[(0, 1)],
        )

        cat_bbox = BatchBoundingBoxes.cat([bbox1, bbox2, bbox3])

        assert isinstance(cat_bbox, BatchBoundingBoxes)
        assert torch.all(torch.eq(cat_bbox.data, torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]])))
        assert cat_bbox.format == BoundingBoxFormat.XYWH
        assert cat_bbox.canvas_size == (100, 100)
        assert cat_bbox.samples_ranges == [(0, 1), (1, 2), (2, 3)]

    def test_get_sample(self):
        bbox = BatchBoundingBoxes(
            torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]),
            format=BoundingBoxFormat.XYWH,
            canvas_size=(100, 100),
            samples_ranges=[(0, 1), (1, 2), (2, 3)],
        )

        sample_bbox = bbox.get_sample(1)

        assert isinstance(sample_bbox, BoundingBoxes)
        assert torch.all(torch.eq(sample_bbox.data, torch.tensor([[20, 20, 30, 30]])))
        assert sample_bbox.format == BoundingBoxFormat.XYWH
        assert sample_bbox.canvas_size == (100, 100)

    def test_get_chunk(self):
        bbox = BatchBoundingBoxes(
            torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]),
            format=BoundingBoxFormat.XYWH,
            canvas_size=(100, 100),
            samples_ranges=[(0, 1), (1, 2), (2, 3)],
        )

        chunk_indices = torch.tensor([0, 2])
        chunk_bbox = bbox.get_chunk(chunk_indices)

        assert isinstance(chunk_bbox, BatchBoundingBoxes)
        assert torch.all(torch.eq(chunk_bbox.data, torch.tensor([[0, 0, 10, 10], [40, 40, 50, 50]])))
        assert chunk_bbox.format == BoundingBoxFormat.XYWH
        assert chunk_bbox.canvas_size == (100, 100)
        assert chunk_bbox.samples_ranges == [(0, 1), (1, 2)]

    def test_update_chunk(self):
        # Create a batch of bounding boxes
        batch = BatchBoundingBoxes(
            torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]]),
            format="xywh",
            canvas_size=(100, 100),
            samples_ranges=[(0, 1), (1, 2)],
        )

        # Create a chunk of bounding boxes to update
        chunk = BatchBoundingBoxes([[50, 50, 60, 60]], format="xywh", canvas_size=(100, 100), samples_ranges=[(0, 1)])

        # Get the indices of the chunk to update
        chunk_indices = torch.tensor([0])

        # Update the chunk in the batch
        updated_batch = batch.update_chunk_(chunk, chunk_indices)

        # Verify that the format and canvas size are the same
        assert updated_batch.format == batch.format
        assert updated_batch.canvas_size == batch.canvas_size

        # Verify that the chunk was updated correctly
        expected_boxes = torch.tensor([[50, 50, 60, 60], [30, 30, 40, 40]])
        assert torch.all(torch.eq(updated_batch, expected_boxes))
        assert updated_batch.samples_ranges == [(0, 1), (1, 2)]
        assert batch is updated_batch

    def test_masked_select(self):
        bbox = BatchBoundingBoxes(
            torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]),
            format=BoundingBoxFormat.XYWH,
            canvas_size=(100, 100),
            samples_ranges=[(0, 1), (1, 2), (2, 3)],
        )
        mask = torch.tensor([True, False, True])

        new_bbox = BatchBoundingBoxes.masked_select(bbox, mask)
        assert isinstance(new_bbox, BatchBoundingBoxes)
        assert torch.all(torch.eq(new_bbox.data, torch.tensor([[0, 0, 10, 10], [40, 40, 50, 50]])))
        assert new_bbox.format == BoundingBoxFormat.XYWH
        assert new_bbox.canvas_size == (100, 100)
        assert new_bbox.samples_ranges == [(0, 1), (1, 1), (1, 2)]


def test_convert_batch_bboxes_to_bboxes():
    # Create a BatchBoundingBoxes object
    bboxes = BatchBoundingBoxes(
        torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
        canvas_size=(100, 100),
        format="xywh",
        samples_ranges=[(0, 2), (2, 2)],
    )

    # Call the function to convert BatchBoundingBoxes to a list of BoundingBoxes
    list_bboxes = convert_batch_bboxes_to_bboxes(bboxes)

    # Check the output
    assert len(list_bboxes) == 2
    assert isinstance(list_bboxes[0], BoundingBoxes)
    assert isinstance(list_bboxes[1], BoundingBoxes)

    assert list_bboxes[0].canvas_size == (100, 100)
    assert list_bboxes[0].format == BoundingBoxFormat.XYWH
    assert torch.equal(list_bboxes[0].data, torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]))
    assert list(list_bboxes[1].shape) == [0, 4]


def test_convert_bboxes_to_batch_bboxes():
    # Create a list of BoundingBoxes objects
    bboxes = [
        BoundingBoxes(
            torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]), canvas_size=(100, 100), format="xywh"
        ),
        BoundingBoxes(
            torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]), canvas_size=(100, 100), format="xywh"
        ),
        BoundingBoxes(torch.empty((0, 4)), canvas_size=(100, 100), format="xywh"),
    ]

    # Call the function under test
    batch_bboxes = convert_bboxes_to_batch_bboxes(bboxes)

    # Assert the output is of type BatchBoundingBoxes
    assert isinstance(batch_bboxes, BatchBoundingBoxes)

    # Assert the attributes of BatchBoundingBoxes are correct
    assert torch.equal(
        batch_bboxes.data,
        torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]),
    )
    assert batch_bboxes.canvas_size == (100, 100)
    assert batch_bboxes.format == BoundingBoxFormat.XYWH
    assert batch_bboxes.samples_ranges == [(0, 2), (2, 4), (4, 4)]
    assert batch_bboxes.batch_size == 3
    assert batch_bboxes.num_data == 4
    assert batch_bboxes.get_num_data_sample(0) == 2
    assert batch_bboxes.get_num_data_sample(1) == 2
    assert batch_bboxes.get_num_data_sample(2) == 0

    # Assert ValueError is raised when bounding boxes have different attributes
    bboxes = [
        BoundingBoxes(
            torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]), canvas_size=(100, 100), format="xywh"
        ),
        BoundingBoxes(
            torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]),
            canvas_size=(200, 200),  # Different canvas size
            format="xywh",
        ),
    ]

    with pytest.raises(ValueError):
        convert_bboxes_to_batch_bboxes(bboxes)
