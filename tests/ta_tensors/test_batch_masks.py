import pytest
import torch

from torchaug.ta_tensors import BatchMasks, Mask, convert_batch_masks_to_masks, convert_masks_to_batch_masks


class TestBatchMasks:
    @pytest.fixture
    def batch_masks(self):
        data = torch.tensor([[[1, 1, 0], [0, 1, 1]], [[2, 2, 0], [0, 2, 2]]])
        range_samples = [(0, 1), (1, 2)]
        return BatchMasks(data, range_samples=range_samples)

    def test_batch_size(self, batch_masks):
        assert batch_masks.batch_size == 2

    def test_num_data(self, batch_masks):
        assert batch_masks.num_data == 2

    def test_cat(self, batch_masks):
        batch_masks_1 = BatchMasks(
            torch.tensor([[[3, 3, 0], [0, 3, 3]], [[4, 4, 0], [0, 4, 4]]]), range_samples=[(0, 2), (2, 2)]
        )
        batch_masks_2 = BatchMasks(torch.tensor([[[5, 5, 0], [0, 5, 5]]]), range_samples=[(0, 1)])

        result = BatchMasks.cat([batch_masks, batch_masks_1, batch_masks_2])

        expected_data = torch.tensor(
            [
                [[1, 1, 0], [0, 1, 1]],
                [[2, 2, 0], [0, 2, 2]],
                [[3, 3, 0], [0, 3, 3]],
                [[4, 4, 0], [0, 4, 4]],
                [[5, 5, 0], [0, 5, 5]],
            ]
        )
        expected_range_samples = [(0, 1), (1, 2), (2, 4), (4, 4), (4, 5)]

        assert torch.all(torch.eq(result.data, expected_data))
        assert result.range_samples == expected_range_samples

    def test_get_sample(self, batch_masks):
        sample_0 = batch_masks.get_sample(0)
        sample_1 = batch_masks.get_sample(1)

        expected_sample_0 = torch.tensor([[[1, 1, 0], [0, 1, 1]]])
        expected_sample_1 = torch.tensor([[[2, 2, 0], [0, 2, 2]]])

        assert torch.equal(sample_0.data, expected_sample_0)
        assert torch.equal(sample_1.data, expected_sample_1)

    def test_get_chunk(self, batch_masks):
        chunk_indices = torch.tensor([1, 0])

        result = batch_masks.get_chunk(chunk_indices)

        expected_data = torch.tensor([[[2, 2, 0], [0, 2, 2]], [[1, 1, 0], [0, 1, 1]]])
        expected_range_samples = [(0, 1), (1, 2)]

        assert torch.equal(result.data, expected_data)
        assert result.range_samples == expected_range_samples

    def test_update_chunk_(self, batch_masks):
        chunk_indices = torch.tensor([0])
        chunk = BatchMasks(torch.tensor([[[6, 6, 0], [0, 6, 6]]]), range_samples=[(0, 1)])

        result = batch_masks.update_chunk_(chunk, chunk_indices)

        expected_data = torch.tensor([[[6, 6, 0], [0, 6, 6]], [[2, 2, 0], [0, 2, 2]]])
        expected_range_samples = [(0, 1), (1, 2)]

        assert torch.equal(result.data, expected_data)
        assert result.range_samples == expected_range_samples

    def test_masked_select(self, batch_masks):
        mask = torch.tensor([True, False])

        result = BatchMasks.masked_select(batch_masks, mask)

        expected_data = torch.tensor([[[1, 1, 0], [0, 1, 1]]])
        expected_range_samples = [(0, 1), (1, 1)]

        assert torch.equal(result.data, expected_data)
        assert result.range_samples == expected_range_samples


def test_convert_masks_to_batch_masks():
    masks = [
        Mask(torch.tensor([[[1, 1], [1, 1]]])),
        Mask(torch.tensor([[0, 0], [0, 0]])),
        Mask(torch.empty((0, 2, 2)), dtype=torch.int64),
        Mask(torch.tensor([[[0, 1], [1, 1]], [[0, 1], [1, 1]]])),
    ]

    expected_masks_data = torch.tensor([[[1, 1], [1, 1]], [[0, 0], [0, 0]], [[0, 1], [1, 1]], [[0, 1], [1, 1]]])
    expected_range_samples = [(0, 1), (1, 2), (2, 2), (2, 4)]

    batch_masks = convert_masks_to_batch_masks(masks)

    assert torch.allclose(batch_masks.data, expected_masks_data)
    assert batch_masks.range_samples == expected_range_samples
    assert isinstance(batch_masks, BatchMasks)
    assert batch_masks.batch_size == 4
    assert batch_masks.num_data == 4


def test_convert_batch_masks_to_masks():
    batch_masks = BatchMasks(
        torch.tensor(
            [
                [[1, 1, 0, 0], [0, 1, 1, 0]],
                [[0, 0, 1, 1], [1, 0, 1, 0]],
                [[1, 0, 1, 0], [0, 1, 0, 1]],
                [[1, 1, 1, 1], [0, 0, 0, 0]],
            ]
        ),
        range_samples=[(0, 2), (2, 2), (2, 4)],
    )

    list_masks = convert_batch_masks_to_masks(batch_masks)

    assert len(list_masks) == 3
    assert isinstance(list_masks[0], Mask)
    assert isinstance(list_masks[1], Mask)
    assert isinstance(list_masks[2], Mask)
    assert torch.equal(list_masks[0].data, torch.tensor([[[1, 1, 0, 0], [0, 1, 1, 0]], [[0, 0, 1, 1], [1, 0, 1, 0]]]))
    assert list(list_masks[1].shape) == [0, 2, 4]
    assert torch.equal(list_masks[2].data, torch.tensor([[[1, 0, 1, 0], [0, 1, 0, 1]], [[1, 1, 1, 1], [0, 0, 0, 0]]]))
