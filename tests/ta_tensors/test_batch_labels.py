import pytest
import torch

from torchaug.ta_tensors import BatchLabels, Labels, convert_batch_labels_to_labels, convert_labels_to_batch_labels


class TestBatchLabels:
    @pytest.fixture
    def batch_labels(self):
        data = torch.tensor([[[1, 1, 0], [0, 1, 1]], [[2, 2, 0], [0, 2, 2]]])
        samples_ranges = [(0, 1), (1, 2)]
        return BatchLabels(data, samples_ranges=samples_ranges)

    def test_batch_size(self, batch_labels):
        assert batch_labels.batch_size == 2

    def test_num_data(self, batch_labels):
        assert batch_labels.num_data == 2

    def test_cat(self, batch_labels):
        batch_labels_1 = BatchLabels(
            torch.tensor([[[3, 3, 0], [0, 3, 3]], [[4, 4, 0], [0, 4, 4]]]), samples_ranges=[(0, 2), (2, 2)]
        )
        batch_labels_2 = BatchLabels(torch.tensor([[[5, 5, 0], [0, 5, 5]]]), samples_ranges=[(0, 1)])

        result = BatchLabels.cat([batch_labels, batch_labels_1, batch_labels_2])

        expected_data = torch.tensor(
            [
                [[1, 1, 0], [0, 1, 1]],
                [[2, 2, 0], [0, 2, 2]],
                [[3, 3, 0], [0, 3, 3]],
                [[4, 4, 0], [0, 4, 4]],
                [[5, 5, 0], [0, 5, 5]],
            ]
        )
        expected_samples_ranges = [(0, 1), (1, 2), (2, 4), (4, 4), (4, 5)]

        assert torch.all(torch.eq(result.data, expected_data))
        assert result.samples_ranges == expected_samples_ranges

    def test_get_sample(self, batch_labels):
        sample_0 = batch_labels.get_sample(0)
        sample_1 = batch_labels.get_sample(1)

        expected_sample_0 = torch.tensor([[[1, 1, 0], [0, 1, 1]]])
        expected_sample_1 = torch.tensor([[[2, 2, 0], [0, 2, 2]]])

        assert torch.equal(sample_0.data, expected_sample_0)
        assert torch.equal(sample_1.data, expected_sample_1)

    def test_get_chunk(self, batch_labels):
        chunk_indices = torch.tensor([1, 0])

        result = batch_labels.get_chunk(chunk_indices)

        expected_data = torch.tensor([[[2, 2, 0], [0, 2, 2]], [[1, 1, 0], [0, 1, 1]]])
        expected_samples_ranges = [(0, 1), (1, 2)]

        assert torch.equal(result.data, expected_data)
        assert result.samples_ranges == expected_samples_ranges

    def test_update_chunk_(self, batch_labels):
        chunk_indices = torch.tensor([0])
        chunk = BatchLabels(torch.tensor([[[6, 6, 0], [0, 6, 6]]]), samples_ranges=[(0, 1)])

        result = batch_labels.update_chunk_(chunk, chunk_indices)

        expected_data = torch.tensor([[[6, 6, 0], [0, 6, 6]], [[2, 2, 0], [0, 2, 2]]])
        expected_samples_ranges = [(0, 1), (1, 2)]

        assert torch.equal(result.data, expected_data)
        assert result.samples_ranges == expected_samples_ranges

    def test_masked_select(self, batch_labels):
        mask = torch.tensor([True, False])

        result = BatchLabels.masked_select(batch_labels, mask)

        expected_data = torch.tensor([[[1, 1, 0], [0, 1, 1]]])
        expected_samples_ranges = [(0, 1), (1, 1)]

        assert torch.equal(result.data, expected_data)
        assert result.samples_ranges == expected_samples_ranges


def test_convert_labels_to_batch_labels():
    labels = [
        Labels(torch.tensor([[[1, 1], [1, 1]]])),
        Labels(torch.tensor([[[0, 0], [0, 0]]])),
        Labels(torch.empty((0, 2, 2)), dtype=torch.int64),
        Labels(torch.tensor([[[0, 1], [1, 1]], [[0, 1], [1, 1]]])),
    ]
    expected_labels_data = torch.tensor([[[1, 1], [1, 1]], [[0, 0], [0, 0]], [[0, 1], [1, 1]], [[0, 1], [1, 1]]])
    expected_samples_ranges = [(0, 1), (1, 2), (2, 2), (2, 4)]

    batch_labels = convert_labels_to_batch_labels(labels)

    assert torch.allclose(batch_labels.data, expected_labels_data)
    assert batch_labels.samples_ranges == expected_samples_ranges
    assert isinstance(batch_labels, BatchLabels)
    assert batch_labels.batch_size == 4
    assert batch_labels.num_data == 4


def test_convert_batch_labels_to_labels():
    batch_labels = BatchLabels(
        torch.tensor(
            [
                [[1, 1, 0, 0], [0, 1, 1, 0]],
                [[0, 0, 1, 1], [1, 0, 1, 0]],
                [[1, 0, 1, 0], [0, 1, 0, 1]],
                [[1, 1, 1, 1], [0, 0, 0, 0]],
            ]
        ),
        samples_ranges=[(0, 2), (2, 2), (2, 4)],
    )

    list_labels = convert_batch_labels_to_labels(batch_labels)

    assert len(list_labels) == 3
    assert isinstance(list_labels[0], Labels)
    assert isinstance(list_labels[1], Labels)
    assert isinstance(list_labels[2], Labels)
    assert torch.equal(list_labels[0].data, torch.tensor([[[1, 1, 0, 0], [0, 1, 1, 0]], [[0, 0, 1, 1], [1, 0, 1, 0]]]))
    assert list(list_labels[1].shape) == [0, 2, 4]
    assert torch.equal(list_labels[2].data, torch.tensor([[[1, 0, 1, 0], [0, 1, 0, 1]], [[1, 1, 1, 1], [0, 0, 0, 0]]]))
