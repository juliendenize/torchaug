import pytest
import torch

from torchaug.ta_tensors import BatchImages, Image


class TestBatchImages:
    def test_new(self):
        # Test creating a BatchImages instance
        data = torch.randn(4, 3, 32, 32)
        batch_images = BatchImages(data)

        assert isinstance(batch_images, BatchImages)
        assert torch.all(torch.eq(batch_images.data, data))

    def test_cat(self):
        # Test concatenating multiple BatchImages instances
        data1 = torch.randn(2, 3, 32, 32)
        data2 = torch.randn(3, 3, 32, 32)
        data3 = torch.randn(2, 3, 32, 32)

        batch_images1 = BatchImages(data1)
        batch_images2 = BatchImages(data2)
        batch_images3 = BatchImages(data3)

        expected_output_data = torch.cat([data1, data2, data3], 0)
        expected_output = BatchImages(expected_output_data)

        actual_output = BatchImages.cat([batch_images1, batch_images2, batch_images3])

        assert isinstance(actual_output, BatchImages)
        assert torch.all(torch.eq(actual_output.data, expected_output.data))

    def test_to_samples(self):
        # Test converting BatchImages to a list of Images
        data = torch.randn(4, 3, 32, 32)
        batch_images = BatchImages(data)

        samples = batch_images.to_samples()

        assert isinstance(samples, list)
        assert len(samples) == len(batch_images)
        assert all(isinstance(sample, Image) for sample in samples)
        assert all(torch.all(torch.eq(sample.data, image.data)) for sample, image in zip(samples, batch_images))
