import numpy as np
import pytest
import torch

import torchaug.transforms as transforms
import torchaug.transforms.functional as F
from torchaug import ta_tensors

from ..utils import (
    make_batch_images,
    make_image,
    make_image_tensor,
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
