import functools
import itertools
import math
import re

import numpy as np
import pytest
import torch
import torchaug.transforms as transforms
import torchaug.transforms.functional as F
import torchvision
from torch import nn
from torchaug import ta_tensors
from torchaug.transforms.functional._utils import is_pure_tensor
from torchvision.transforms.v2._utils import check_type

from ...utils import (
    assert_equal,
    BOUNDING_BOXES_MAKERS,
    check_batch_transform,
    check_functional,
    check_functional_kernel_signature_match,
    check_kernel,
    check_transform,
    cpu_and_cuda,
    freeze_rng_state,
    IMAGE_MAKERS,
    make_batch_bounding_boxes,
    make_batch_images,
    make_batch_images_tensor,
    make_batch_videos,
    make_bounding_boxes,
    make_image,
    make_image_tensor,
    make_video,
    needs_cuda,
    transform_cls_to_functional,
    VIDEO_MAKERS,
)


class TestToImage:
    @pytest.mark.parametrize(
        "make_input", [make_image_tensor, make_image, make_batch_images]
    )
    @pytest.mark.parametrize(
        "fn", [F.to_image, transform_cls_to_functional(transforms.ToImage)]
    )
    def test_functional_and_transform(self, make_input, fn):
        input = make_input()
        output = fn(input)

        assert isinstance(output, ta_tensors.Image)

        input_size = (
            list(input.shape[:2])
            if isinstance(input, np.ndarray)
            else F.get_size(input)
        )
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
    @pytest.mark.parametrize(
        "fn", [F.to_batch_images, transform_cls_to_functional(transforms.ToBatchImages)]
    )
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
