import decimal
import functools
import itertools
import math
import random
import re

import numpy as np
import pytest
import torch
import torchaug.transforms as transforms
import torchaug.transforms.functional as F
import torchvision
import torchvision.transforms.v2.functional as TVF
from torch import nn
from torch.utils._pytree import tree_map
from torchaug import ta_tensors
from torchaug.transforms.functional._utils import is_pure_tensor
from torchvision.transforms.v2._utils import check_type

from ...utils import (
    ALL_IMAGES_MAKERS,
    assert_equal,
    BATCH_IMAGES_TENSOR_AND_MAKERS,
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
    make_batch_detection_masks,
    make_batch_images,
    make_batch_images_tensor,
    make_batch_videos,
    make_bounding_boxes,
    make_detection_masks,
    make_image,
    make_image_tensor,
    make_video,
    needs_cuda,
    transform_cls_to_functional,
    VIDEO_MAKERS,
)


class TestLambda:
    @pytest.mark.parametrize(
        "input", [object(), torch.empty(()), np.empty(()), "string", 1, 0.0]
    )
    @pytest.mark.parametrize("types", [(), (torch.Tensor, np.ndarray)])
    def test_transform(self, input, types):
        was_applied = False

        def was_applied_fn(input):
            nonlocal was_applied
            was_applied = True
            return input

        transform = transforms.Lambda(was_applied_fn, *types)
        output = transform(input)

        assert output is input
        assert was_applied is (not types or isinstance(input, types))


class TestLinearTransform:
    def _make_matrix_and_vector(self, input, *, device=None):
        device = device or input.device
        numel = math.prod(F.get_dimensions(input))
        transformation_matrix = torch.randn((numel, numel), device=device)
        mean_vector = torch.randn((numel,), device=device)
        return transformation_matrix, mean_vector

    def _make_batch_matrix_and_vector(self, input, *, device=None):
        device = device or input.device
        numel = math.prod(F.get_dimensions(input))
        transformation_matrix = torch.randn(
            (input.shape[0], numel, numel), device=device
        )
        mean_vector = torch.randn((input.shape[0], numel), device=device)
        return transformation_matrix, mean_vector

    def _sample_input_adapter(self, transform, input, device):
        return {key: value for key, value in input.items()}

    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image,
            make_video,
            make_batch_images,
            make_batch_videos,
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, dtype, device):
        input = make_input(dtype=dtype, device=device)
        check_transform(
            transforms.LinearTransformation(*self._make_matrix_and_vector(input)),
            input,
            check_sample_input=self._sample_input_adapter,
        )

    def test_transform_error(self):
        with pytest.raises(ValueError, match="transformation_matrix should be square"):
            transforms.LinearTransformation(
                transformation_matrix=torch.rand(2, 3), mean_vector=torch.rand(2)
            )

        with pytest.raises(ValueError, match="mean_vector should have the same length"):
            transforms.LinearTransformation(
                transformation_matrix=torch.rand(2, 2), mean_vector=torch.rand(1)
            )

        for matrix_dtype, vector_dtype in [
            (torch.float32, torch.float64),
            (torch.float64, torch.float32),
        ]:
            with pytest.raises(
                ValueError, match="Input tensors should have the same dtype"
            ):
                transforms.LinearTransformation(
                    transformation_matrix=torch.rand(2, 2, dtype=matrix_dtype),
                    mean_vector=torch.rand(2, dtype=vector_dtype),
                )

        image = make_image()
        transform = transforms.LinearTransformation(
            transformation_matrix=torch.rand(2, 2), mean_vector=torch.rand(2)
        )
        with pytest.raises(
            ValueError,
            match="Input tensor and transformation matrix have incompatible shape",
        ):
            transform(image)

    @needs_cuda
    def test_transform_error_cuda(self):
        for matrix_device, vector_device in [("cuda", "cpu"), ("cpu", "cuda")]:
            with pytest.raises(
                ValueError, match="Input tensors should be on the same device"
            ):
                transforms.LinearTransformation(
                    transformation_matrix=torch.rand(2, 2, device=matrix_device),
                    mean_vector=torch.rand(2, device=vector_device),
                )

        for input_device, param_device in [("cuda", "cpu"), ("cpu", "cuda")]:
            input = make_image(device=input_device)
            transform = transforms.LinearTransformation(
                *self._make_matrix_and_vector(input, device=param_device)
            )
            with pytest.raises(
                ValueError,
                match="Input tensor should be on the same device as transformation matrix and mean vector",
            ):
                transform(input)


class TestNormalize:
    MEANS_STDS = [
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    ]
    MEAN, STD = MEANS_STDS[0]

    @pytest.mark.parametrize(("mean", "std"), [*MEANS_STDS, (0.5, 2.0)])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, mean, std, device, make_input):
        check_kernel(
            F.normalize_image,
            make_input(dtype=torch.float32, device=device),
            mean=self.MEAN,
            std=self.STD,
        )

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image_inplace(self, device, make_input):
        input = make_input(dtype=torch.float32, device=device)
        input_version = input._version

        output_out_of_place = F.normalize_image(input, mean=self.MEAN, std=self.STD)
        assert output_out_of_place.data_ptr() != input.data_ptr()
        assert output_out_of_place is not input

        output_inplace = F.normalize_image(
            input, mean=self.MEAN, std=self.STD, inplace=True
        )
        assert output_inplace.data_ptr() == input.data_ptr()
        assert output_inplace._version > input_version

        # TODO investigate why this is not working
        # assert output_inplace is input

        assert_equal(output_inplace, output_out_of_place)

    @pytest.mark.parametrize("make_input", VIDEO_MAKERS)
    def test_kernel_video(self, make_input):
        check_kernel(
            F.normalize_video,
            make_video(dtype=torch.float32),
            mean=self.MEAN,
            std=self.STD,
        )

    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image,
            make_video,
            make_batch_images,
            make_batch_videos,
        ],
    )
    def test_functional(self, make_input):
        check_functional(
            F.normalize, make_input(dtype=torch.float32), mean=self.MEAN, std=self.STD
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.normalize_image, torch.Tensor),
            (F.normalize_image, ta_tensors.Image),
            (F.normalize_image, ta_tensors.BatchImages),
            (F.normalize_video, ta_tensors.Video),
            (F.normalize_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.normalize, kernel=kernel, input_type=input_type
        )

    def test_functional_error(self):
        with pytest.raises(TypeError, match="should be a float tensor"):
            F.normalize_image(
                make_image(dtype=torch.uint8), mean=self.MEAN, std=self.STD
            )

        with pytest.raises(ValueError, match="tensor image of size"):
            F.normalize_image(
                torch.rand(16, 16, dtype=torch.float32), mean=self.MEAN, std=self.STD
            )

        for std in [0, [0, 0, 0], [0, 1, 1]]:
            with pytest.raises(
                ValueError, match="std evaluated to zero, leading to division by zero"
            ):
                F.normalize_image(
                    make_image(dtype=torch.float32), mean=self.MEAN, std=std
                )

    def _sample_input_adapter(self, transform, input, device):
        adapted_input = {}
        for key, value in input.items():
            if check_type(value, (is_pure_tensor, ta_tensors.Image, ta_tensors.Video)):
                # normalize doesn't support integer images
                value = F.to_dtype(value, torch.float32, scale=True)
            adapted_input[key] = value
        return adapted_input

    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image,
            make_video,
            make_batch_images,
            make_batch_videos,
        ],
    )
    def test_transform(self, make_input):
        check_transform(
            transforms.Normalize(mean=self.MEAN, std=self.STD),
            make_input(dtype=torch.float32),
            check_sample_input=self._sample_input_adapter,
        )

    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image,
            make_video,
            make_batch_images,
            make_batch_videos,
        ],
    )
    def test_transform(self, make_input):
        check_transform(
            transforms.Normalize(mean=self.MEAN, std=self.STD),
            make_input(dtype=torch.float32),
            check_sample_input=self._sample_input_adapter,
        )

    def _reference_normalize_image(self, image, *, mean, std):
        image = image.numpy()
        mean, std = [
            np.array(stat, dtype=image.dtype).reshape((-1, 1, 1))
            for stat in [mean, std]
        ]
        return ta_tensors.Image((image - mean) / std)

    def _reference_normalize_batch_images(self, image, *, mean, std):
        image = image.numpy()
        mean, std = [
            np.array(stat, dtype=image.dtype).reshape((-1, 1, 1))
            for stat in [mean, std]
        ]
        return ta_tensors.BatchImages((image - mean) / std)

    @pytest.mark.parametrize(("mean", "std"), MEANS_STDS)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "fn", [F.normalize, transform_cls_to_functional(transforms.Normalize)]
    )
    def test_correctness_image(self, mean, std, dtype, fn):
        image = make_image(dtype=dtype)

        actual = fn(image, mean=mean, std=std)
        expected = self._reference_normalize_image(image, mean=mean, std=std)

        assert_equal(actual, expected)

    @pytest.mark.parametrize(("mean", "std"), MEANS_STDS)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "fn", [F.normalize, transform_cls_to_functional(transforms.Normalize)]
    )
    def test_correctness_batch_images(self, mean, std, dtype, fn):
        image = make_batch_images(dtype=dtype)

        actual = fn(image, mean=mean, std=std)
        expected = self._reference_normalize_batch_images(image, mean=mean, std=std)

        assert_equal(actual, expected)


class TestGaussianBlur:
    @pytest.mark.parametrize("kernel_size", [1, 3, (3, 1), [3, 5]])
    @pytest.mark.parametrize(
        "sigma", [None, 1.0, 1, (0.5,), [0.3], (0.3, 0.7), [0.9, 0.2]]
    )
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, kernel_size, sigma, make_input):
        check_kernel(
            F.gaussian_blur_image,
            make_input(),
            kernel_size=kernel_size,
            sigma=sigma,
            check_scripted_vs_eager=not (
                isinstance(kernel_size, int) or isinstance(sigma, (float, int))
            ),
        )

    @pytest.mark.parametrize("kernel_size", [1, 3, (3, 1), [3, 5]])
    @pytest.mark.parametrize(
        "sigma",
        [
            None,
            1.0,
            1,
            (0.5,),
            [0.3],
            (0.3, 0.7),
            [0.9, 0.2],
            torch.tensor([0.9, 0.2]),
            torch.tensor([[0.9, 0.5], [0.4, 0.5]]),
            torch.tensor([0.9]),
        ],
    )
    def test_batch_kernel_image(self, kernel_size, sigma):
        check_kernel(
            F.gaussian_blur_batch_images,
            make_batch_images(),
            kernel_size=kernel_size,
            sigma=sigma,
            check_scripted_vs_eager=not (
                isinstance(kernel_size, int)
                or isinstance(sigma, (list, tuple, float, int))
            ),
        )

    def test_kernel_image_errors(self):
        image = make_image_tensor()

        with pytest.raises(
            ValueError, match="kernel_size is a sequence its length should be 2"
        ):
            F.gaussian_blur_image(image, kernel_size=[1, 2, 3])

        for kernel_size in [2, -1]:
            with pytest.raises(
                ValueError, match="kernel_size should have odd and positive integers"
            ):
                F.gaussian_blur_image(image, kernel_size=kernel_size)

        with pytest.raises(
            ValueError, match="sigma is a sequence, its length should be 2"
        ):
            F.gaussian_blur_image(image, kernel_size=1, sigma=[1, 2, 3])

        with pytest.raises(
            TypeError, match="sigma should be either float or sequence of floats"
        ):
            F.gaussian_blur_image(image, kernel_size=1, sigma=object())

        with pytest.raises(ValueError, match="sigma should have positive values"):
            F.gaussian_blur_image(image, kernel_size=1, sigma=-1)

    def test_batch_kernel_image_errors(self):
        image = make_batch_images_tensor()

        with pytest.raises(
            ValueError, match="kernel_size is a sequence its length should be 2"
        ):
            F.gaussian_blur_batch_images(image, kernel_size=[1, 2, 3])

        for kernel_size in [2, -1]:
            with pytest.raises(
                ValueError, match="kernel_size should have odd and positive integers"
            ):
                F.gaussian_blur_batch_images(image, kernel_size=kernel_size)

        with pytest.raises(
            ValueError, match="sigma is a sequence, its length should be 2"
        ):
            F.gaussian_blur_batch_images(image, kernel_size=1, sigma=[1, 2, 3])

        with pytest.raises(
            TypeError, match="sigma should be either float or sequence of floats"
        ):
            F.gaussian_blur_batch_images(image, kernel_size=1, sigma=object())

        with pytest.raises(ValueError, match="sigma should have positive values"):
            F.gaussian_blur_batch_images(image, kernel_size=1, sigma=-1)

        with pytest.raises(ValueError, match="sigma should have positive values"):
            F.gaussian_blur_batch_images(image, kernel_size=1, sigma=torch.tensor(-1))

        with pytest.raises(ValueError, match="sigma should have 1 or 2 dimensions"):
            F.gaussian_blur_batch_images(
                image, kernel_size=1, sigma=torch.tensor([[[1.0]]])
            )

        with pytest.raises(
            ValueError,
            match="sigma should have one element or the same length as the batch size",
        ):
            F.gaussian_blur_batch_images(
                image, kernel_size=1, sigma=torch.tensor([[1.0, 1.0, 1.0]])
            )

    @pytest.mark.parametrize("make_input", VIDEO_MAKERS)
    def test_kernel_video(self, make_input):
        check_kernel(F.gaussian_blur_video, make_input(), kernel_size=(3, 3))

    def test_kernel_batch_videos(self):
        check_kernel(
            F.gaussian_blur_batch_videos, make_batch_videos(), kernel_size=(3, 3)
        )

    @pytest.mark.parametrize(
        "make_input",
        IMAGE_MAKERS,
    )
    def test_functional(self, make_input):
        check_functional(F.gaussian_blur, make_input(), kernel_size=(3, 3))

    def test_batch_functional(self):
        check_functional(F.gaussian_blur_batch, make_batch_images(), kernel_size=(3, 3))

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.gaussian_blur_image, torch.Tensor),
            (F.gaussian_blur_image, ta_tensors.Image),
            (F.gaussian_blur_image, ta_tensors.BatchImages),
            (F.gaussian_blur_video, ta_tensors.Video),
            (F.gaussian_blur_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.gaussian_blur, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.gaussian_blur_batch_images, torch.Tensor),
            (F.gaussian_blur_batch_images, ta_tensors.BatchImages),
            (F.gaussian_blur_batch_videos, ta_tensors.BatchVideos),
        ],
    )
    def test_batch_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.gaussian_blur_batch, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        "make_input",
        ALL_IMAGES_MAKERS,
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("sigma", [5, 2.0, (0.5, 2), [1.3, 2.7]])
    def test_transform(self, make_input, device, sigma):
        check_transform(
            transforms.RandomGaussianBlur(kernel_size=3, sigma=sigma, p=1),
            make_input(device=device),
        )

    @pytest.mark.parametrize(
        "make_input",
        BATCH_IMAGES_TENSOR_AND_MAKERS,
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("sigma", [5, 2.0, (0.5, 2), [1.3, 2.7]])
    @pytest.mark.parametrize("p", [0, 0.5, 1])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("batch_inplace", [True, False])
    def test_batch_transform(
        self, make_input, device, sigma, batch_size, batch_inplace, p
    ):
        check_batch_transform(
            transforms.RandomGaussianBlur(
                kernel_size=3,
                sigma=sigma,
                batch_transform=True,
                batch_inplace=batch_inplace,
                p=p,
            ),
            make_input(device=device, batch_dims=(batch_size,)),
            batch_size=batch_size,
        )

    def test_assertions(self):
        with pytest.raises(
            ValueError, match="Kernel size should be a tuple/list of two integers"
        ):
            transforms.RandomGaussianBlur([10, 12, 14])

        with pytest.raises(
            ValueError, match="Kernel size value should be an odd and positive number"
        ):
            transforms.RandomGaussianBlur(4)

        with pytest.raises(
            ValueError,
            match="If sigma is a sequence its length should be 1 or 2. Got 3",
        ):
            transforms.RandomGaussianBlur(3, sigma=[1, 2, 3])

        with pytest.raises(
            ValueError, match="sigma values should be positive and of the form"
        ):
            transforms.RandomGaussianBlur(3, sigma=-1.0)

        with pytest.raises(
            ValueError, match="sigma values should be positive and of the form"
        ):
            transforms.RandomGaussianBlur(3, sigma=[2.0, 1.0])

        with pytest.raises(
            TypeError, match="sigma should be a number or a sequence of numbers"
        ):
            transforms.RandomGaussianBlur(3, sigma={})

    @pytest.mark.parametrize("sigma", [10.0, [10.0, 12.0], (10, 12.0), [10]])
    def test__get_params(self, sigma):
        transform = transforms.RandomGaussianBlur(3, sigma=sigma, p=1)
        params = transform._get_params(
            [],
            1,
            (
                torch.tensor(
                    [0],
                )
            ),
        )[0]

        if isinstance(sigma, float):
            assert params["sigma"][0] == params["sigma"][1] == sigma
        elif isinstance(sigma, list) and len(sigma) == 1:
            assert params["sigma"][0] == params["sigma"][1] == sigma[0]
        else:
            assert sigma[0] <= params["sigma"][0] <= sigma[1]
            assert sigma[0] <= params["sigma"][1] <= sigma[1]

    @pytest.mark.parametrize("sigma", [10.0, [10.0, 12.0], (10, 12.0), [10]])
    @pytest.mark.parametrize(
        "num_chunks, chunks_indices",
        ([1, (torch.tensor([0, 1]),)], [2, (torch.tensor([0]), torch.tensor([1]))]),
    )
    def test__get_batch_params(self, sigma, num_chunks, chunks_indices):
        transform = transforms.RandomGaussianBlur(
            3, sigma=sigma, p=1, batch_transform=True
        )
        list_params = transform._get_params(
            [make_batch_images()], num_chunks, chunks_indices
        )

        for params in list_params:
            if isinstance(sigma, float):
                assert torch.logical_and(
                    torch.eq(params["sigma"][:, 0], sigma),
                    torch.eq(params["sigma"][:, 1], sigma),
                ).all()
            elif isinstance(sigma, list) and len(sigma) == 1:
                assert torch.logical_and(
                    torch.eq(params["sigma"][:, 0], sigma[0]),
                    torch.eq(params["sigma"][:, 1], sigma[0]),
                ).all()
            else:
                assert torch.logical_and(
                    torch.ge(params["sigma"][:, 0], sigma[0]),
                    torch.le(params["sigma"][:, 0], sigma[1]),
                ).all()
                assert torch.logical_and(
                    torch.ge(params["sigma"][:, 1], sigma[0]),
                    torch.le(params["sigma"][:, 1], sigma[1]),
                ).all()

    @pytest.mark.parametrize(
        ("dimensions", "kernel_size", "sigma"),
        [
            ((3, 10, 12), (3, 3), 0.8),
            ((3, 10, 12), (3, 3), 0.5),
            ((3, 10, 12), (3, 5), 0.8),
            ((3, 10, 12), (3, 5), 0.5),
            ((1, 26, 28), (23, 23), 1.7),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_functional_image_correctness(
        self, dimensions, kernel_size, sigma, dtype, device
    ):
        if dtype is torch.float16 and device == "cpu":
            pytest.skip("reflection_pad2d not implemented for float16 on CPU")

        num_channels, height, width = dimensions

        image = ta_tensors.Image(
            torch.arange(num_channels * height * width, dtype=torch.uint8)
            .reshape(height, width, num_channels)
            .permute(2, 0, 1),
            dtype=dtype,
            device=device,
        )

        actual = F.gaussian_blur_image(image, kernel_size=kernel_size, sigma=sigma)
        expected = TVF.gaussian_blur_image(torch.as_tensor(image), kernel_size, sigma)

        torch.testing.assert_close(actual, expected, rtol=0, atol=1)

    @pytest.mark.parametrize(
        ("dimensions", "kernel_size", "sigma"),
        [
            ((3, 10, 12), (3, 3), torch.tensor(0.8)),
            ((3, 10, 12), (3, 3), 0.5),
            ((3, 10, 12), (3, 5), torch.tensor([0.8, 0.5])),
            ((3, 10, 12), (3, 5), torch.tensor([[0.8, 0.5]])),
            ((3, 10, 12), (3, 5), torch.tensor([0.8, 0.5, 0.3, 0.5])),
            (
                (3, 10, 12),
                (3, 5),
                torch.tensor([[0.8, 0.2], [0.5, 0.3], [0.3, 0.6], [0.5, 0.5]]),
            ),
            ((1, 10, 12), (3, 5), 0.5),
        ],
    )
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_functional_batch_images_correctness(
        self, dimensions, batch_size, kernel_size, sigma, dtype, device
    ):
        if dtype is torch.float16 and device == "cpu":
            pytest.skip("reflection_pad2d not implemented for float16 on CPU")

        if (
            isinstance(sigma, torch.Tensor)
            and sigma.ndim >= 1
            and sigma.shape[0] > batch_size
        ):
            pytest.skip(
                "sigma should have one element or the same length as the batch size"
            )

        num_channels, height, width = dimensions

        images = ta_tensors.BatchImages(
            torch.randint(
                0,
                256,
                (batch_size, num_channels, height, width),
                dtype=dtype,
                device=device,
            )
        )

        actual = F.gaussian_blur_batch_images(
            images, kernel_size=kernel_size, sigma=sigma
        )
        for i, a in enumerate(actual):
            if type(sigma) in (int, float):
                s = sigma
            else:
                if sigma.numel() == 1:
                    s = float(sigma)
                elif sigma.numel() == 2:
                    if sigma.ndim == 2:
                        s = sigma[0].tolist()
                    else:
                        pytest.skip(
                            "Use case hard to test as it is unclear if the tensor is for a batch or not."
                        )
                else:
                    s = sigma[i].tolist()

            e = TVF.gaussian_blur_image(images[i], kernel_size=kernel_size, sigma=s)
            torch.testing.assert_close(a, e, rtol=0, atol=1)


class TestToDtype:
    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.to_dtype_image, make_image_tensor),
            (F.to_dtype_image, make_image),
            (F.to_dtype_image, make_batch_images),
            (F.to_dtype_video, make_video),
            (F.to_dtype_video, make_batch_videos),
        ],
    )
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64, torch.uint8])
    @pytest.mark.parametrize(
        "output_dtype", [torch.float32, torch.float64, torch.uint8]
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("scale", (True, False))
    def test_kernel(self, kernel, make_input, input_dtype, output_dtype, device, scale):
        check_kernel(
            kernel,
            make_input(dtype=input_dtype, device=device),
            dtype=output_dtype,
            scale=scale,
        )

    @pytest.mark.parametrize("make_input", [make_image_tensor, make_image, make_video])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64, torch.uint8])
    @pytest.mark.parametrize(
        "output_dtype", [torch.float32, torch.float64, torch.uint8]
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("scale", (True, False))
    def test_functional(self, make_input, input_dtype, output_dtype, device, scale):
        check_functional(
            F.to_dtype,
            make_input(dtype=input_dtype, device=device),
            dtype=output_dtype,
            scale=scale,
        )

    @pytest.mark.parametrize("make_input", ALL_IMAGES_MAKERS)
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64, torch.uint8])
    @pytest.mark.parametrize(
        "output_dtype", [torch.float32, torch.float64, torch.uint8]
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("scale", (True, False))
    @pytest.mark.parametrize("as_dict", (True, False))
    def test_transform(
        self, make_input, input_dtype, output_dtype, device, scale, as_dict
    ):
        input = make_input(dtype=input_dtype, device=device)
        if as_dict:
            output_dtype = {type(input): output_dtype}
        check_transform(
            transforms.ToDtype(dtype=output_dtype, scale=scale),
            input,
            check_sample_input=not as_dict,
        )

    def reference_convert_dtype_image_tensor(
        self, image, dtype=torch.float, scale=False
    ):
        input_dtype = image.dtype
        output_dtype = dtype

        if not scale:
            return image.to(dtype)

        if output_dtype == input_dtype:
            return image

        def fn(value):
            if input_dtype.is_floating_point:
                if output_dtype.is_floating_point:
                    return value
                else:
                    return round(decimal.Decimal(value) * torch.iinfo(output_dtype).max)
            else:
                input_max_value = torch.iinfo(input_dtype).max

                if output_dtype.is_floating_point:
                    return float(decimal.Decimal(value) / input_max_value)
                else:
                    output_max_value = torch.iinfo(output_dtype).max

                    if input_max_value > output_max_value:
                        factor = (input_max_value + 1) // (output_max_value + 1)
                        return value / factor
                    else:
                        factor = (output_max_value + 1) // (input_max_value + 1)
                        return value * factor

        return torch.tensor(
            tree_map(fn, image.tolist()), dtype=dtype, device=image.device
        )

    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64, torch.uint8])
    @pytest.mark.parametrize(
        "output_dtype", [torch.float32, torch.float64, torch.uint8]
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("scale", (True, False))
    @pytest.mark.parametrize("make_input", [make_image, make_batch_images])
    def test_image_correctness(
        self, input_dtype, output_dtype, device, scale, make_input
    ):
        if input_dtype.is_floating_point and output_dtype == torch.int64:
            pytest.xfail("float to int64 conversion is not supported")

        input = make_input(dtype=input_dtype, device=device)

        out = F.to_dtype(input, dtype=output_dtype, scale=scale)
        expected = self.reference_convert_dtype_image_tensor(
            input, dtype=output_dtype, scale=scale
        )

        if (
            input_dtype.is_floating_point
            and not output_dtype.is_floating_point
            and scale
        ):
            torch.testing.assert_close(out, expected, atol=1, rtol=0)
        else:
            torch.testing.assert_close(out, expected)

    def was_scaled(self, inpt):
        # this assumes the target dtype is float
        return inpt.max() <= 1

    def make_inpt_with_bbox_and_mask(self, make_input, is_batch=False):
        H, W = 10, 10
        inpt_dtype = torch.uint8
        bbox_dtype = torch.float32
        mask_dtype = torch.bool
        sample = {
            "inpt": make_input(size=(H, W), dtype=inpt_dtype),
            "bbox": make_bounding_boxes(canvas_size=(H, W), dtype=bbox_dtype)
            if not is_batch
            else make_batch_bounding_boxes(canvas_size=(H, W), dtype=bbox_dtype),
            "mask": make_detection_masks(size=(H, W), dtype=mask_dtype)
            if not is_batch
            else make_batch_detection_masks(size=(H, W), dtype=mask_dtype),
        }

        return sample, inpt_dtype, bbox_dtype, mask_dtype

    @pytest.mark.parametrize(
        "make_input",
        (
            make_image_tensor,
            make_image,
            make_video,
            make_batch_images,
            make_batch_videos,
        ),
    )
    @pytest.mark.parametrize("scale", (True, False))
    def test_dtype_not_a_dict(self, make_input, scale):
        # assert only inpt gets transformed when dtype isn't a dict

        sample, inpt_dtype, bbox_dtype, mask_dtype = self.make_inpt_with_bbox_and_mask(
            make_input, is_batch=make_input in (make_batch_images, make_batch_videos)
        )
        out = transforms.ToDtype(dtype=torch.float32, scale=scale)(sample)

        assert out["inpt"].dtype != inpt_dtype
        assert out["inpt"].dtype == torch.float32
        if scale:
            assert self.was_scaled(out["inpt"])
        else:
            assert not self.was_scaled(out["inpt"])
        assert out["bbox"].dtype == bbox_dtype
        assert out["mask"].dtype == mask_dtype

    @pytest.mark.parametrize(
        "make_input",
        (
            make_image_tensor,
            make_image,
            make_video,
            make_batch_images,
            make_batch_videos,
        ),
    )
    def test_others_catch_all_and_none(self, make_input):
        # make sure "others" works as a catch-all and that None means no conversion

        sample, inpt_dtype, bbox_dtype, mask_dtype = self.make_inpt_with_bbox_and_mask(
            make_input, is_batch=make_input in (make_batch_images, make_batch_videos)
        )
        out = transforms.ToDtype(
            dtype={
                ta_tensors.Mask: torch.int64,
                ta_tensors.BatchMasks: torch.int64,
                "others": None,
            }
        )(sample)
        assert out["inpt"].dtype == inpt_dtype
        assert out["bbox"].dtype == bbox_dtype
        assert out["mask"].dtype != mask_dtype
        assert out["mask"].dtype == torch.int64

    @pytest.mark.parametrize(
        "make_input",
        (
            make_image_tensor,
            make_image,
            make_video,
            make_batch_images,
            make_batch_videos,
        ),
    )
    def test_typical_use_case(self, make_input):
        # Typical use-case: want to convert dtype and scale for inpt and just dtype for masks.
        # This just makes sure we now have a decent API for this

        sample, inpt_dtype, bbox_dtype, mask_dtype = self.make_inpt_with_bbox_and_mask(
            make_input, is_batch=make_input in (make_batch_images, make_batch_videos)
        )
        out = transforms.ToDtype(
            dtype={
                type(sample["inpt"]): torch.float32,
                ta_tensors.Mask: torch.int64,
                ta_tensors.BatchMasks: torch.int64,
                "others": None,
            },
            scale=True,
        )(sample)
        assert out["inpt"].dtype != inpt_dtype
        assert out["inpt"].dtype == torch.float32
        assert self.was_scaled(out["inpt"])
        assert out["bbox"].dtype == bbox_dtype
        assert out["mask"].dtype != mask_dtype
        assert out["mask"].dtype == torch.int64

    @pytest.mark.parametrize(
        "make_input",
        (
            make_image_tensor,
            make_image,
            make_video,
            make_batch_images,
            make_batch_videos,
        ),
    )
    def test_errors_warnings(self, make_input):
        sample, inpt_dtype, bbox_dtype, mask_dtype = self.make_inpt_with_bbox_and_mask(
            make_input, is_batch=make_input in (make_batch_images, make_batch_videos)
        )

        with pytest.raises(ValueError, match="No dtype was specified for"):
            out = transforms.ToDtype(dtype={ta_tensors.Mask: torch.float32})(sample)
        with pytest.warns(
            UserWarning,
            match=re.escape("plain `torch.Tensor` will *not* be transformed"),
        ):
            transforms.ToDtype(
                dtype={torch.Tensor: torch.float32, ta_tensors.Image: torch.float32}
            )
        with pytest.warns(UserWarning, match="no scaling will be done"):
            out = transforms.ToDtype(dtype={"others": None}, scale=True)(sample)
        assert out["inpt"].dtype == inpt_dtype
        assert out["bbox"].dtype == bbox_dtype
        assert out["mask"].dtype == mask_dtype


class TestSanitizeBoundingBoxes:
    @pytest.mark.parametrize("min_size", (1, 10))
    @pytest.mark.parametrize(
        "labels_getter",
        ("default", lambda inputs: inputs["labels"], None, lambda inputs: None),
    )
    @pytest.mark.parametrize("sample_type", (tuple, dict))
    @pytest.mark.parametrize("batch", [False, True])
    def test_transform(self, min_size, labels_getter, sample_type, batch):

        if sample_type is tuple and not isinstance(labels_getter, str):
            # The "lambda inputs: inputs["labels"]" labels_getter used in this test
            # doesn't work if the input is a tuple.
            return

        H, W = 256, 128

        boxes_and_validity = [
            ([0, 1, 10, 1], False),  # Y1 == Y2
            ([0, 1, 0, 20], False),  # X1 == X2
            ([0, 0, min_size - 1, 10], False),  # H < min_size
            ([0, 0, 10, min_size - 1], False),  # W < min_size
            ([0, 0, 10, H + 1], False),  # Y2 > H
            ([0, 0, W + 1, 10], False),  # X2 > W
            ([-1, 1, 10, 20], False),  # any < 0
            ([0, 0, -1, 20], False),  # any < 0
            ([0, 0, -10, -1], False),  # any < 0
            ([0, 0, min_size, 10], True),  # H < min_size
            ([0, 0, 10, min_size], True),  # W < min_size
            ([0, 0, W, H], True),  # TODO: Is that actually OK?? Should it be -1?
            ([1, 1, 30, 20], True),
            ([0, 0, 10, 10], True),
            ([1, 1, 30, 20], True),
        ]

        random.shuffle(
            boxes_and_validity
        )  # For test robustness: mix order of wrong and correct cases
        boxes, is_valid_mask = zip(*boxes_and_validity)
        valid_indices = [i for (i, is_valid) in enumerate(is_valid_mask) if is_valid]

        if batch:
            valid_indices = valid_indices + [v + len(boxes) for v in valid_indices]

        boxes = torch.tensor(boxes)
        labels = (
            torch.arange(boxes.shape[0])
            if not batch
            else torch.arange(boxes.shape[0] * 2)
        )

        boxes = (
            ta_tensors.BoundingBoxes(
                boxes,
                format=ta_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(H, W),
            )
            if not batch
            else ta_tensors.BatchBoundingBoxes(
                torch.cat([boxes, boxes], 0),
                format=ta_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(H, W),
                idx_sample=[0, boxes.shape[0], boxes.shape[0] * 2],
            )
        )

        masks = (
            ta_tensors.Mask(torch.randint(0, 2, size=(boxes.shape[0], H, W)))
            if not batch
            else ta_tensors.BatchMasks(
                torch.randint(0, 2, size=(boxes.shape[0], 2, H, W)),
                idx_sample=[0, boxes.shape[0] // 2, boxes.shape[0]],
            )
        )
        whatever = torch.rand(10)
        input_img = (
            torch.randint(0, 256, size=(1, 3, H, W), dtype=torch.uint8)
            if not batch
            else torch.randint(0, 256, size=(2, 3, H, W), dtype=torch.uint8)
        )

        sample = {
            "image": input_img,
            "labels": labels,
            "boxes": boxes,
            "whatever": whatever,
            "None": None,
            "masks": masks,
        }

        if sample_type is tuple:
            img = sample.pop("image")
            sample = (img, sample)

        out = transforms.SanitizeBoundingBoxes(
            min_size=min_size, labels_getter=labels_getter
        )(sample)

        if sample_type is tuple:
            out_image = out[0]
            out_labels = out[1]["labels"]
            out_boxes = out[1]["boxes"]
            out_masks = out[1]["masks"]
            out_whatever = out[1]["whatever"]
        else:
            out_image = out["image"]
            out_labels = out["labels"]
            out_boxes = out["boxes"]
            out_masks = out["masks"]
            out_whatever = out["whatever"]

        assert out_image is input_img
        assert out_whatever is whatever

        assert isinstance(
            out_boxes,
            ta_tensors.BoundingBoxes if not batch else ta_tensors.BatchBoundingBoxes,
        )
        assert (
            isinstance(out_masks, ta_tensors.Mask)
            if not batch
            else isinstance(out_masks, ta_tensors.BatchMasks)
        )

        if labels_getter is None or (
            callable(labels_getter) and labels_getter({"labels": "blah"}) is None
        ):
            assert out_labels is labels
        else:
            assert isinstance(out_labels, torch.Tensor)
            assert out_boxes.shape[0] == out_labels.shape[0] == out_masks.shape[0]
            # This works because we conveniently set labels to arange(num_boxes)
            assert out_labels.tolist() == valid_indices

    @pytest.mark.parametrize("batch", [False, True])
    def test_no_label(self, batch):
        # Non-regression test for https://github.com/pytorch/vision/issues/7878

        img = make_image() if not batch else make_batch_images()
        boxes = make_bounding_boxes() if not batch else make_batch_bounding_boxes()

        with pytest.raises(
            ValueError, match="or a two-tuple whose second item is a dict"
        ):
            transforms.SanitizeBoundingBoxes()(img, boxes)

        out_img, out_boxes = transforms.SanitizeBoundingBoxes(labels_getter=None)(
            img, boxes
        )
        assert (
            isinstance(out_img, ta_tensors.Image)
            if not batch
            else isinstance(out_img, ta_tensors.BatchImages)
        )
        assert (
            isinstance(out_boxes, ta_tensors.BoundingBoxes)
            if not batch
            else isinstance(out_boxes, ta_tensors.BatchBoundingBoxes)
        )

    def test_errors(self):
        good_bbox = ta_tensors.BoundingBoxes(
            [[0, 0, 10, 10]],
            format=ta_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(20, 20),
        )

        with pytest.raises(ValueError, match="min_size must be >= 1"):
            transforms.SanitizeBoundingBoxes(min_size=0)
        with pytest.raises(
            ValueError, match="labels_getter should either be 'default'"
        ):
            transforms.SanitizeBoundingBoxes(labels_getter=12)

        with pytest.raises(ValueError, match="Could not infer where the labels are"):
            bad_labels_key = {
                "bbox": good_bbox,
                "BAD_KEY": torch.arange(good_bbox.shape[0]),
            }
            transforms.SanitizeBoundingBoxes()(bad_labels_key)

        with pytest.raises(ValueError, match="must be a tensor"):
            not_a_tensor = {
                "bbox": good_bbox,
                "labels": torch.arange(good_bbox.shape[0]).tolist(),
            }
            transforms.SanitizeBoundingBoxes()(not_a_tensor)

        with pytest.raises(ValueError, match="Number of boxes"):
            different_sizes = {
                "bbox": good_bbox,
                "labels": torch.arange(good_bbox.shape[0] + 3),
            }
            transforms.SanitizeBoundingBoxes()(different_sizes)
