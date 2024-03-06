import re

import pytest
import torch
import torchaug.transforms as transforms
import torchaug.transforms.functional as F
import torchvision.transforms.v2 as tv_transforms
import torchvision.transforms.v2.functional as TVF
from torchaug import ta_tensors
from torchaug.transforms.functional._utils._tensor import _max_value as get_max_value

from ..utils import (
    assert_equal,
    check_batch_transform,
    check_functional,
    check_functional_kernel_signature_match,
    check_kernel,
    check_transform,
    cpu_and_cuda,
    freeze_rng_state,
    IMAGE_MAKERS,
    make_batch_images,
    make_batch_images_tensor,
    make_batch_videos,
    make_image,
    make_image_tensor,
    make_video,
    transform_cls_to_functional,
    VIDEO_MAKERS,
)


class TestRgbToGrayscale:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_image):
        check_kernel(F.rgb_to_grayscale_image, make_image(dtype=dtype, device=device))

    @pytest.mark.parametrize(
        "make_input", IMAGE_MAKERS + [make_image_tensor, make_batch_images_tensor]
    )
    def test_functional(self, make_input):
        check_functional(F.rgb_to_grayscale, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.rgb_to_grayscale_image, torch.Tensor),
            (F.rgb_to_grayscale_image, ta_tensors.Image),
            (F.rgb_to_grayscale_image, ta_tensors.BatchImages),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.rgb_to_grayscale, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        "transform", [transforms.Grayscale(), transforms.RandomGrayscale(p=1)]
    )
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_transform(self, transform, make_input):
        check_transform(transform, make_input())

    @pytest.mark.parametrize("num_output_channels", [1, 3])
    @pytest.mark.parametrize("color_space", ["RGB", "GRAY"])
    @pytest.mark.parametrize(
        "fn", [F.rgb_to_grayscale, transform_cls_to_functional(transforms.Grayscale)]
    )
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_image_correctness(self, num_output_channels, color_space, fn, make_image):
        # TODO Remove this once Torchvision 0.18.0 is released.
        if color_space == "GRAY":
            pytest.skip("No change in image for Torchvision in 0.17.0")

        image = make_image(dtype=torch.uint8, device="cpu", color_space=color_space)

        actual = fn(image, num_output_channels=num_output_channels)

        expected = TVF.rgb_to_grayscale(
            torch.as_tensor(image), num_output_channels=num_output_channels
        )

        assert_equal(actual, expected, rtol=0, atol=1)

    def test_expanded_channels_are_not_views_into_the_same_underlying_tensor(self):
        image = make_image(dtype=torch.uint8, device="cpu", color_space="GRAY")

        output_image = F.rgb_to_grayscale(image, num_output_channels=3)
        assert_equal(output_image[0][0][0], output_image[1][0][0])
        output_image[0][0][0] = output_image[0][0][0] + 1
        assert output_image[0][0][0] != output_image[1][0][0]

    @pytest.mark.parametrize("num_input_channels", [1, 3])
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_random_transform_correctness(self, num_input_channels, make_image):
        image = make_image(
            color_space={
                1: "GRAY",
                3: "RGB",
            }[num_input_channels],
            dtype=torch.uint8,
            device="cpu",
        )

        transform = transforms.RandomGrayscale(p=1)

        actual = transform(image)
        expected = TVF.rgb_to_grayscale(num_output_channels=num_input_channels)

        assert_equal(actual, expected, rtol=0, atol=1)

    @pytest.mark.parametrize("num_input_channels", [1, 3])
    @pytest.mark.parametrize("make_image", [make_batch_images])
    @pytest.mark.parametrize("batch_inplace", [False, True])
    def test_random_transform_correctness(
        self, num_input_channels, make_image, batch_inplace
    ):
        image = make_image(
            color_space={
                1: "GRAY",
                3: "RGB",
            }[num_input_channels],
            dtype=torch.uint8,
            device="cpu",
        )

        transform = transforms.RandomGrayscale(
            p=1, batch_inplace=batch_inplace, batch_transform=True
        )

        actual = transform(image)
        expected = TVF.rgb_to_grayscale(
            torch.as_tensor(image), num_output_channels=num_input_channels
        )

        assert_equal(actual, expected, rtol=0, atol=1)


class TestColorJitter:
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
    @pytest.mark.parametrize("p", [0, 0.5, 1])
    def test_transform(self, make_input, dtype, device, p):
        check_transform(
            transforms.RandomColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=p
            ),
            make_input(dtype=dtype, device=device),
        )

    @pytest.mark.parametrize(
        "make_input",
        [make_batch_images_tensor, make_batch_images, make_batch_videos],
    )
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("batch_inplace", [False, True])
    @pytest.mark.parametrize("num_chunks", [1, 2])
    @pytest.mark.parametrize("permute_chunks", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("p", [0, 0.5, 1])
    def test_batch_transform(
        self,
        make_input,
        dtype,
        device,
        batch_inplace,
        num_chunks,
        permute_chunks,
        batch_size,
        p,
    ):
        if num_chunks > batch_size:
            pytest.skip("num_chunks cannot be greater than batch_size")

        check_batch_transform(
            transforms.RandomColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.25,
                p=p,
                batch_inplace=batch_inplace,
                num_chunks=num_chunks,
                permute_chunks=permute_chunks,
                batch_transform=True,
            ),
            make_input(dtype=dtype, device=device, batch_dims=(batch_size,)),
            batch_size=batch_size,
        )

    @pytest.mark.parametrize("is_batch", [False, True])
    def test_transform_noop(self, is_batch):
        input = make_image() if not is_batch else make_batch_images()
        input_version = input._version

        transform = transforms.RandomColorJitter(batch_transform=is_batch, p=1)
        output = transform(input)

        assert output is input
        assert output.data_ptr() == input.data_ptr()
        assert output._version == input_version

    def test_transform_error(self):
        with pytest.raises(ValueError, match="must be non negative"):
            transforms.RandomColorJitter(brightness=-1)

        for brightness in [object(), [1, 2, 3]]:
            with pytest.raises(
                TypeError, match="single number or a sequence with length 2"
            ):
                transforms.RandomColorJitter(brightness=brightness)

        with pytest.raises(ValueError, match="values should be between"):
            transforms.RandomColorJitter(brightness=(-1, 0.5))

        with pytest.raises(ValueError, match="values should be between"):
            transforms.RandomColorJitter(hue=1)

    @pytest.mark.parametrize("brightness", [None, 0.1, (0.2, 0.3)])
    @pytest.mark.parametrize("contrast", [None, 0.4, (0.5, 0.6)])
    @pytest.mark.parametrize("saturation", [None, 0.7, (0.8, 0.9)])
    @pytest.mark.parametrize("hue", [None, 0.3, (-0.1, 0.2)])
    def test_transform_get_params_correctness(
        self, brightness, contrast, saturation, hue, mocker
    ):
        image = make_image(dtype=torch.uint8, device="cpu")
        transform = transforms.RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=1,
        )

        params = transform._get_params(image, 1, (torch.tensor([0]),))[0]

        for param_name, value in params.items():
            assert param_name in [
                "brightness_factor",
                "contrast_factor",
                "saturation_factor",
                "hue_factor",
                "fn_idx",
            ]

            if param_name == "fn_idx":
                assert len(value) == 4
            else:
                if value is not None:
                    name = param_name.split("_")[0]
                    original_value = locals()[name]
                    if value is None:
                        assert original_value is None
                    else:
                        if type(original_value) is float:
                            if name == "hue":
                                original_value = (-original_value, original_value)
                            else:
                                original_value = (
                                    1 - original_value,
                                    1 + original_value,
                                )
                        assert original_value[0] <= value <= original_value[1]

    @pytest.mark.parametrize("brightness", [None, 0.1, (0.2, 0.3)])
    @pytest.mark.parametrize("contrast", [None, 0.4, (0.5, 0.6)])
    @pytest.mark.parametrize("saturation", [None, 0.7, (0.8, 0.9)])
    @pytest.mark.parametrize("hue", [None, 0.3, (-0.1, 0.2)])
    @pytest.mark.parametrize(
        "num_chunks, chunks_indices",
        [
            (1, (torch.tensor([0, 1]),)),
            (2, (torch.tensor([0, 1]), torch.tensor([2, 3]))),
            (2, (torch.tensor([0, 1]), torch.tensor([2, 3, 4]))),
        ],
    )
    def test_batch_transform_get_params_correctness(
        self, brightness, contrast, saturation, hue, num_chunks, chunks_indices, mocker
    ):
        image = make_batch_images(dtype=torch.uint8, device="cpu")
        transform = transforms.RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=1,
            num_chunks=num_chunks,
            batch_transform=True,
        )

        list_params = transform._get_params(image, num_chunks, chunks_indices)

        for i, params in enumerate(list_params):
            for param_name, value in params.items():
                assert param_name in [
                    "brightness_factor",
                    "contrast_factor",
                    "saturation_factor",
                    "hue_factor",
                    "fn_idx",
                ]

                if param_name == "fn_idx":
                    assert len(value) == 4
                else:
                    if value is not None:
                        name = param_name.split("_")[0]
                        original_value = locals()[name]
                        if value is None:
                            assert original_value is None
                        else:
                            assert value.numel() == chunks_indices[i].numel()
                            if type(original_value) is float:
                                if name == "hue":
                                    original_value = (-original_value, original_value)
                                else:
                                    original_value = (
                                        1 - original_value,
                                        1 + original_value,
                                    )
                            assert torch.all(
                                torch.logical_and(
                                    torch.ge(value, torch.tensor(original_value[0])),
                                    torch.le(value, torch.tensor(original_value[1])),
                                )
                            )

    @pytest.mark.parametrize("brightness", [None, 0.1])
    @pytest.mark.parametrize("contrast", [None, 0.4])
    @pytest.mark.parametrize("saturation", [None, 0.7])
    @pytest.mark.parametrize("hue", [None, 0.3])
    def test_batch_transform_correctness(
        self, brightness, contrast, saturation, hue, mocker
    ):
        image = make_batch_images(dtype=torch.uint8, device="cpu")

        params = {
            "fn_idx": [0, 1, 2, 3],
            "brightness_factor": brightness,
            "contrast_factor": contrast,
            "saturation_factor": saturation,
            "hue_factor": hue,
        }

        transform = transforms.RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=1,
        )

        transform._get_params = mocker.MagicMock(return_value=[params])

        tv_transform = tv_transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        tv_transform._get_params = mocker.MagicMock(return_value=params)

        with freeze_rng_state():
            actual = transform(image)

            expected = tv_transform(torch.as_tensor(image))

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 2

    @pytest.mark.parametrize("brightness", [None, 0.1])
    @pytest.mark.parametrize("contrast", [None, 0.4])
    @pytest.mark.parametrize("saturation", [None, 0.7])
    @pytest.mark.parametrize("hue", [None, 0.3])
    @pytest.mark.parametrize("num_chunks", [1, 2])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_transform_correctness(
        self, brightness, contrast, saturation, hue, num_chunks, batch_size
    ):
        if num_chunks > batch_size:
            pytest.skip("num_chunks cannot be greater than batch_size")

        image = make_batch_images(
            dtype=torch.uint8, device="cpu", batch_dims=(batch_size,)
        )

        transform = transforms.RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=1,
            batch_transform=True,
            num_chunks=num_chunks,
        )

        sample_transform = transforms.RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=1,
            batch_transform=False,
        )

        with freeze_rng_state():
            chunks_indices = transform._get_chunks_indices(
                batch_size, num_chunks, image.device
            )
            list_params = transform._get_params(image, num_chunks, chunks_indices)

            for params, chunk_indices in zip(list_params, chunks_indices):
                image_chunk = image[chunk_indices]

                actual = transform._transform(image_chunk, params)

                samples_params = []
                for j in range(image_chunk.shape[0]):
                    samples_params.append(
                        {
                            key: float(params[key][j])
                            if isinstance(params[key], torch.Tensor)
                            else params[key]
                            for key in params
                        }
                    )

                expected = torch.stack(
                    [
                        sample_transform._transform(image, sample_param)
                        for image, sample_param in zip(image_chunk, samples_params)
                    ]
                )

                mae = (actual.float() - expected.float()).abs().mean()
                assert mae < 2

    def test_instantiate_color_jitter_transform(self):
        transform = transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25
        )
        assert isinstance(transform, transforms.RandomColorJitter)
        assert transform.p == 1


class TestPermuteChannels:
    _DEFAULT_PERMUTATION = [2, 0, 1]

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.permute_channels_image, make_image_tensor),
            (F.permute_channels_image, make_image),
            (F.permute_channels_image, make_batch_images),
            (F.permute_channels_video, make_video),
            (F.permute_channels_video, make_batch_videos),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel(self, kernel, make_input, dtype, device):
        check_kernel(
            kernel,
            make_input(dtype=dtype, device=device),
            permutation=self._DEFAULT_PERMUTATION,
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
            F.permute_channels, make_input(), permutation=self._DEFAULT_PERMUTATION
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.permute_channels_image, torch.Tensor),
            (F.permute_channels_image, ta_tensors.Image),
            (F.permute_channels_image, ta_tensors.BatchImages),
            (F.permute_channels_video, ta_tensors.Video),
            (F.permute_channels_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.permute_channels, kernel=kernel, input_type=input_type
        )

    def reference_image_correctness(self, image, permutation):
        channel_images = image.split(1, dim=-3)
        permuted_channel_images = [
            channel_images[channel_idx] for channel_idx in permutation
        ]
        return ta_tensors.Image(torch.concat(permuted_channel_images, dim=-3))

    def reference_batch_images_correctness(self, image, permutation):
        channel_images = image.split(1, dim=-3)
        permuted_channel_images = [
            channel_images[channel_idx] for channel_idx in permutation
        ]
        return ta_tensors.BatchImages(torch.concat(permuted_channel_images, dim=-3))

    @pytest.mark.parametrize(
        "permutation", [[2, 0, 1], [1, 2, 0], [2, 0, 1], [0, 1, 2]]
    )
    @pytest.mark.parametrize("batch_dims", [(), (2,), (2, 1)])
    @pytest.mark.parametrize("make_input", [make_image])
    def test_image_correctness(self, permutation, batch_dims, make_input):
        image = make_input(batch_dims=batch_dims)

        actual = F.permute_channels(image, permutation=permutation)
        expected = self.reference_image_correctness(image, permutation=permutation)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "permutation", [[2, 0, 1], [1, 2, 0], [2, 0, 1], [0, 1, 2]]
    )
    @pytest.mark.parametrize("batch_dims", [(2,), (2, 1)])
    def test_batch_images_correctness(self, permutation, batch_dims):
        image = make_batch_images(batch_dims=batch_dims)

        actual = F.permute_channels(image, permutation=permutation)
        expected = self.reference_batch_images_correctness(
            image, permutation=permutation
        )

        torch.testing.assert_close(actual, expected)


class TestRandomPhotometricDistort:
    # Tests are light because this largely relies on the already tested
    # `adjust_{brightness,contrast,saturation,hue}` and `permute_channels` kernels.

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
        check_transform(
            transforms.RandomPhotometricDistort(
                brightness=(0.3, 0.4),
                contrast=(0.5, 0.6),
                saturation=(0.7, 0.8),
                hue=(-0.1, 0.2),
                p=1,
            ),
            make_input(dtype=dtype, device=device),
        )

    @pytest.mark.parametrize(
        "make_input",
        [make_batch_images_tensor, make_batch_images, make_batch_videos],
    )
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("batch_inplace", [False, True])
    @pytest.mark.parametrize("num_chunks", [1, 2])
    @pytest.mark.parametrize("permute_chunks", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("p", [0, 0.5, 1])
    def test_transform(
        self,
        make_input,
        dtype,
        device,
        batch_inplace,
        num_chunks,
        permute_chunks,
        batch_size,
        p,
    ):
        if num_chunks > batch_size:
            pytest.skip("num_chunks cannot be greater than batch_size")
        check_batch_transform(
            transforms.RandomPhotometricDistort(
                brightness=(0.3, 0.4),
                contrast=(0.5, 0.6),
                saturation=(0.7, 0.8),
                hue=(-0.1, 0.2),
                p=p,
                batch_inplace=batch_inplace,
                num_chunks=num_chunks,
                permute_chunks=permute_chunks,
                batch_transform=True,
            ),
            make_input(dtype=dtype, device=device, batch_dims=(batch_size,)),
            batch_size,
        )


class TestEqualize:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_image):
        check_kernel(F.equalize_image, make_image(dtype=dtype, device=device))

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(F.equalize_video, make_video())

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
        check_functional(F.equalize, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.equalize_image, torch.Tensor),
            (F.equalize_image, ta_tensors.Image),
            (F.equalize_image, ta_tensors.BatchImages),
            (F.equalize_video, ta_tensors.Video),
            (F.equalize_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.equalize, kernel=kernel, input_type=input_type
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
        check_transform(transforms.RandomEqualize(p=1), make_input())

    @pytest.mark.parametrize(
        "make_input",
        [make_batch_images_tensor, make_batch_images, make_batch_videos],
    )
    @pytest.mark.parametrize("batch_inplace", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_transform(self, make_input, batch_inplace, batch_size):
        check_batch_transform(
            transforms.RandomEqualize(
                p=1, batch_inplace=batch_inplace, batch_transform=True
            ),
            make_input(batch_dims=(batch_size,)),
            batch_size,
        )

    @pytest.mark.parametrize(
        ("low", "high"),
        [(0, 64), (64, 192), (192, 256), (0, 1), (127, 128), (255, 256)],
    )
    @pytest.mark.parametrize(
        "fn", [F.equalize, transform_cls_to_functional(transforms.RandomEqualize, p=1)]
    )
    @pytest.mark.parametrize("is_batch", [False, True])
    def test_image_correctness(self, low, high, fn, is_batch):
        # We are not using the default `make_image` here since that uniformly samples the values over the whole value
        # range. Since the whole point of F.equalize is to transform an arbitrary distribution of values into a uniform
        # one over the full range, the information gain is low if we already provide something really close to the
        # expected value.
        image = (
            ta_tensors.Image(
                torch.testing.make_tensor(
                    (3, 117, 253), dtype=torch.uint8, device="cpu", low=low, high=high
                )
            )
            if is_batch
            else ta_tensors.BatchImages(
                torch.testing.make_tensor(
                    (2, 3, 117, 253),
                    dtype=torch.uint8,
                    device="cpu",
                    low=low,
                    high=high,
                )
            )
        )

        actual = fn(image)
        expected = TVF.equalize(torch.as_tensor(image))

        assert_equal(actual, expected)


class TestInvert:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_image):
        check_kernel(F.invert_image, make_image(dtype=dtype, device=device))

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(F.invert_video, make_video())

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
        check_functional(F.invert, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.invert_image, torch.Tensor),
            (F.invert_image, ta_tensors.Image),
            (F.invert_image, ta_tensors.BatchImages),
            (F.invert_video, ta_tensors.Video),
            (F.invert_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.invert, kernel=kernel, input_type=input_type
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
        check_transform(transforms.RandomInvert(p=1), make_input())

    @pytest.mark.parametrize(
        "make_input", [make_batch_images_tensor, make_batch_images, make_batch_videos]
    )
    @pytest.mark.parametrize("batch_inplace", [False, True])
    @pytest.mark.parametrize("p", [0, 0.5, 1])
    def test_batch_transform(self, make_input, batch_inplace, p):
        check_batch_transform(
            transforms.RandomInvert(
                p=p, batch_transform=True, batch_inplace=batch_inplace
            ),
            make_input(batch_dims=(2,)),
            2,
        )

    @pytest.mark.parametrize(
        "fn", [F.invert, transform_cls_to_functional(transforms.RandomInvert, p=1)]
    )
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_correctness_image(self, fn, make_image):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = TVF.invert(torch.as_tensor(image))

        assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "fn",
        [
            F.invert,
            transform_cls_to_functional(
                transforms.RandomInvert, p=1, batch_transform=True
            ),
        ],
    )
    def test_correctness_batch_images(self, fn):
        image = make_batch_images(dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = TVF.invert(torch.as_tensor(image))

        assert_equal(actual, expected)


class TestPosterize:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_image):
        check_kernel(F.posterize_image, make_image(dtype=dtype, device=device), bits=1)

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(F.posterize_video, make_video(), bits=1)

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
        check_functional(F.posterize, make_input(), bits=1)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.posterize_image, torch.Tensor),
            (F.posterize_image, ta_tensors.Image),
            (F.posterize_image, ta_tensors.BatchImages),
            (F.posterize_video, ta_tensors.Video),
            (F.posterize_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.posterize, kernel=kernel, input_type=input_type
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
        check_transform(transforms.RandomPosterize(bits=1, p=1), make_input())

    @pytest.mark.parametrize(
        "make_input", [make_batch_images_tensor, make_batch_images, make_batch_videos]
    )
    @pytest.mark.parametrize("batch_inplace", [False, True])
    @pytest.mark.parametrize("p", [0, 0.5, 1])
    def test_batch_transform(self, make_input, batch_inplace, p):
        check_batch_transform(
            transforms.RandomPosterize(
                bits=1, p=p, batch_transform=True, batch_inplace=batch_inplace
            ),
            make_input(batch_dims=(2,)),
            2,
        )

    @pytest.mark.parametrize("bits", [1, 4, 8])
    @pytest.mark.parametrize(
        "fn",
        [F.posterize, transform_cls_to_functional(transforms.RandomPosterize, p=1)],
    )
    def test_correctness_image(self, bits, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image, bits=bits)
        expected = TVF.posterize(torch.as_tensor(image), bits=bits)

        assert_equal(actual, expected)

    @pytest.mark.parametrize("bits", [1, 4, 8])
    @pytest.mark.parametrize(
        "fn",
        [
            F.posterize,
            transform_cls_to_functional(
                transforms.RandomPosterize, p=1, batch_transform=True
            ),
        ],
    )
    def test_correctness_batch_images(self, bits, fn):
        image = make_batch_images(dtype=torch.uint8, device="cpu")

        actual = fn(image, bits=bits)
        expected = TVF.posterize(torch.as_tensor(image), bits=bits)

        assert_equal(actual, expected)


class TestSolarize:
    def _make_threshold(self, input, *, factor=0.5):
        dtype = input.dtype if isinstance(input, torch.Tensor) else torch.uint8
        return (float if dtype.is_floating_point else int)(
            get_max_value(dtype) * factor
        )

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_image):
        image = make_image(dtype=dtype, device=device)
        check_kernel(F.solarize_image, image, threshold=self._make_threshold(image))

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        video = make_video()
        check_kernel(F.solarize_video, video, threshold=self._make_threshold(video))

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
        input = make_input()
        check_functional(F.solarize, input, threshold=self._make_threshold(input))

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.solarize_image, torch.Tensor),
            (F.solarize_image, ta_tensors.Image),
            (F.solarize_image, ta_tensors.BatchImages),
            (F.solarize_video, ta_tensors.Video),
            (F.solarize_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.solarize, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        ("dtype", "threshold"), [(torch.uint8, 256), (torch.float, 1.5)]
    )
    def test_functional_error(self, dtype, threshold):
        with pytest.raises(
            TypeError,
            match="Threshold should be less or equal the maximum value of the dtype",
        ):
            F.solarize(make_image(dtype=dtype), threshold=threshold)

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
        input = make_input()
        check_transform(
            transforms.RandomSolarize(threshold=self._make_threshold(input), p=1), input
        )

    @pytest.mark.parametrize(
        "make_input", [make_batch_images_tensor, make_batch_images, make_batch_videos]
    )
    @pytest.mark.parametrize("batch_inplace", [False, True])
    @pytest.mark.parametrize("p", [0, 0.5, 1])
    def test_batch_transform(self, make_input, batch_inplace, p):
        check_batch_transform(
            transforms.RandomSolarize(
                threshold=self._make_threshold(input),
                p=p,
                batch_inplace=batch_inplace,
                batch_transform=True,
            ),
            make_input(batch_dims=(2,)),
            2,
        )

    @pytest.mark.parametrize("threshold_factor", [0.0, 0.1, 0.5, 0.9, 1.0])
    @pytest.mark.parametrize(
        "fn", [F.solarize, transform_cls_to_functional(transforms.RandomSolarize, p=1)]
    )
    def test_correctness_image(self, threshold_factor, fn):
        image = make_image(dtype=torch.uint8, device="cpu")
        threshold = self._make_threshold(image, factor=threshold_factor)

        actual = fn(image, threshold=threshold)
        expected = TVF.solarize(torch.as_tensor(image), threshold=threshold)

        assert_equal(actual, expected)

    @pytest.mark.parametrize("threshold_factor", [0.0, 0.1, 0.5, 0.9, 1.0])
    @pytest.mark.parametrize(
        "fn",
        [
            F.solarize,
            transform_cls_to_functional(
                transforms.RandomSolarize, p=1, batch_transform=True
            ),
        ],
    )
    def test_correctness_batch_images(self, threshold_factor, fn):
        image = make_batch_images(dtype=torch.uint8, device="cpu")
        threshold = self._make_threshold(image, factor=threshold_factor)

        actual = fn(image, threshold=threshold)
        expected = TVF.solarize(torch.as_tensor(image), threshold=threshold)

        assert_equal(actual, expected)


class TestAutocontrast:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel_image(self, dtype, device):
        check_kernel(F.autocontrast_image, make_image(dtype=dtype, device=device))

    def test_kernel_video(self):
        check_kernel(F.autocontrast_video, make_video())

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
        check_functional(F.autocontrast, make_input())

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.autocontrast_image, torch.Tensor),
            (F.autocontrast_image, ta_tensors.Image),
            (F.autocontrast_image, ta_tensors.BatchImages),
            (F.autocontrast_video, ta_tensors.Video),
            (F.autocontrast_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.autocontrast, kernel=kernel, input_type=input_type
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
            transforms.RandomAutocontrast(p=1),
            make_input(),
        )

    @pytest.mark.parametrize(
        "make_input", [make_batch_images_tensor, make_batch_images, make_batch_videos]
    )
    @pytest.mark.parametrize("batch_inplace", [False, True])
    @pytest.mark.parametrize("p", [0, 0.5, 1])
    def test_batch_transform(self, make_input, batch_inplace, p):
        check_batch_transform(
            transforms.RandomAutocontrast(
                p=p, batch_inplace=batch_inplace, batch_transform=True
            ),
            make_input(batch_dims=(2,)),
            2,
        )

    @pytest.mark.parametrize(
        "fn",
        [
            F.autocontrast,
            transform_cls_to_functional(transforms.RandomAutocontrast, p=1),
        ],
    )
    def test_correctness_image(self, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = F.autocontrast(torch.as_tensor(image))

        torch.testing.assert_close(actual, expected, rtol=0, atol=1)

    @pytest.mark.parametrize(
        "fn",
        [
            F.autocontrast,
            transform_cls_to_functional(
                transforms.RandomAutocontrast, p=1, batch_transform=True
            ),
        ],
    )
    def test_correctness_batch_images(self, fn):
        image = make_batch_images(dtype=torch.uint8, device="cpu")

        actual = fn(image)
        expected = F.autocontrast(torch.as_tensor(image))

        torch.testing.assert_close(actual, expected, rtol=0, atol=1)


class TestAdjustSharpness:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_image):
        check_kernel(
            F.adjust_sharpness_image,
            make_image(dtype=dtype, device=device),
            sharpness_factor=0.5,
        )

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(F.adjust_sharpness_video, make_video(), sharpness_factor=0.5)

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
        check_functional(F.adjust_sharpness, make_input(), sharpness_factor=0.5)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_sharpness_image, torch.Tensor),
            (F.adjust_sharpness_image, ta_tensors.Image),
            (F.adjust_sharpness_image, ta_tensors.BatchImages),
            (F.adjust_sharpness_video, ta_tensors.Video),
            (F.adjust_sharpness_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.adjust_sharpness, kernel=kernel, input_type=input_type
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
            transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=1), make_input()
        )

    @pytest.mark.parametrize(
        "make_input", [make_batch_images_tensor, make_batch_images, make_batch_videos]
    )
    @pytest.mark.parametrize("batch_inplace", [False, True])
    @pytest.mark.parametrize("p", [0, 0.5, 1])
    def test_batch_transform(self, make_input, batch_inplace, p):
        check_batch_transform(
            transforms.RandomAdjustSharpness(
                sharpness_factor=0.5,
                p=p,
                batch_inplace=batch_inplace,
                batch_transform=True,
            ),
            make_input(batch_dims=(2,)),
            2,
        )

    def test_functional_error(self):
        with pytest.raises(TypeError, match="can have 1 or 3 channels"):
            F.adjust_sharpness(make_image(color_space="RGBA"), sharpness_factor=0.5)

        with pytest.raises(ValueError, match="is not non-negative"):
            F.adjust_sharpness(make_image(), sharpness_factor=-1)

    @pytest.mark.parametrize("sharpness_factor", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize(
        "fn",
        [
            F.adjust_sharpness,
            transform_cls_to_functional(transforms.RandomAdjustSharpness, p=1),
        ],
    )
    def test_correctness_image(self, sharpness_factor, fn):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = fn(image, sharpness_factor=sharpness_factor)
        expected = F.adjust_sharpness(
            torch.as_tensor(image), sharpness_factor=sharpness_factor
        )

        assert_equal(actual, expected)

    @pytest.mark.parametrize("sharpness_factor", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize(
        "fn",
        [
            F.adjust_sharpness,
            transform_cls_to_functional(
                transforms.RandomAdjustSharpness, p=1, batch_transform=True
            ),
        ],
    )
    def test_correctness_image(self, sharpness_factor, fn):
        image = make_batch_images(dtype=torch.uint8, device="cpu")

        actual = fn(image, sharpness_factor=sharpness_factor)
        expected = F.adjust_sharpness(
            torch.as_tensor(image), sharpness_factor=sharpness_factor
        )

        assert_equal(actual, expected)


class TestAdjustContrast:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_image):
        check_kernel(
            F.adjust_contrast_image,
            make_image(dtype=dtype, device=device),
            contrast_factor=0.5,
        )

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", [make_batch_images])
    def test_kernel_batch_images(self, dtype, device, make_image):
        check_kernel(
            F.adjust_contrast_batch_images,
            make_image(dtype=dtype, device=device),
            contrast_factor=0.5,
        )

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(F.adjust_contrast_video, make_video(), contrast_factor=0.5)

    @pytest.mark.parametrize("make_video", [make_batch_videos])
    def test_kernel_batch_videos(self, make_video):
        check_kernel(F.adjust_contrast_batch_videos, make_video(), contrast_factor=0.5)

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
        check_functional(F.adjust_contrast, make_input(), contrast_factor=0.5)

    @pytest.mark.parametrize(
        "make_input", [make_batch_images_tensor, make_batch_images, make_batch_videos]
    )
    def test_batch_functional(self, make_input):
        check_functional(F.adjust_contrast, make_input(), contrast_factor=0.5)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_contrast_image, torch.Tensor),
            (F.adjust_contrast_image, ta_tensors.Image),
            (F.adjust_contrast_image, ta_tensors.BatchImages),
            (F.adjust_contrast_video, ta_tensors.Video),
            (F.adjust_contrast_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.adjust_contrast, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_contrast_batch_images, torch.Tensor),
            (F.adjust_contrast_batch_images, ta_tensors.BatchImages),
            (F.adjust_contrast_batch_videos, ta_tensors.BatchVideos),
        ],
    )
    def test_batch_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.adjust_contrast_batch, kernel=kernel, input_type=input_type
        )

    def test_functional_error(self):
        with pytest.raises(TypeError, match="permitted channel values are 1 or 3"):
            F.adjust_contrast(make_image(color_space="RGBA"), contrast_factor=0.5)

        with pytest.raises(ValueError, match="is not non-negative"):
            F.adjust_contrast(make_image(), contrast_factor=-1)

    def test_batch_functional_error(self):
        with pytest.raises(TypeError, match="permitted channel values are 1 or 3"):
            F.adjust_contrast_batch(
                make_batch_images(color_space="RGBA"), contrast_factor=0.5
            )

        with pytest.raises(ValueError, match="factor should be in the range"):
            F.adjust_contrast_batch(make_batch_images(), contrast_factor=-1)

    @pytest.mark.parametrize("contrast_factor", [0.1, 0.5, 1.0])
    def test_correctness_image(self, contrast_factor):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = F.adjust_contrast(image, contrast_factor=contrast_factor)
        expected = TVF.adjust_contrast(
            torch.as_tensor(image), contrast_factor=contrast_factor
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=1)

    @pytest.mark.parametrize(
        "contrast_factor", [0.1, torch.tensor([0.1]), torch.tensor([0.1, 0.5, 1.0])]
    )
    def test_correctness_batch_images(self, contrast_factor):
        images = make_batch_images(dtype=torch.uint8, device="cpu", batch_dims=(3,))

        actual = F.adjust_contrast_batch(images, contrast_factor=contrast_factor)
        tensor_images = torch.as_tensor(images)

        for i, a in enumerate(actual):
            if type(contrast_factor) is not torch.Tensor:
                c = contrast_factor
            else:
                c = (
                    contrast_factor[i].item()
                    if contrast_factor.numel() > 1
                    else contrast_factor.item()
                )

            e = F.adjust_contrast(tensor_images[i], contrast_factor=c)
            torch.testing.assert_close(a, e, rtol=0, atol=1)


class TestAdjustGamma:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_image):
        check_kernel(
            F.adjust_gamma_image, make_image(dtype=dtype, device=device), gamma=0.5
        )

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(F.adjust_gamma_video, make_video(), gamma=0.5)

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
        check_functional(F.adjust_gamma, make_input(), gamma=0.5)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_gamma_image, torch.Tensor),
            (F.adjust_gamma_image, ta_tensors.Image),
            (F.adjust_gamma_image, ta_tensors.BatchImages),
            (F.adjust_gamma_video, ta_tensors.BatchVideos),
            (F.adjust_gamma_video, ta_tensors.Video),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.adjust_gamma, kernel=kernel, input_type=input_type
        )

    def test_functional_error(self):
        with pytest.raises(
            ValueError, match="Gamma should be a non-negative real number"
        ):
            F.adjust_gamma(make_image(), gamma=-1)

    @pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("gain", [0.1, 1.0, 2.0])
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_correctness_image(self, gamma, gain, make_image):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = F.adjust_gamma(image, gamma=gamma, gain=gain)
        expected = TVF.adjust_gamma(torch.as_tensor(image), gamma=gamma, gain=gain)

        assert_equal(actual, expected)


class TestAdjustHue:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_image):
        check_kernel(
            F.adjust_hue_image, make_image(dtype=dtype, device=device), hue_factor=0.25
        )

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", [make_batch_images])
    def test_kernel_batch_images(self, dtype, device, make_image):
        check_kernel(
            F.adjust_hue_batch_images,
            make_image(dtype=dtype, device=device),
            hue_factor=0.25,
        )

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(F.adjust_hue_video, make_video(), hue_factor=0.25)

    @pytest.mark.parametrize("make_video", [make_batch_videos])
    def test_kernel_batch_videos(self, make_video):
        check_kernel(F.adjust_hue_batch_videos, make_video(), hue_factor=0.25)

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
        check_functional(F.adjust_hue, make_input(), hue_factor=0.25)

    @pytest.mark.parametrize(
        "make_input", [make_batch_images_tensor, make_batch_images, make_batch_videos]
    )
    def test_batch_functional(self, make_input):
        check_functional(F.adjust_hue_batch, make_input(), hue_factor=0.25)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_hue_image, torch.Tensor),
            (F.adjust_hue_image, ta_tensors.Image),
            (F.adjust_hue_image, ta_tensors.BatchImages),
            (F.adjust_hue_video, ta_tensors.Video),
            (F.adjust_hue_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.adjust_hue, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_hue_batch_images, torch.Tensor),
            (F.adjust_hue_batch_images, ta_tensors.BatchImages),
            (F.adjust_hue_batch_videos, ta_tensors.BatchVideos),
        ],
    )
    def test_batch_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.adjust_hue_batch, kernel=kernel, input_type=input_type
        )

    def test_functional_error(self):
        with pytest.raises(TypeError, match="permitted channel values are 1 or 3"):
            F.adjust_hue(make_image(color_space="RGBA"), hue_factor=0.25)

        for hue_factor in [-1, 1]:
            with pytest.raises(ValueError, match=re.escape("is not in [-0.5, 0.5]")):
                F.adjust_hue(make_image(), hue_factor=hue_factor)

    def test_batch_functional_error(self):
        with pytest.raises(TypeError, match="permitted channel values are 1 or 3"):
            F.adjust_hue_batch(make_batch_images(color_space="RGBA"), hue_factor=0.25)

        for hue_factor in [-1, 1]:
            with pytest.raises(
                ValueError, match=re.escape("factor should be in the range [-0.5, 0.5]")
            ):
                F.adjust_hue_batch(
                    make_batch_images(), hue_factor=hue_factor, value_check=True
                )

    @pytest.mark.parametrize("hue_factor", [-0.5, -0.3, 0.0, 0.2, 0.5])
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_correctness_image(self, hue_factor, make_image):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = F.adjust_hue(image, hue_factor=hue_factor)
        expected = TVF.adjust_hue(torch.as_tensor(image), hue_factor=hue_factor)

        mae = (actual.float() - expected.float()).abs().mean()
        assert mae < 2

    @pytest.mark.parametrize(
        "hue_factor",
        [0.1, torch.tensor([0.1]), torch.tensor([-0.5, -0.3, 0.0, 0.2, 0.5])],
    )
    def test_correctness_batch_images(self, hue_factor):
        images = make_batch_images(dtype=torch.uint8, device="cpu", batch_dims=(5,))

        actual = F.adjust_hue_batch(images, hue_factor=hue_factor)
        tensor_images = torch.as_tensor(images)

        for i, a in enumerate(actual):
            if type(hue_factor) is not torch.Tensor:
                c = hue_factor
            else:
                c = (
                    hue_factor[i].item()
                    if hue_factor.numel() > 1
                    else hue_factor.item()
                )

            e = F.adjust_hue(tensor_images[i], hue_factor=c)
            torch.testing.assert_close(a, e, rtol=0, atol=1)


class TestAdjustSaturation:
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_image):
        check_kernel(
            F.adjust_saturation_image,
            make_image(dtype=dtype, device=device),
            saturation_factor=0.5,
        )

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_image", [make_batch_images])
    def test_kernel_batch_images(self, dtype, device, make_image):
        check_kernel(
            F.adjust_saturation_batch_images,
            make_image(dtype=dtype, device=device),
            saturation_factor=0.5,
        )

    @pytest.mark.parametrize("make_video", VIDEO_MAKERS)
    def test_kernel_video(self, make_video):
        check_kernel(F.adjust_saturation_video, make_video(), saturation_factor=0.5)

    @pytest.mark.parametrize("make_video", [make_batch_videos])
    def test_kernel_batch_videos(self, make_video):
        check_kernel(
            F.adjust_saturation_batch_videos, make_video(), saturation_factor=0.5
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
        check_functional(F.adjust_saturation, make_input(), saturation_factor=0.5)

    @pytest.mark.parametrize(
        "make_input", [make_batch_images_tensor, make_batch_images, make_batch_videos]
    )
    def test_batch_functional(self, make_input):
        check_functional(F.adjust_saturation_batch, make_input(), saturation_factor=0.5)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_saturation_image, torch.Tensor),
            (F.adjust_saturation_image, ta_tensors.Image),
            (F.adjust_saturation_image, ta_tensors.BatchImages),
            (F.adjust_saturation_video, ta_tensors.Video),
            (F.adjust_saturation_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.adjust_saturation, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_saturation_batch_images, torch.Tensor),
            (F.adjust_saturation_batch_images, ta_tensors.BatchImages),
            (F.adjust_saturation_batch_videos, ta_tensors.BatchVideos),
        ],
    )
    def test_batch_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.adjust_saturation_batch, kernel=kernel, input_type=input_type
        )

    def test_functional_error(self):
        with pytest.raises(TypeError, match="permitted channel values are 1 or 3"):
            F.adjust_saturation(make_image(color_space="RGBA"), saturation_factor=0.5)

        with pytest.raises(ValueError, match="is not non-negative"):
            F.adjust_saturation(make_image(), saturation_factor=-1)

    def test_batch_functional_error(self):
        with pytest.raises(TypeError, match="permitted channel values are 1 or 3"):
            F.adjust_saturation_batch(
                make_batch_images(color_space="RGBA"), saturation_factor=0.5
            )

        with pytest.raises(ValueError, match="factor should be in the range"):
            F.adjust_saturation_batch(make_batch_images(), saturation_factor=-1)

    @pytest.mark.parametrize("saturation_factor", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("make_image", IMAGE_MAKERS)
    def test_correctness_image(self, saturation_factor, make_image):
        image = make_image(dtype=torch.uint8, device="cpu")

        actual = F.adjust_saturation(image, saturation_factor=saturation_factor)
        expected = F.adjust_saturation(
            torch.as_tensor(image), saturation_factor=saturation_factor
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=1)

    @pytest.mark.parametrize(
        "saturation_factor", [0.1, torch.tensor([0.1]), torch.tensor([0.1, 0.5, 1.0])]
    )
    def test_correctness_batch_images(self, saturation_factor):
        images = make_batch_images(dtype=torch.uint8, device="cpu", batch_dims=(3,))

        actual = F.adjust_saturation_batch(images, saturation_factor=saturation_factor)
        tensor_images = torch.as_tensor(images)

        for i, a in enumerate(actual):
            if type(saturation_factor) is not torch.Tensor:
                c = saturation_factor
            else:
                c = (
                    saturation_factor[i].item()
                    if saturation_factor.numel() > 1
                    else saturation_factor.item()
                )

            e = F.adjust_saturation(tensor_images[i], saturation_factor=c)
            torch.testing.assert_close(a, e, rtol=0, atol=1)


class TestAdjustBrightness:
    _CORRECTNESS_BRIGHTNESS_FACTORS = [0.5, 0.0, 1.0, 5.0]
    _DEFAULT_BRIGHTNESS_FACTOR = _CORRECTNESS_BRIGHTNESS_FACTORS[0]

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.adjust_brightness_image, make_image),
            (F.adjust_brightness_image, make_batch_images),
            (F.adjust_brightness_video, make_video),
            (F.adjust_brightness_video, make_batch_videos),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel(self, kernel, make_input, dtype, device):
        check_kernel(
            kernel,
            make_input(dtype=dtype, device=device),
            brightness_factor=self._DEFAULT_BRIGHTNESS_FACTOR,
        )

    @pytest.mark.parametrize(
        ("kernel", "make_input"),
        [
            (F.adjust_brightness_batch_images, make_batch_images),
            (F.adjust_brightness_batch_videos, make_batch_videos),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_kernel(self, kernel, make_input, dtype, device):
        check_kernel(
            kernel,
            make_input(dtype=dtype, device=device),
            brightness_factor=self._DEFAULT_BRIGHTNESS_FACTOR,
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
            F.adjust_brightness,
            make_input(),
            brightness_factor=self._DEFAULT_BRIGHTNESS_FACTOR,
        )

    @pytest.mark.parametrize(
        "make_input", [make_batch_images_tensor, make_batch_images, make_batch_videos]
    )
    def test_functional(self, make_input):
        check_functional(
            F.adjust_brightness_batch,
            make_input(),
            brightness_factor=self._DEFAULT_BRIGHTNESS_FACTOR,
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_brightness_image, torch.Tensor),
            (F.adjust_brightness_image, ta_tensors.Image),
            (F.adjust_brightness_image, ta_tensors.BatchImages),
            (F.adjust_brightness_video, ta_tensors.Video),
            (F.adjust_brightness_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.adjust_brightness, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.adjust_brightness_batch_images, torch.Tensor),
            (F.adjust_brightness_batch_images, ta_tensors.BatchImages),
            (F.adjust_brightness_batch_videos, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.adjust_brightness_batch, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize("brightness_factor", _CORRECTNESS_BRIGHTNESS_FACTORS)
    @pytest.mark.parametrize("make_input", [make_image, make_batch_images])
    def test_image_correctness(self, brightness_factor, make_input):
        image = make_input(dtype=torch.uint8, device="cpu")

        actual = F.adjust_brightness(image, brightness_factor=brightness_factor)
        expected = F.adjust_brightness(
            torch.as_tensor(image), brightness_factor=brightness_factor
        )

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "brightness_factor",
        [0.1, torch.tensor([0.1]), torch.tensor(_CORRECTNESS_BRIGHTNESS_FACTORS)],
    )
    def test_correctness_batch_images(self, brightness_factor):
        images = make_batch_images(dtype=torch.uint8, device="cpu", batch_dims=(4,))

        actual = F.adjust_brightness_batch(images, brightness_factor=brightness_factor)
        tensor_images = torch.as_tensor(images)

        for i, a in enumerate(actual):
            if type(brightness_factor) is not torch.Tensor:
                c = brightness_factor
            else:
                c = (
                    brightness_factor[i].item()
                    if brightness_factor.numel() > 1
                    else brightness_factor.item()
                )

            e = F.adjust_brightness(tensor_images[i], brightness_factor=c)
            torch.testing.assert_close(a, e, rtol=0, atol=1)
