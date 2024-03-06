import functools
import re

import pytest
import torchaug.transforms as transforms
import torchaug.transforms.functional as F
from torch import nn

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


class TestContainerTransforms:
    class BuiltinTransform(transforms.Transform):
        def _transform(self, inpt, params):
            return inpt

    class PackedInputTransform(nn.Module):
        def forward(self, sample):
            assert len(sample) == 2
            return sample

    class UnpackedInputTransform(nn.Module):
        def forward(self, image, label):
            return image, label

    @pytest.mark.parametrize(
        "transform_cls",
        [
            transforms.Compose,
            functools.partial(transforms.RandomApply, p=1),
            transforms.RandomOrder,
        ],
    )
    @pytest.mark.parametrize(
        "wrapped_transform_clss",
        [
            [BuiltinTransform],
            [PackedInputTransform],
            [UnpackedInputTransform],
            [BuiltinTransform, BuiltinTransform],
            [PackedInputTransform, PackedInputTransform],
            [UnpackedInputTransform, UnpackedInputTransform],
            [BuiltinTransform, PackedInputTransform, BuiltinTransform],
            [BuiltinTransform, UnpackedInputTransform, BuiltinTransform],
            [PackedInputTransform, BuiltinTransform, PackedInputTransform],
            [UnpackedInputTransform, BuiltinTransform, UnpackedInputTransform],
        ],
    )
    @pytest.mark.parametrize("unpack", [True, False])
    @pytest.mark.parametrize("batch", [True, False])
    def test_packed_unpacked(
        self, transform_cls, wrapped_transform_clss, unpack, batch
    ):
        needs_packed_inputs = any(
            issubclass(cls, self.PackedInputTransform) for cls in wrapped_transform_clss
        )
        needs_unpacked_inputs = any(
            issubclass(cls, self.UnpackedInputTransform)
            for cls in wrapped_transform_clss
        )
        assert not (needs_packed_inputs and needs_unpacked_inputs)

        transform = transform_cls([cls() for cls in wrapped_transform_clss])

        image = make_image() if not batch else make_batch_images()
        label = 3
        packed_input = (image, label)

        def call_transform():
            if unpack:
                return transform(*packed_input)
            else:
                return transform(packed_input)

        if needs_unpacked_inputs and not unpack:
            with pytest.raises(
                TypeError, match="missing 1 required positional argument"
            ):
                call_transform()
        elif needs_packed_inputs and unpack:
            with pytest.raises(
                TypeError, match="takes 2 positional arguments but 3 were given"
            ):
                call_transform()
        else:
            output = call_transform()

            assert isinstance(output, tuple) and len(output) == 2
            assert output[0] is image
            assert output[1] is label

    @pytest.mark.parametrize("batch", [True, False])
    def test_compose(self, batch):
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1, batch_transform=batch),
                transforms.RandomVerticalFlip(p=1, batch_transform=batch),
            ]
        )

        input = make_image() if not batch else make_batch_images()

        actual = check_transform(transform, input)
        expected = F.vertical_flip(F.horizontal_flip(input))

        assert_equal(actual, expected)

    @pytest.mark.parametrize("p", [0.0, 1.0])
    @pytest.mark.parametrize("sequence_type", [list, nn.ModuleList])
    @pytest.mark.parametrize("batch", [True, False])
    def test_random_apply(self, p, sequence_type, batch):
        transform = transforms.RandomApply(
            sequence_type(
                [
                    transforms.RandomHorizontalFlip(p=1, batch_transform=batch),
                    transforms.RandomVerticalFlip(p=1, batch_transform=batch),
                ]
            ),
            p=p,
        )

        input = make_image_tensor() if not batch else make_batch_images_tensor()
        output = check_transform(
            transform,
            input,
            batch=batch,
        )

        if p == 1:
            assert_equal(output, F.vertical_flip(F.horizontal_flip(input)))
        else:
            assert output is input

    @pytest.mark.parametrize("p", [(0, 1), (1, 0)])
    @pytest.mark.parametrize("batch", [True, False])
    def test_random_choice(self, p, batch):
        transform = transforms.RandomChoice(
            [
                transforms.RandomHorizontalFlip(p=1, batch_transform=batch),
                transforms.RandomVerticalFlip(p=1, batch_transform=batch),
            ],
            p=p,
        )

        input = make_image() if not batch else make_batch_images()
        output = check_transform(transform, input)

        p_horz, p_vert = p
        if p_horz:
            assert_equal(output, F.horizontal_flip(input))
        else:
            assert_equal(output, F.vertical_flip(input))

    @pytest.mark.parametrize("batch", [True, False])
    def test_random_order(self, batch):
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1, batch_transform=batch),
                transforms.RandomVerticalFlip(p=1, batch_transform=batch),
            ]
        )

        input = make_image() if not batch else make_batch_images()

        actual = check_transform(transform, input, batch=batch)
        # We can't really check whether the transforms are actually applied in random order. However, horizontal and
        # vertical flip are commutative. Meaning, even under the assumption that the transform applies them in random
        # order, we can use a fixed order to compute the expected value.
        expected = F.vertical_flip(F.horizontal_flip(input))

        assert_equal(actual, expected)

    def test_errors(self):
        for cls in [
            transforms.Compose,
            transforms.RandomChoice,
            transforms.RandomOrder,
        ]:
            with pytest.raises(
                TypeError, match="Argument transforms should be a sequence of callables"
            ):
                cls(lambda x: x)

        with pytest.raises(ValueError, match="at least one transform"):
            transforms.Compose([])

        for p in [-1, 2]:
            with pytest.raises(
                ValueError, match=re.escape("value in the interval [0.0, 1.0]")
            ):
                transforms.RandomApply([lambda x: x], p=p)

        for transforms_, p in [([lambda x: x], []), ([], [1.0])]:
            with pytest.raises(
                ValueError, match="Length of p doesn't match the number of transforms"
            ):
                transforms.RandomChoice(transforms_, p=p)
