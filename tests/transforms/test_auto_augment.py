import pytest
import torch
import torchaug.transforms as transforms
import torchaug.transforms.functional as F
import torchvision.transforms.v2 as tv_transforms
from torchaug import ta_tensors
from torchaug.transforms.functional._utils import is_pure_tensor

from ..utils import (
    BATCH_IMAGES_TENSOR_AND_MAKERS,
    check_transform,
    check_type,
    CORRECTNESS_FILLS,
    cpu_and_cuda,
    freeze_rng_state,
    IMAGE_MAKERS,
    make_batch_images,
    make_batch_images_tensor,
    make_batch_videos,
    make_image,
    make_image_tensor,
    make_video,
)


class TestAutoAugmentTransforms:
    # These transforms have a lot of branches in their `forward()` passes which are conditioned on random sampling.
    # It's typically very hard to test the effect on some parameters without heavy mocking logic.
    # This class adds correctness tests for the kernels that are specific to those transforms. The rest of kernels, e.g.
    # rotate, are tested in their respective classes. The rest of the tests here are mostly smoke tests.

    @pytest.mark.parametrize(
        "transform_id", ["ShearX", "ShearY", "TranslateX", "TranslateY"]
    )
    @pytest.mark.parametrize("magnitude", [0.3, -0.2, 0.0])
    @pytest.mark.parametrize(
        "interpolation",
        [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR],
    )
    @pytest.mark.parametrize("fill", CORRECTNESS_FILLS)
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_correctness_shear_translate(
        self, transform_id, magnitude, interpolation, fill, make_input
    ):
        # ShearX/Y and TranslateX/Y are the only ops that are native to the AA transforms. They are modeled after the
        # reference implementation:
        # https://github.com/tensorflow/models/blob/885fda091c46c59d6c7bb5c7e760935eacc229da/research/autoaugment/augmentation_transforms.py#L273-L362
        # All other ops are checked in their respective dedicated tests.

        image = make_input(dtype=torch.uint8, device="cpu")

        if "Translate" in transform_id:
            # For TranslateX/Y magnitude is a value in pixels
            magnitude *= min(F.get_size(image))

        actual = transforms.AutoAugment()._apply_image_or_video_transform(
            image,
            transform_id=transform_id,
            magnitude=magnitude,
            interpolation=interpolation,
            fill={type(image): fill},
        )
        expected = tv_transforms.AutoAugment()._apply_image_or_video_transform(
            image,
            transform_id=transform_id,
            magnitude=magnitude,
            interpolation=interpolation,
            fill={type(image): fill},
        )

        if "Shear" in transform_id:
            mae = (actual.float() - expected.float()).abs().mean()
            assert mae < (
                12 if interpolation is transforms.InterpolationMode.NEAREST else 5
            )
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=1)

    def _sample_input_adapter(self, transform, input, device):
        adapted_input = {}
        image_or_video_found = False
        for key, value in input.items():
            if isinstance(
                value,
                (
                    ta_tensors.BoundingBoxes,
                    ta_tensors.Mask,
                    ta_tensors.BatchBoundingBoxes,
                    ta_tensors.BatchMasks,
                ),
            ):
                # AA transforms don't support bounding boxes or masks
                continue
            elif check_type(
                value,
                (
                    ta_tensors.Image,
                    ta_tensors.Video,
                    is_pure_tensor,
                    ta_tensors.BatchImages,
                    ta_tensors.BatchVideos,
                ),
            ):
                if image_or_video_found:
                    # AA transforms only support a single image or video
                    continue
                image_or_video_found = True
            adapted_input[key] = value
        return adapted_input

    @pytest.mark.parametrize(
        "transform",
        [
            transforms.AutoAugment(),
            transforms.RandAugment(),
            transforms.TrivialAugmentWide(),
            transforms.AugMix(),
        ],
    )
    @pytest.mark.parametrize(
        "make_input",
        [
            make_image_tensor,
            make_image,
            make_video,
            make_batch_images_tensor,
            make_batch_images,
            make_batch_videos,
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform_smoke(self, transform, make_input, dtype, device):
        if (
            type(transform) is transforms.AugMix
            and make_input not in BATCH_IMAGES_TENSOR_AND_MAKERS
        ):
            pytest.skip("AugMix only supports batched input")

        input = make_input(dtype=dtype, device=device)

        with freeze_rng_state():
            # By default every test starts from the same random seed. This leads to minimal coverage of the sampling
            # that happens inside forward(). To avoid calling the transform multiple times to achieve higher coverage,
            # we build a reproducible random seed from the input type, dtype, and device.
            torch.manual_seed(hash((make_input, dtype, device)))

            # For v2, we changed the random sampling of the AA transforms. This makes it impossible to compare the v1
            # and v2 outputs without complicated mocking and monkeypatching. Thus, we skip the v1 compatibility checks
            # here and only check if we can script the v2 transform and subsequently call the result.
            check_transform(
                transform,
                input,
                check_sample_input=self._sample_input_adapter,
            )

    def test_auto_augment_policy_error(self):
        with pytest.raises(ValueError, match="provided policy"):
            transforms.AutoAugment(policy=None)

    @pytest.mark.parametrize("severity", [0, 11])
    def test_aug_mix_severity_error(self, severity):
        with pytest.raises(ValueError, match="severity must be between"):
            transforms.AugMix(severity=severity)
