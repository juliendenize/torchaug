import pytest
import torch
import torchaug.transforms as transforms
import torchaug.transforms.functional as F
import torchvision.transforms.v2.functional as TVF
from torchaug import ta_tensors

from ..utils import (
    assert_equal,
    check_functional,
    check_functional_kernel_signature_match,
    check_kernel,
    check_transform,
    cpu_and_cuda,
    make_batch_videos,
    make_video,
    make_video_tensor,
    transform_cls_to_functional,
    VIDEO_MAKERS,
)


class TestUniformTemporalSubsample:
    @pytest.mark.parametrize("make_input", VIDEO_MAKERS)
    def test_kernel_video(self, make_input):
        check_kernel(F.uniform_temporal_subsample_video, make_input(), num_samples=2)

    @pytest.mark.parametrize(
        "make_input", [make_video_tensor, make_video, make_batch_videos]
    )
    def test_functional(self, make_input):
        check_functional(F.uniform_temporal_subsample, make_input(), num_samples=2)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.uniform_temporal_subsample_video, torch.Tensor),
            (F.uniform_temporal_subsample_video, ta_tensors.Video),
            (F.uniform_temporal_subsample_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(
            F.uniform_temporal_subsample, kernel=kernel, input_type=input_type
        )

    @pytest.mark.parametrize(
        "make_input", [make_video_tensor, make_video, make_batch_videos]
    )
    def test_transform(self, make_input):
        check_transform(
            transforms.UniformTemporalSubsample(num_samples=2), make_input()
        )

    CORRECTNESS_NUM_FRAMES = 5

    @pytest.mark.parametrize("num_samples", list(range(1, CORRECTNESS_NUM_FRAMES + 1)))
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize(
        "fn",
        [
            F.uniform_temporal_subsample,
            transform_cls_to_functional(transforms.UniformTemporalSubsample),
        ],
    )
    @pytest.mark.parametrize("make_input", VIDEO_MAKERS)
    def test_video_correctness(self, num_samples, dtype, device, fn, make_input):
        video = make_input(
            num_frames=self.CORRECTNESS_NUM_FRAMES, dtype=dtype, device=device
        )

        actual = fn(video, num_samples=num_samples)
        expected = TVF.uniform_temporal_subsample_video(video, num_samples=num_samples)

        assert_equal(actual, expected)
