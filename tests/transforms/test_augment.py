from importlib.metadata import version

import pytest
import torch
import torchvision.transforms.v2 as tv_transforms
import torchvision.transforms.v2.functional as TVF
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import torchaug.transforms as transforms
import torchaug.transforms.functional as F
from torchaug import ta_tensors

from ..utils import (
    IMAGE_MAKERS,
    VIDEO_MAKERS,
    assert_equal,
    check_batch_transform,
    check_functional,
    check_functional_kernel_signature_match,
    check_kernel,
    check_kernel_cuda_vs_cpu,
    check_transform,
    cpu_and_cuda,
    freeze_rng_state,
    make_batch_images,
    make_batch_images_tensor,
    make_batch_videos,
    make_image,
    make_image_tensor,
    make_video,
    needs_cuda,
    param_value_parametrization,
)


class TestErase:
    INPUT_SIZE = (17, 11)
    FUNCTIONAL_KWARGS = dict(
        zip(
            "ijhwv",
            [
                2,
                2,
                10,
                8,
                torch.tensor(0.0, dtype=torch.float32, device="cpu").reshape(-1, 1, 1),
            ],
        )
    )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image(self, dtype, device, make_input):
        check_kernel(
            F.erase_image,
            make_input(self.INPUT_SIZE, dtype=dtype, device=device),
            **self.FUNCTIONAL_KWARGS,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_kernel_image_inplace(self, dtype, device, make_input):
        input = make_image(self.INPUT_SIZE, dtype=dtype, device=device)
        input_version = input._version

        output_out_of_place = F.erase_image(input, **self.FUNCTIONAL_KWARGS)
        assert output_out_of_place.data_ptr() != input.data_ptr()
        assert output_out_of_place is not input

        output_inplace = F.erase_image(input, **self.FUNCTIONAL_KWARGS, inplace=True)
        assert output_inplace.data_ptr() == input.data_ptr()
        assert output_inplace._version > input_version
        assert output_inplace is input

        assert_equal(output_inplace, output_out_of_place)

    @pytest.mark.parametrize("make_input", VIDEO_MAKERS)
    def test_kernel_video(self, make_input):
        check_kernel(F.erase_video, make_input(self.INPUT_SIZE), **self.FUNCTIONAL_KWARGS)

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
        check_functional(F.erase, make_input(), **self.FUNCTIONAL_KWARGS)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.erase_image, torch.Tensor),
            (F.erase_image, ta_tensors.Image),
            (F.erase_image, ta_tensors.BatchImages),
            (F.erase_video, ta_tensors.Video),
            (F.erase_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.erase, kernel=kernel, input_type=input_type)

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
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_transform(self, make_input, device):
        input = make_input(device=device)

        check_transform(
            transforms.RandomErasing(p=1),
            input,
            check_sample_input=False,
        )

    @pytest.mark.parametrize(
        "make_input",
        [
            make_batch_images_tensor,
            make_batch_images,
            make_batch_videos,
        ],
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("batch_inplace", [False, True])
    @pytest.mark.parametrize("num_chunks", [1, 2])
    @pytest.mark.parametrize("permute_chunks", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_transform(
        self,
        make_input,
        device,
        p,
        batch_inplace,
        num_chunks,
        permute_chunks,
        batch_size,
    ):
        input = make_input(device=device, batch_dims=(batch_size,))

        check_batch_transform(
            transforms.RandomErasing(
                p=p,
                batch_transform=True,
                inplace=batch_inplace,
                num_chunks=num_chunks,
                permute_chunks=permute_chunks,
            ),
            input,
            check_sample_input=False,
            batch_size=batch_size,
        )

    def _reference_erase_image(self, image, *, i, j, h, w, v):
        mask = torch.zeros_like(image, dtype=torch.bool)
        mask[..., i : i + h, j : j + w] = True

        # The broadcasting and type casting logic is handled automagically in the kernel through indexing
        value = torch.broadcast_to(v, (*image.shape[:-2], h, w)).to(image)

        erased_image = torch.empty_like(image)
        erased_image[mask] = value.flatten()
        erased_image[~mask] = image[~mask]

        return erased_image

    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_functional_image_correctness(self, dtype, device, make_input):
        image = make_input(dtype=dtype, device=device)

        actual = F.erase(image, **self.FUNCTIONAL_KWARGS)
        expected = self._reference_erase_image(image, **self.FUNCTIONAL_KWARGS)

        assert_equal(actual, expected)

    @param_value_parametrization(
        scale=[(0.1, 0.2), [0.0, 1.0]],
        ratio=[(0.3, 0.7), [0.1, 5.0]],
        value=[0, 0.5, (0, 1, 0), [-0.2, 0.0, 1.3], "random"],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seed", list(range(5)))
    @pytest.mark.parametrize("make_input", IMAGE_MAKERS)
    def test_transform_image_correctness(self, param, value, dtype, device, seed, make_input):
        transform = transforms.RandomErasing(**{param: value}, p=1)

        image = make_input(dtype=dtype, device=device)

        with freeze_rng_state():
            torch.manual_seed(seed)

            params = transform._get_params([image], 1, [torch.tensor([0], device=image.device)])[0]

            torch.manual_seed(seed)
            actual = transform(image)

        expected = self._reference_erase_image(image, **params)

        assert_equal(actual, expected)

    def test_transform_errors(self):
        with pytest.raises(
            TypeError,
            match="Argument value should be either a number or str or a sequence",
        ):
            transforms.RandomErasing(value={})

        with pytest.raises(ValueError, match="If value is str, it should be 'random'"):
            transforms.RandomErasing(value="abc")

        with pytest.raises(TypeError, match="Scale should be a sequence"):
            transforms.RandomErasing(scale=123)

        with pytest.raises(TypeError, match="Ratio should be a sequence"):
            transforms.RandomErasing(ratio=123)

        with pytest.raises(ValueError, match="Scale should be between 0 and 1"):
            transforms.RandomErasing(scale=[-1, 2])

        transform = transforms.RandomErasing(value=[1, 2, 3, 4])

        with pytest.raises(
            ValueError,
            match="If value is a sequence, it should have either a single value",
        ):
            transform._get_params([make_image()], 1, [torch.tensor([0])])


class TestCutMixMixUp:
    class DummyDataset:
        def __init__(self, size, num_classes):
            self.size = size
            self.num_classes = num_classes
            assert size < num_classes

        def __getitem__(self, idx):
            img = torch.rand(3, 100, 100)
            label = idx  # This ensures all labels in a batch are unique and makes testing easier
            return img, label

        def __len__(self):
            return self.size

    @pytest.mark.parametrize("T", [transforms.CutMix, transforms.MixUp])
    def test_supported_input_structure(self, T):
        batch_size = 32
        num_classes = 100

        dataset = self.DummyDataset(size=batch_size, num_classes=num_classes)

        cutmix_mixup = T(num_classes=num_classes)

        dl = DataLoader(dataset, batch_size=batch_size)

        # Input sanity checks
        img, target = next(iter(dl))
        input_img_size = img.shape[-3:]
        assert isinstance(img, torch.Tensor) and isinstance(target, torch.Tensor)
        assert target.shape == (batch_size,)

        def check_output(img, target):
            assert img.shape == (batch_size, *input_img_size)
            assert target.shape == (batch_size, num_classes)
            torch.testing.assert_close(target.sum(axis=-1), torch.ones(batch_size))
            num_non_zero_labels = (target != 0).sum(axis=-1)
            assert (num_non_zero_labels == 2).all()

        # After Dataloader, as unpacked input
        img, target = next(iter(dl))
        assert target.shape == (batch_size,)
        img, target = cutmix_mixup(img, target)
        check_output(img, target)

        # After Dataloader, as packed input
        packed_from_dl = next(iter(dl))
        assert isinstance(packed_from_dl, list)
        img, target = cutmix_mixup(packed_from_dl)
        check_output(img, target)

        # As collation function. We expect default_collate to be used by users.
        def collate_fn_1(batch):
            return cutmix_mixup(default_collate(batch))

        def collate_fn_2(batch):
            return cutmix_mixup(*default_collate(batch))

        for collate_fn in (collate_fn_1, collate_fn_2):
            dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
            img, target = next(iter(dl))
            check_output(img, target)

    @needs_cuda
    @pytest.mark.parametrize("T", [transforms.CutMix, transforms.MixUp])
    def test_cpu_vs_gpu(self, T):
        num_classes = 10
        batch_size = 3
        H, W = 12, 12

        imgs = torch.rand(batch_size, 3, H, W)
        labels = torch.randint(0, num_classes, (batch_size,))
        cutmix_mixup = T(alpha=0.5, num_classes=num_classes)

        check_kernel_cuda_vs_cpu(cutmix_mixup, imgs, labels, rtol=None, atol=None)

    @pytest.mark.parametrize("T", [transforms.CutMix, transforms.MixUp])
    def test_error(self, T):
        num_classes = 10
        batch_size = 9

        imgs = torch.rand(batch_size, 3, 12, 12)
        cutmix_mixup = T(alpha=0.5, num_classes=num_classes)

        for input_with_bad_type in (
            ta_tensors.Image(torch.rand(3, 12, 12)),
            ta_tensors.Video(torch.rand(3, 12, 12, 12)),
            ta_tensors.Mask(torch.rand(12, 12)),
            ta_tensors.BoundingBoxes(torch.rand(2, 4), format="XYXY", canvas_size=12),
        ):
            with pytest.raises(ValueError, match="supports only batch of images or videos."):
                cutmix_mixup(input_with_bad_type)

        with pytest.raises(ValueError, match="Could not infer where the labels are"):
            cutmix_mixup({"img": imgs, "Nothing_else": 3})

        with pytest.raises(ValueError, match="labels tensor should be of shape"):
            # Note: the error message isn't ideal, but that's because the label heuristic found the img as the label
            # It's OK, it's an edge-case. The important thing is that this fails loudly instead of passing silently
            cutmix_mixup(imgs)

        with pytest.raises(ValueError, match="When using the default labels_getter"):
            cutmix_mixup(imgs, "not_a_tensor")

        with pytest.raises(ValueError, match="labels tensor should be of shape"):
            cutmix_mixup(imgs, torch.randint(0, 2, size=(2, 3)))

        with pytest.raises(ValueError, match="Expected a batched input with 4 dims"):
            cutmix_mixup(imgs[None, None], torch.randint(0, num_classes, size=(batch_size,)))

        with pytest.raises(ValueError, match="does not match the batch size of the labels"):
            cutmix_mixup(imgs, torch.randint(0, num_classes, size=(batch_size + 1,)))

        with pytest.raises(ValueError, match="labels tensor should be of shape"):
            # The purpose of this check is more about documenting the current
            # behaviour of what happens on a Compose(), rather than actually
            # asserting the expected behaviour. We may support Compose() in the
            # future, e.g. for 2 consecutive CutMix?
            labels = torch.randint(0, num_classes, size=(batch_size,))
            transforms.Compose([cutmix_mixup, cutmix_mixup])(imgs, labels)


@pytest.mark.parametrize("key", ("labels", "LABELS", "LaBeL", "SOME_WEIRD_KEY_THAT_HAS_LABeL_IN_IT"))
@pytest.mark.parametrize("sample_type", (tuple, list, dict))
def test_labels_getter_default_heuristic(key, sample_type):
    labels = torch.arange(10)
    sample = {key: labels, "another_key": "whatever"}
    if sample_type is not dict:
        sample = sample_type((None, sample, "whatever_again"))
    assert tv_transforms._utils._find_labels_default_heuristic(sample) is labels

    if key.lower() != "labels":
        # If "labels" is in the dict (case-insensitive),
        # it takes precedence over other keys which would otherwise be a match
        d = {key: "something_else", "labels": labels}
        assert tv_transforms._utils._find_labels_default_heuristic(d) is labels


@pytest.mark.skipif(version("torchvision") < "0.18.0", reason="requires torchvision>=0.18.0")
class TestJPEG:
    @pytest.mark.parametrize("quality", [5, 75])
    @pytest.mark.parametrize("color_space", ["RGB", "GRAY"])
    def test_kernel_image(self, quality, color_space):
        check_kernel(F.jpeg_image, make_image(color_space=color_space), quality=quality)

    def test_kernel_video(self):
        check_kernel(F.jpeg_video, make_video(), quality=5)

    @pytest.mark.parametrize(
        "make_input", [make_image_tensor, make_image, make_video, make_batch_images, make_batch_videos]
    )
    def test_functional(self, make_input):
        check_functional(F.jpeg, make_input(), quality=5)

    @pytest.mark.parametrize(
        ("kernel", "input_type"),
        [
            (F.jpeg_image, torch.Tensor),
            (F.jpeg_image, ta_tensors.Image),
            (F.jpeg_image, ta_tensors.BatchImages),
            (F.jpeg_video, ta_tensors.Video),
            (F.jpeg_video, ta_tensors.BatchVideos),
        ],
    )
    def test_functional_signature(self, kernel, input_type):
        check_functional_kernel_signature_match(F.jpeg, kernel=kernel, input_type=input_type)

    @pytest.mark.parametrize(
        "make_input", [make_image_tensor, make_image, make_batch_images, make_video, make_batch_videos]
    )
    @pytest.mark.parametrize("quality", [5, (10, 20)])
    @pytest.mark.parametrize("color_space", ["RGB", "GRAY"])
    def test_transform(self, make_input, quality, color_space):
        check_transform(transforms.JPEG(quality=quality), make_input(color_space=color_space))

    @pytest.mark.parametrize("quality", [5])
    def test_functional_image_correctness(self, quality):
        image = make_image()

        actual = F.jpeg(image, quality=quality)
        expected = TVF.jpeg(torch.as_tensor(image), quality=quality)

        # NOTE: this will fail if torchvision and Pillow use different JPEG encoder/decoder
        torch.testing.assert_close(actual, expected, rtol=0, atol=1)

    @pytest.mark.parametrize("quality", [5, (10, 20)])
    @pytest.mark.parametrize("color_space", ["RGB", "GRAY"])
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_transform_image_correctness(self, quality, color_space, seed):
        image = make_image(color_space=color_space)

        transform = transforms.JPEG(quality=quality)

        with freeze_rng_state():
            torch.manual_seed(seed)
            actual = transform(image)

            torch.manual_seed(seed)
            quality = transform._get_params([image], 1, (torch.tensor([0], device=image.device),))[0]["quality"]
            expected = TVF.jpeg(torch.as_tensor(image), quality=quality)

        torch.testing.assert_close(actual, expected, rtol=0, atol=1)

    @pytest.mark.parametrize("quality", [5, (10, 20)])
    @pytest.mark.parametrize("seed", list(range(10)))
    def test_transform_get_params_bounds(self, quality, seed):
        transform = transforms.JPEG(quality=quality)

        with freeze_rng_state():
            torch.manual_seed(seed)
            params = transform._get_params([], 1, (torch.tensor([0], device="cpu"),))[0]

        if isinstance(quality, int):
            assert params["quality"] == quality
        else:
            assert quality[0] <= params["quality"] <= quality[1]

    @pytest.mark.parametrize("quality", [[0], [0, 0, 0]])
    def test_transform_sequence_len_error(self, quality):
        with pytest.raises(ValueError, match="quality should be a sequence of length 2"):
            transforms.JPEG(quality=quality)

    @pytest.mark.parametrize("quality", [-1, 0, 150])
    def test_transform_invalid_quality_error(self, quality):
        with pytest.raises(ValueError, match="quality must be an integer from 1 to 100"):
            transforms.JPEG(quality=quality)
