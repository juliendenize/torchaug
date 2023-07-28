import pytest
import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as F_tv
from torchvision.transforms.functional import InterpolationMode

import torchaug.batch_transforms as transforms
import torchaug.transforms as mono_transforms


def test_batch_random_apply():
    torch.manual_seed(28)

    # Checking if BatchRandomApply can be printed as string
    transforms.BatchRandomApply(
        [mono_transforms.Normalize([225, 225, 225], [0.25, 0.25, 0.25])], 0.5
    ).__repr__()

    imgs = torch.randn(4, 3, 8, 8)
    indices_to_apply = torch.tensor([2, 1])
    torchvision_out = imgs.clone()
    torchvision_out[indices_to_apply] = F_tv.normalize(
        imgs[indices_to_apply], [225, 225, 225], [0.25, 0.25, 0.25]
    )
    out = transforms.BatchRandomApply(
        [mono_transforms.Normalize([225, 225, 225], [0.25, 0.25, 0.25])], 0.5
    )(imgs)

    # test consistency with Torchvision
    torch.testing.assert_close(out, torchvision_out)

    # test not inplace
    transforms.BatchRandomApply(
        [mono_transforms.Normalize([225, 225, 225], [0.25, 0.25, 0.25])],
        0.5,
        inplace=False,
    )(imgs)

    # test p = 0. and p = 1.
    torch.testing.assert_close(
        transforms.BatchRandomApply(
            [mono_transforms.Normalize([225, 225, 225], [0.25, 0.25, 0.25])], 0.0
        )(imgs),
        imgs,
    )
    torch.testing.assert_close(
        transforms.BatchRandomApply(
            [mono_transforms.Normalize([225, 225, 225], [0.25, 0.25, 0.25])],
            1.0,
            inplace=False,
        )(imgs),
        F_tv.normalize(imgs, [225, 225, 225], [0.25, 0.25, 0.25]),
    )


def test_batch_random_color_jitter():
    torch.manual_seed(28)

    # Checking if BatchRandomColorJitter can be printed as string
    transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 0).__repr__()
    transforms.BatchRandomColorJitter(None, None, None, None, 0.5, 0).__repr__()

    imgs = torch.randn(8, 3, 8, 8)
    indices_to_apply = torch.tensor([[1, 0], [7, 5]])
    orders = torch.tensor([[1, 0, 2, 3], [2, 0, 1, 3]])
    brightnesss = torch.tensor([[0.5807553, 1.0611039], [1.0105517, 0.9287435]])
    contrasts = torch.tensor([[0.8417916, 0.9093333], [0.8968358, 1.4652505]])
    saturations = torch.tensor([[1.2338004, 1.3543849], [0.5746976, 0.6757473]])
    hues = torch.tensor([[0.0161706, 0.0335830], [-0.0669264, -0.0657860]])

    torchvision_out = imgs.clone()

    for indices, order, brightnesss, contrast, saturation, hue in zip(
        indices_to_apply, orders, brightnesss, contrasts, saturations, hues
    ):
        for fn_id in order:
            for i, b, c, s, h in zip(indices, brightnesss, contrast, saturation, hue):
                if fn_id == 0 and b is not None:
                    torchvision_out[i] = F_tv.adjust_brightness(torchvision_out[i], b)
                elif fn_id == 1 and c is not None:
                    torchvision_out[i] = F_tv.adjust_contrast(torchvision_out[i], c)
                elif fn_id == 2 and s is not None:
                    torchvision_out[i] = F_tv.adjust_saturation(torchvision_out[i], s)
                elif fn_id == 3 and h is not None:
                    torchvision_out[i] = F_tv.adjust_hue(torchvision_out[i], h)
    out = transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 2)(imgs)

    # test consistency with Torchvision
    torch.testing.assert_close(out, torchvision_out)

    # test different num_random calls
    transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 0)(imgs)
    transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, -1)(imgs)
    transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 16)(imgs)

    # test p = 0. and p = 1.
    torch.testing.assert_close(
        transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.0, 16)(imgs), imgs
    )
    transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 1.0, 16)(imgs)

    # test not inplace
    transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, inplace=False)(imgs)

    # test wrong num rand calls
    with pytest.raises(
        ValueError,
        match="num_rand_calls attribute should be an int superior to -1, ahah given.",
    ):
        transforms.BatchRandomColorJitter(num_rand_calls="ahah")
    with pytest.raises(
        ValueError,
        match="num_rand_calls attribute should be an int superior to -1, -2 given.",
    ):
        transforms.BatchRandomColorJitter(num_rand_calls=-2)


def test_batch_random_gaussian_blur():
    torch.manual_seed(28)

    # Checking if BatchRandomGaussianBlur can be printed as string
    transforms.BatchRandomGaussianBlur([3, 3], [0.1, 2.0], 0.5).__repr__()

    imgs = torch.randn(4, 3, 8, 8)
    indices_to_apply = torch.tensor([2, 1])
    sigmas = [[0.7487103, 0.7487103], [0.8353596, 0.8353596]]
    torchvision_out = imgs.clone()

    for indice, sigma in zip(indices_to_apply, sigmas):
        torchvision_out[indice] = F_tv.gaussian_blur(imgs[indice], [3, 3], sigma)

    out = transforms.BatchRandomGaussianBlur([3, 3], [0.1, 2.0], 0.5)(imgs)

    # test consistency with Torchvision
    torch.testing.assert_close(out, torchvision_out)

    # test not inplace
    transforms.BatchRandomGaussianBlur([3, 3], [0.1, 2.0], 0.5, inplace=False)(imgs)

    # test p = 0. and p = 1.
    torch.testing.assert_close(
        transforms.BatchRandomGaussianBlur([3, 3], [0.1, 2.0], 0.0)(imgs), imgs
    )
    transforms.BatchRandomGaussianBlur([3, 3], [0.1, 2.0], 1.0)(imgs)

    # test various type kernel
    transforms.BatchRandomGaussianBlur(3, [0.1, 2.0], 0.5, inplace=False)
    transforms.BatchRandomGaussianBlur([3], [0.1, 2.0], 0.5, inplace=False)

    # test various type sigma
    transforms.BatchRandomGaussianBlur([3], 1, 0.5, inplace=False)
    transforms.BatchRandomGaussianBlur([3], 1.0, 0.5, inplace=False)

    # test wrong kernel value
    with pytest.raises(
        ValueError, match="Kernel size value should be an odd and positive number."
    ):
        transforms.BatchRandomGaussianBlur(0, [0.1, 2.0], 0.5, inplace=False)
        transforms.BatchRandomGaussianBlur(-1, [0.1, 2.0], 0.5, inplace=False)

    # test sigma number is inferior or equal to 0.
    with pytest.raises(
        ValueError, match="If sigma is a single number, it must be positive."
    ):
        transforms.BatchRandomGaussianBlur(3, 0, 0.5, inplace=False)
        transforms.BatchRandomGaussianBlur(3, 0.0, 0.5, inplace=False)
        transforms.BatchRandomGaussianBlur(3, -1, 0.5, inplace=False)
        transforms.BatchRandomGaussianBlur(3, -1.0, 0.5, inplace=False)

    # test sigma sequence wrong
    with pytest.raises(
        ValueError,
        match=r"sigma values should be positive and of the form \(min, max\).",
    ):
        transforms.BatchRandomGaussianBlur(3, [2.0, 1.0], 0.5, inplace=False)
        transforms.BatchRandomGaussianBlur(3, [0.0, 1.0], 0.5, inplace=False)

    # test sigma wrong type
    with pytest.raises(
        ValueError,
        match="sigma should be a single number or a list/tuple with length 2.",
    ):
        transforms.BatchRandomGaussianBlur(3, [2.0, 3.0, 4.0], 0.5, inplace=False)
        transforms.BatchRandomGaussianBlur(3, "ahah", 0.5, inplace=False)


def test_batch_random_gray_scale():
    torch.manual_seed(28)

    # Checking if BatchRandomGrayScale can be printed as string
    transforms.BatchRandomGrayScale(0.5).__repr__()

    imgs = torch.randn(4, 3, 8, 8)
    indices_to_apply = torch.tensor([2, 1])

    torchvision_out = imgs.clone()
    torchvision_out[indices_to_apply] = F_tv.rgb_to_grayscale(imgs[indices_to_apply], 3)

    out = transforms.BatchRandomGrayScale(0.5)(imgs)

    # test consistency with Torchvision
    torch.testing.assert_close(out, torchvision_out)

    # test not inplace
    transforms.BatchRandomGrayScale(0.5, inplace=False)(imgs)

    # test p = 0. and p = 1.
    torch.testing.assert_close(transforms.BatchRandomGrayScale(0.0)(imgs), imgs)
    torch.testing.assert_close(
        transforms.BatchRandomGrayScale(1.0)(imgs), F_tv.rgb_to_grayscale(imgs, 3)
    )


def test_batch_random_horizontal_flip():
    torch.manual_seed(28)

    # Checking if BatchRandomHorizontalFlip can be printed as string
    transforms.BatchRandomHorizontalFlip(0.5).__repr__()

    imgs = torch.randn(4, 3, 8, 8)
    indices_to_apply = torch.tensor([2, 1])

    torchvision_out = imgs.clone()
    torchvision_out[indices_to_apply] = F_tv.hflip(imgs[indices_to_apply])

    out = transforms.BatchRandomHorizontalFlip(0.5)(imgs)

    # test consistency with Torchvision
    torch.testing.assert_close(out, torchvision_out)

    # test not inplace
    transforms.BatchRandomHorizontalFlip(0.5, inplace=False)(imgs)

    # test p = 0. and p = 1.
    torch.testing.assert_close(transforms.BatchRandomHorizontalFlip(0.0)(imgs), imgs)
    transforms.BatchRandomHorizontalFlip(1.0)(imgs)


def test_batch_random_resized_crop():
    torch.manual_seed(28)

    # Checking if BatchRandomResizedCrop can be printed as string
    transforms.BatchRandomResizedCrop(
        4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=2
    ).__repr__()

    imgs = torch.randn(8, 3, 8, 8)
    indices_to_apply = torch.tensor([0, 1, 7, 5, 6, 4, 2, 3])

    i_s = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
    j_s = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2])
    h_s = torch.tensor([7, 7, 7, 7, 4, 4, 4, 4])
    w_s = torch.tensor([8, 8, 8, 8, 4, 4, 4, 4])

    torchvision_out = torch.empty((8, 3, 4, 4))

    for idx, i, j, h, w in zip(indices_to_apply, i_s, j_s, h_s, w_s):
        torchvision_out[idx] = F_tv.resized_crop(
            imgs[idx],
            i,
            j,
            h,
            w,
            [4, 4],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

    out = transforms.BatchRandomResizedCrop(
        4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=2
    )(imgs)

    # test consistency with Torchvision
    torch.testing.assert_close(out, torchvision_out)

    # test different num rand calls
    transforms.BatchRandomResizedCrop(4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=-1)(
        imgs
    )
    transforms.BatchRandomResizedCrop(4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=0)(
        imgs
    )
    transforms.BatchRandomResizedCrop(4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=16)(
        imgs
    )

    # test size as list of one int
    transforms.BatchRandomResizedCrop(
        [4], (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=16
    )(imgs)

    # test wrong num rand calls
    with pytest.raises(
        ValueError,
        match="num_rand_calls attribute should be an int superior to -1, ahah given.",
    ):
        transforms.BatchRandomResizedCrop(
            4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls="ahah"
        )
    with pytest.raises(
        ValueError,
        match="num_rand_calls attribute should be an int superior to -1, -2 given.",
    ):
        transforms.BatchRandomResizedCrop(
            4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=-2
        )

    # test wrong size
    with pytest.raises(
        TypeError,
        match="size should be a int or a sequence of int. Got 4..",
    ):
        transforms.BatchRandomResizedCrop(
            4.0, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=16
        )
    with pytest.raises(
        TypeError,
        match="size should be a int or a sequence of int. Got ahah.",
    ):
        transforms.BatchRandomResizedCrop(
            "ahah", (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=16
        )


def test_batch_random_solarize():
    torch.manual_seed(28)

    # Checking if BatchRandomSolarize can be printed as string
    transforms.BatchRandomSolarize(0.5, 0.5).__repr__()

    imgs = torch.randn(4, 3, 8, 8)
    indices_to_apply = torch.tensor([2, 1])
    torchvision_out = imgs.clone()
    torchvision_out[indices_to_apply] = F_tv.solarize(imgs[indices_to_apply], 0.5)
    out = transforms.BatchRandomSolarize(0.5, 0.5)(imgs)

    # test consistency with Torchvision
    torch.testing.assert_close(out, torchvision_out)

    # test not inplace
    transforms.BatchRandomSolarize(0.5, 0.5, inplace=False)(imgs)

    # test p = 0. and p = 1.
    torch.testing.assert_close(transforms.BatchRandomSolarize(0.5, 0.0)(imgs), imgs)
    torch.testing.assert_close(
        transforms.BatchRandomSolarize(0.5, 1.0)(imgs), F_tv.solarize(imgs, 0.5)
    )


def test_batch_video_wrapper():
    torch.manual_seed(28)

    transform = mono_transforms.Normalize(
        (0.5,), (0.5,), inplace=False, value_check=True
    )

    # test if BatchVideoWrapper can be printed as string
    transforms.BatchVideoWrapper(transform=transform).__repr__()

    # test CTHW format
    tensor = torch.rand((2, 3, 2, 16, 16))
    torchvision_out = tv_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    out = transforms.BatchVideoWrapper(transform=transform)(tensor)
    torch.testing.assert_close(out, torchvision_out)

    # test TCHW format
    torchvision_out = tv_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    out = transforms.BatchVideoWrapper(transform=transform, video_format="TCHW")(
        tensor.permute(0, 2, 1, 3, 4)
    )
    torch.testing.assert_close(out.permute(0, 2, 1, 3, 4), torchvision_out)

    # test same_on_frames
    image = torch.randn((3, 224, 224))
    video = torch.stack([image, image])
    batch_video = torch.stack([video, video])
    out = transforms.BatchVideoWrapper(
        transform=transforms.BatchRandomColorJitter(0.5, p=1.0),
        video_format="TCHW",
        same_on_frames=True,
    )(batch_video)
    torch.testing.assert_close(out[:, 0], out[:, 1])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out[0], out[1])
    out = transforms.BatchVideoWrapper(
        transform=transforms.BatchRandomColorJitter(0.5, p=1.0),
        video_format="TCHW",
        same_on_frames=False,
    )(batch_video)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out[:, 0], out[:, 1])

    # test wrong video_format
    with pytest.raises(
        ValueError, match="video_format should be either 'CTHW' or 'TCHW'. Got ahah."
    ):
        transforms.BatchVideoWrapper(transform=transform, video_format="ahah")(tensor)

    # test wrong tensor dimension
    tensor = torch.rand((6, 3, 2, 3, 16, 16))
    with pytest.raises(TypeError, match="Tensor is not a torch batch of videos."):
        transforms.BatchVideoWrapper(transform=transform)(tensor)
