import torch
import torchvision.transforms.functional as F_tv
from torchvision.transforms.functional import InterpolationMode

import gpu_torchvision.batch_transforms as transforms
import gpu_torchvision.transforms as mono_transforms


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

    torch.testing.assert_close(out, torchvision_out)


def test_batch_random_color_jitter():
    torch.manual_seed(28)

    # Checking if BatchRandomColorJitter can be printed as string
    transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 0).__repr__()

    imgs = torch.randn(8, 3, 8, 8)
    indices_to_apply = torch.tensor([[0, 1], [7, 5]])
    orders = torch.tensor([[2, 1, 3, 0], [3, 2, 1, 0]])
    brightnesss = torch.tensor([[1.2300718, 0.7117820], [0.5825809, 1.0661871]])
    contrasts = torch.tensor([[0.9351641, 0.8775994], [1.2118987, 1.4057836]])
    saturations = torch.tensor([[1.3657195, 0.6135133], [0.7385228, 1.1076477]])
    hues = torch.tensor([[0.0319767, 0.0745332], [0.0630328, 0.0541631]])

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

    torch.testing.assert_close(out, torchvision_out)

    transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 0)(imgs)
    transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, -1)(imgs)
    transforms.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 16)(imgs)


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

    torch.testing.assert_close(out, torchvision_out)


def test_batch_random_gray_scale():
    torch.manual_seed(28)

    # Checking if BatchRandomGrayScale can be printed as string
    transforms.BatchRandomGrayScale(0.5).__repr__()

    imgs = torch.randn(4, 3, 8, 8)
    indices_to_apply = torch.tensor([2, 1])

    torchvision_out = imgs.clone()
    torchvision_out[indices_to_apply] = F_tv.rgb_to_grayscale(imgs[indices_to_apply], 3)

    out = transforms.BatchRandomGrayScale(3, 0.5)(imgs)

    torch.testing.assert_close(out, torchvision_out)


def test_batch_random_horizontal_flip():
    torch.manual_seed(28)

    # Checking if BatchRandomHorizontalFlip can be printed as string
    transforms.BatchRandomHorizontalFlip(0.5).__repr__()

    imgs = torch.randn(4, 3, 8, 8)
    indices_to_apply = torch.tensor([2, 1])

    torchvision_out = imgs.clone()
    torchvision_out[indices_to_apply] = F_tv.hflip(imgs[indices_to_apply])

    out = transforms.BatchRandomHorizontalFlip(0.5)(imgs)

    torch.testing.assert_close(out, torchvision_out)


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

    torch.testing.assert_close(out, torchvision_out)

    transforms.BatchRandomResizedCrop(4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=-1)(
        imgs
    )
    transforms.BatchRandomResizedCrop(4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=0)(
        imgs
    )
    transforms.BatchRandomResizedCrop(4, (0.08, 1), (3 / 4, 4 / 3), num_rand_calls=16)(
        imgs
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

    torch.testing.assert_close(out, torchvision_out)
