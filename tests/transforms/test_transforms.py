import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as F_tv

import torchaug.transforms as transforms


def test_normalize():
    torch.manual_seed(28)

    # Checking if Normalize can be printed as string
    transforms.Normalize((0.5,), (0.5,)).__repr__()
    transforms.Normalize(
        torch.tensor(
            0.5,
        ),
        torch.tensor(
            0.5,
        ),
    ).__repr__()

    # Checking the optional in-place behaviour
    tensor = torch.rand((1, 16, 16))
    torchvision_out = tv_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    out = transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    torch.testing.assert_close(out, torchvision_out)
    out_inplace = transforms.Normalize((0.5,), (0.5,), inplace=True)(tensor)
    torch.testing.assert_close(out_inplace, torchvision_out)


def test_random_apply():
    torch.manual_seed(28)

    # Checking if RandomApply can be printed as string
    transforms.RandomApply(
        [
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomGaussianBlur((3, 3), (0.1, 2.0)),
        ]
    ).__repr__()
    transforms.RandomApply(transforms.Normalize((0.5,), (0.5,))).__repr__()

    tensor = torch.rand((1, 16, 16))
    torchvision_out = tv_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)

    out_norm = transforms.RandomApply(
        [
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomGaussianBlur((3, 3), (0.1, 2.0)),
        ],
        p=1,
    )(tensor)
    transforms.RandomApply(
        [
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
        ]
    )(tensor)
    out_not_norm = transforms.RandomApply(
        [
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0),
        ],
        p=0.0,
    )(tensor)

    torch.testing.assert_close(out_not_norm, tensor)
    torch.testing.assert_close(out_norm, torchvision_out)


def test_random_color_jitter():
    torch.manual_seed(28)

    # Checking if RandomColorJitter can be printed as string
    transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.5).__repr__()

    tensor = torch.rand((3, 16, 16))

    out_jittered = transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 1.0)(tensor)
    transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.5)(tensor)
    out_not_jittered = transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.0)(tensor)

    brightness, contrast, saturation, hue = (
        torch.tensor(0.8870313763618469),
        torch.tensor(0.8208379745483398),
        torch.tensor(1.0451856851577759),
        torch.tensor(0.05531376972794533),
    )

    torchvision_out = F_tv.adjust_hue(
        F_tv.adjust_contrast(
            F_tv.adjust_saturation(
                F_tv.adjust_brightness(tensor, brightness), saturation
            ),
            contrast,
        ),
        hue,
    )

    torch.testing.assert_close(out_jittered, torchvision_out)
    torch.testing.assert_close(out_not_jittered, tensor)


def test_random_gaussian_blur():
    torch.manual_seed(28)

    # Checking if RandomGaussianBlur can be printed as string
    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.5).__repr__()

    tensor = torch.rand((3, 16, 16))

    out_not_blurred = transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.0)(tensor)
    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.5)(tensor)
    out_blurred = transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 1.0)(tensor)

    torchvision_out = F_tv.gaussian_blur(tensor, (3, 3), 0.7487103)

    torch.testing.assert_close(out_blurred, torchvision_out)
    torch.testing.assert_close(out_not_blurred, tensor)


def test_random_solarize():
    torch.manual_seed(28)

    # Checking if RandomSolarize can be printed as string
    transforms.RandomSolarize(0.5, 0.5).__repr__()

    tensor = torch.rand((3, 16, 16))

    out_not_solarized = transforms.RandomSolarize(0.5, 0.0)(tensor)
    transforms.RandomSolarize(0.5, 0.5)(tensor)
    out_solarized = transforms.RandomSolarize(0.5, 1.0)(tensor)

    torchvision_out = F_tv.solarize(tensor, 0.5)

    torch.testing.assert_close(out_solarized, torchvision_out)
    torch.testing.assert_close(out_not_solarized, tensor)
