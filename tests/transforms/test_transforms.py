import torch
import torchvision.transforms as tv_transforms

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
    tensor_tv = tv_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    tensor_norm = transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    torch.testing.assert_close(tensor_norm, tensor_tv)
    tensor_inplace = transforms.Normalize((0.5,), (0.5,), inplace=True)(tensor)
    torch.testing.assert_close(tensor_inplace, tensor_tv)


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

    transforms.RandomApply(
        [
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomGaussianBlur((3, 3), (0.1, 2.0)),
        ]
    )(tensor)
    transforms.RandomApply(
        [
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), p=1),
        ]
    )(tensor)
    transforms.RandomApply(
        [
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0),
        ]
    )(tensor)


def test_random_color_jitter():
    torch.manual_seed(28)

    # Checking if RandomColorJitter can be printed as string
    transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.5).__repr__()

    tensor = torch.rand((3, 16, 16))

    transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.5)(tensor)
    transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 1.0)(tensor)
    transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.0)(tensor)


def test_random_gaussian_blur():
    torch.manual_seed(28)

    # Checking if RandomGaussianBlur can be printed as string
    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.5).__repr__()

    tensor = torch.rand((3, 16, 16))

    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.0)(tensor)
    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.5)(tensor)
    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 1.0)(tensor)


def test_random_solarize():
    torch.manual_seed(28)

    # Checking if RandomSolarize can be printed as string
    transforms.RandomSolarize(0.5, 0.5).__repr__()

    tensor = torch.rand((3, 16, 16))

    transforms.RandomSolarize(0.5, 0.0)(tensor)
    transforms.RandomSolarize(0.5, 0.5)(tensor)
    transforms.RandomSolarize(0.5, 1.0)(tensor)
