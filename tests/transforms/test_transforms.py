import pytest
import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as F_tv

import torchaug.transforms as transforms


def test_normalize():
    torch.manual_seed(28)

    # test if Normalize can be printed as string
    transforms.Normalize((0.5,), (0.5,)).__repr__()

    # test the optional in-place behaviour
    tensor = torch.rand((3, 16, 16))
    torchvision_out = tv_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    out = transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    torch.testing.assert_close(out, torchvision_out)
    out_inplace = transforms.Normalize((0.5,), (0.5,), inplace=True)(tensor.clone())
    torch.testing.assert_close(out_inplace, torchvision_out)

    # test value_check
    out = transforms.Normalize((0.5,), (0.5,), inplace=False, value_check=True)(tensor)
    torch.testing.assert_close(out, torchvision_out)

    # test wrong value_check
    with pytest.raises(
        ValueError, match="std contains a zero leading to division by zero."
    ):
        transforms.Normalize((0.5,), (0.5, 0.1, 0), inplace=True, value_check=True)(
            tensor
        )


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

    # Checking instantiation with one value kernel and sigma
    transforms.RandomGaussianBlur(3, 0.1, 0.5)

    tensor = torch.rand((3, 16, 16))

    out_not_blurred = transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.0)(tensor)
    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.5)(tensor)
    out_blurred = transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 1.0)(tensor)

    torchvision_out = F_tv.gaussian_blur(tensor, (3, 3), 0.7487103)

    torch.testing.assert_close(out_blurred, torchvision_out)
    torch.testing.assert_close(out_not_blurred, tensor)

    # test kernel is not int or not sequence of 2 int.
    with pytest.raises(
        ValueError, match="Kernel size should be a tuple/list of two integers."
    ):
        transforms.RandomGaussianBlur((3, 3, 3), (0.1, 2.0), 1.0)

    # test kernel does not contain even or zero number.
    with pytest.raises(
        ValueError, match="Kernel size value should be an odd and positive number."
    ):
        transforms.RandomGaussianBlur((3, 2), (0.1, 2.0), 1.0)
    with pytest.raises(
        ValueError, match="Kernel size value should be an odd and positive number."
    ):
        transforms.RandomGaussianBlur((3, 0), (0.1, 2.0), 1.0)

    # test sigma number is inferior or equal to 0.
    with pytest.raises(
        ValueError, match="If sigma is a single number, it must be positive."
    ):
        transforms.RandomGaussianBlur((3, 3), 0, 1.0)
    with pytest.raises(
        ValueError, match="If sigma is a single number, it must be positive."
    ):
        transforms.RandomGaussianBlur((3, 3), -1, 1.0)

    # test sigma sequence is in form 0 < sigma[0] < sigma[1].
    with pytest.raises(
        ValueError,
        match=r"sigma values should be positive and of the form \(min, max\).",
    ):
        transforms.RandomGaussianBlur((3, 3), (2.0, 1.0), 1.0)
    with pytest.raises(
        ValueError,
        match=r"sigma values should be positive and of the form \(min, max\).",
    ):
        transforms.RandomGaussianBlur((3, 3), (0.0, 1.0), 1.0)

    # test sigma sequence has more than 2 elements.
    with pytest.raises(
        ValueError,
        match="sigma should be a single number or a list/tuple with length 2.",
    ):
        transforms.RandomGaussianBlur((3, 3), (0.1, 2.0, 3.0), 1.0)

    # test sigma is of wrong type.
    with pytest.raises(
        ValueError,
        match="sigma should be a single number or a list/tuple with length 2.",
    ):
        transforms.RandomGaussianBlur((3, 3), "ahah", 1.0)


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


def test_video_normalize():
    torch.manual_seed(28)

    # test if VideoNormalize can be printed as string
    transforms.VideoNormalize((0.5,), (0.5,)).__repr__()

    # test the optional in-place behaviour
    tensor = torch.rand((3, 2, 16, 16))
    torchvision_out = tv_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    out = transforms.VideoNormalize((0.5,), (0.5,), inplace=False)(tensor)
    torch.testing.assert_close(out, torchvision_out)
    out_inplace = transforms.VideoNormalize((0.5,), (0.5,), inplace=True)(
        tensor.clone()
    )
    torch.testing.assert_close(out_inplace, torchvision_out)

    # test `TCHW` video_format
    out = transforms.VideoNormalize((0.5,), (0.5,), inplace=False, video_format="TCHW")(
        tensor.permute(1, 0, 2, 3)
    )
    torch.testing.assert_close(out.permute(1, 0, 2, 3), torchvision_out)

    # test value_check
    out = transforms.VideoNormalize(
        (0.5,), (0.5,), inplace=False, value_check=True, video_format="CTHW"
    )(tensor)
    torch.testing.assert_close(out, torchvision_out)

    # test batch video tensor
    tensor = torch.rand((3, 2, 3, 16, 16))
    torchvision_out = tv_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    out = transforms.VideoNormalize((0.5,), (0.5,), inplace=False)(tensor)
    torch.testing.assert_close(out, torchvision_out)

    # test batch `TCHW` video tensor
    torchvision_out = tv_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    out = transforms.VideoNormalize((0.5,), (0.5,), inplace=False, video_format="TCHW")(
        tensor.permute(0, 2, 1, 3, 4)
    )
    torch.testing.assert_close(out.permute(0, 2, 1, 3, 4), torchvision_out)

    # test wrong video_format
    with pytest.raises(
        ValueError, match="video_format should be either 'CTHW' or 'TCHW'. Got ahah."
    ):
        transforms.VideoNormalize((0.5,), (0.5,), inplace=True, video_format="ahah")(
            tensor
        )

    # test wrong value_check
    with pytest.raises(
        ValueError, match="std contains a zero leading to division by zero."
    ):
        transforms.VideoNormalize(
            (0.5,), (0.5, 0.1, 0), inplace=True, video_format="CTHW"
        )(tensor)

    # test wrong tensor dimension
    tensor = torch.rand((6, 3, 2, 3, 16, 16))
    with pytest.raises(
        TypeError, match="Tensor is not a torch video or a batch of videos."
    ):
        transforms.VideoNormalize(
            (0.5,), (0.5, 0.1, 0), inplace=True, video_format="CTHW"
        )(tensor)


def test_video_wrapper():
    torch.manual_seed(28)

    transform = transforms.Normalize((0.5,), (0.5,), inplace=False, value_check=True)

    # test if VideoWrapper can be printed as string
    transforms.VideoWrapper(transform=transform).__repr__()

    # test CTHW format
    tensor = torch.rand((3, 2, 16, 16))
    torchvision_out = tv_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    out = transforms.VideoWrapper(transform=transform)(tensor)
    torch.testing.assert_close(out, torchvision_out)

    # test TCHW format
    torchvision_out = tv_transforms.Normalize((0.5,), (0.5,), inplace=False)(tensor)
    out = transforms.VideoWrapper(transform=transform, video_format="TCHW")(
        tensor.permute(1, 0, 2, 3)
    )
    torch.testing.assert_close(out.permute(1, 0, 2, 3), torchvision_out)

    # test wrong video_format
    with pytest.raises(
        ValueError, match="video_format should be either 'CTHW' or 'TCHW'. Got ahah."
    ):
        transforms.VideoWrapper(transform=transform, video_format="ahah")(tensor)

    # test wrong tensor dimension
    tensor = torch.rand((6, 3, 2, 3, 16, 16))
    with pytest.raises(TypeError, match="Tensor is not a torch video."):
        transforms.VideoWrapper(transform=transform)(tensor)
