import re

import pytest
import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as F_tv
from torchvision.transforms.functional import convert_image_dtype

import torchaug.transforms as transforms


def test_div_255():
    torch.manual_seed(28)

    # test if Div255 can be printed as string
    assert isinstance(transforms.Div255(False).__repr__(), str)
    tensor = torch.rand((3, 16, 16))
    expected_out = tensor / 255
    torch.testing.assert_close(transforms.Div255(False)(tensor), expected_out)

    transforms.Div255(True)(tensor)
    torch.testing.assert_close(tensor, expected_out)


def test_mixup():
    torch.manual_seed(28)

    # Checking if MixUp can be printed as string
    assert isinstance(
        transforms.MixUp(0.5, True).__repr__(),
        str,
    )

    imgs = torch.randn(4, 3, 8, 8)
    labels = torch.randint(0, 5, (4, 1)).to(torch.float)
    inpt_imgs = imgs.clone()
    inpt_labels = labels.clone()
    out_imgs, out_labels, out_lam = transforms.MixUp(0.5, inplace=True)(
        inpt_imgs, inpt_labels
    )

    expected_lam = 0.8844035863876343
    expected_out_labels = expected_lam * labels + (1 - expected_lam) * labels.roll(1, 0)
    expected_out_imgs = expected_lam * imgs + (1 - expected_lam) * imgs.roll(1, 0)

    torch.testing.assert_close(torch.tensor(out_lam), torch.tensor(expected_lam))
    torch.testing.assert_close(out_labels, expected_out_labels)
    torch.testing.assert_close(out_imgs, expected_out_imgs)
    torch.testing.assert_close(out_imgs, inpt_imgs)
    torch.testing.assert_close(out_labels, inpt_labels)

    out_imgs, out_labels, out_lam = transforms.MixUp(0.5, inplace=True)(inpt_imgs, None)
    assert out_labels is None

    out_imgs, out_labels, out_lam = transforms.MixUp(0.5, inplace=False)(imgs, None)
    assert out_labels is None

    expected_lam = 0.007286167237907648
    torch.testing.assert_close(torch.tensor(out_lam), torch.tensor(expected_lam))
    expected_out_imgs = expected_lam * imgs + (1 - expected_lam) * imgs.roll(1, 0)
    torch.testing.assert_close(out_imgs, expected_out_imgs)


def test_normalize():
    torch.manual_seed(28)

    # test if Normalize can be printed as string
    assert isinstance(transforms.Normalize((0.5,), (0.5,)).__repr__(), str)

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

    # test keep_dtype
    int_tensor = torch.randint(1, 256, (3, 10, 10), dtype=torch.uint8)
    out = transforms.Normalize(
        (0.5,),
        (0.5,),
        inplace=False,
        cast_dtype=torch.float32,
    )(int_tensor)
    torch.testing.assert_close(
        out,
        F_tv.normalize(
            convert_image_dtype(int_tensor, dtype=torch.float32),
            (0.5,),
            (0.5,),
            inplace=False,
        ),
    )
    with pytest.raises(
        TypeError,
        match="Input tensor should be a float tensor or cast_dtype set to a float dtype. Got torch.uint8 and None.",
    ):
        transforms.Normalize((0.5,), (0.5,), inplace=False, cast_dtype=None)(int_tensor)
    with pytest.raises(
        ValueError,
        match="cast_dtype should be a float dtype. Got torch.int32.",
    ):
        transforms.Normalize((0.5,), (0.5,), inplace=False, cast_dtype=torch.int32)(
            int_tensor
        )

    # test wrong value_check
    with pytest.raises(
        ValueError, match="std contains a zero leading to division by zero."
    ):
        transforms.Normalize((0.5,), (0.5, 0.1, 0), inplace=True, value_check=True)(
            tensor
        )


def test_mul_255():
    torch.manual_seed(28)

    # test if Mul255 can be printed as string
    assert isinstance(transforms.Mul255(False).__repr__(), str)
    tensor = torch.rand((3, 16, 16))
    expected_out = tensor * 255
    torch.testing.assert_close(transforms.Mul255(False)(tensor), expected_out)

    transforms.Mul255(True)(tensor)
    torch.testing.assert_close(tensor, expected_out)


def test_random_apply():
    torch.manual_seed(28)

    # Checking if RandomApply can be printed as string
    assert isinstance(
        transforms.RandomApply(
            [
                transforms.Normalize((0.5,), (0.5,)),
                transforms.RandomGaussianBlur((3, 3), (0.1, 2.0)),
            ]
        ).__repr__(),
        str,
    )
    assert isinstance(
        transforms.RandomApply(transforms.Normalize((0.5,), (0.5,))).__repr__(), str
    )

    # test p < 0 and p > 1
    with pytest.raises(
        ValueError,
        match=re.escape(
            "p should be superior to 0 (included) and inferior to 1 (included). Got -0.1."
        ),
    ):
        transforms.RandomApply(
            [transforms.Normalize([225, 225, 225], [0.25, 0.25, 0.25])],
            -0.1,
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "p should be superior to 0 (included) and inferior to 1 (included). Got 1.1."
        ),
    ):
        transforms.RandomApply(
            [transforms.Normalize([225, 225, 225], [0.25, 0.25, 0.25])],
            1.1,
        )

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
    assert isinstance(
        transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.5).__repr__(), str
    )

    tensor = torch.rand((3, 16, 16))

    out_jittered = transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 1.0)(tensor)
    transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.5)(tensor)
    out_not_jittered = transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.0)(tensor)

    brightness, contrast, saturation, hue = (
        torch.tensor(0.8414264917373657),
        torch.tensor(0.9322187900543213),
        torch.tensor(0.9402793049812317),
        torch.tensor(0.1355568766593933),
    )

    torchvision_out = F_tv.adjust_brightness(
        F_tv.adjust_hue(
            F_tv.adjust_contrast(F_tv.adjust_saturation(tensor, saturation), contrast),
            hue,
        ),
        brightness,
    )

    torch.testing.assert_close(out_jittered, torchvision_out)
    torch.testing.assert_close(out_not_jittered, tensor)


def test_random_gaussian_blur():
    torch.manual_seed(28)

    # Checking if RandomGaussianBlur can be printed as string
    assert isinstance(
        transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.5).__repr__(), str
    )

    # Checking instantiation with one value kernel and sigma
    transforms.RandomGaussianBlur(3, 0.1, 0.5)

    tensor = torch.rand((3, 16, 16))

    out_not_blurred = transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.0)(tensor)
    transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 0.5)(tensor)
    out_blurred = transforms.RandomGaussianBlur((3, 3), (0.1, 2.0), 1.0)(tensor)

    torchvision_out = F_tv.gaussian_blur(tensor, (3, 3), 0.762560)

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
    assert isinstance(transforms.RandomSolarize(0.5, 0.5).__repr__(), str)

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
    assert isinstance(
        transforms.VideoNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).__repr__(), str
    )

    # test the optional in-place behaviour
    tensor = torch.rand((3, 8, 16, 16))
    torchvision_out = tv_transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False
    )(tensor.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
    out = transforms.VideoNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)(
        tensor
    )
    torch.testing.assert_close(out, torchvision_out)
    out_inplace = transforms.VideoNormalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
    )(tensor.clone())
    torch.testing.assert_close(out_inplace, torchvision_out)

    # test `TCHW` video_format
    out = transforms.VideoNormalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False, video_format="TCHW"
    )(tensor.permute(1, 0, 2, 3))
    torch.testing.assert_close(out.permute(1, 0, 2, 3), torchvision_out)

    # test value_check
    out = transforms.VideoNormalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        inplace=False,
        value_check=True,
        video_format="CTHW",
    )(tensor)
    torch.testing.assert_close(out, torchvision_out)

    # test keep_dtype
    int_tensor = torch.randint(1, 256, (8, 3, 10, 10), dtype=torch.uint8)
    out = transforms.VideoNormalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        inplace=False,
        cast_dtype=torch.float32,
        video_format="TCHW",
    )(int_tensor)
    torch.testing.assert_close(
        out,
        F_tv.normalize(
            convert_image_dtype(int_tensor, dtype=torch.float32),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            inplace=False,
        ),
    )
    with pytest.raises(
        TypeError,
        match="Input tensor should be a float tensor or cast_dtype set to a float dtype. Got torch.uint8 and None.",
    ):
        transforms.VideoNormalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            inplace=False,
            cast_dtype=None,
            video_format="TCHW",
        )(int_tensor)
    with pytest.raises(
        ValueError,
        match="cast_dtype should be a float dtype. Got torch.int32.",
    ):
        transforms.VideoNormalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            inplace=False,
            cast_dtype=torch.int32,
            video_format="TCHW",
        )(int_tensor)

    # test batch video tensor
    tensor = torch.rand((3, 10, 3, 16, 16))
    torchvision_out = tv_transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False
    )(tensor)
    out = transforms.VideoNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)(
        tensor.permute(0, 2, 1, 3, 4)
    ).permute(0, 2, 1, 3, 4)
    torch.testing.assert_close(out, torchvision_out)

    # test batch `TCHW` video tensor
    torchvision_out = tv_transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False
    )(tensor)
    out = transforms.VideoNormalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False, video_format="TCHW"
    )(tensor)
    torch.testing.assert_close(out, torchvision_out)

    # test wrong video_format
    with pytest.raises(
        ValueError, match="video_format should be either 'CTHW' or 'TCHW'. Got ahah."
    ):
        transforms.VideoNormalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True, video_format="ahah"
        )(tensor)

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
    assert isinstance(transforms.VideoWrapper(transform=transform).__repr__(), str)

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
