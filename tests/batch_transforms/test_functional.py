import pytest
import torch
import torchvision.transforms.functional as F_tv

import torchaug.batch_transforms.functional as F


def test_batch_adjust_brightness():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x = torch.tensor(x_data, dtype=torch.uint8).reshape(x_shape).permute((2, 1, 0))

    y_0 = F_tv.adjust_brightness(x, 1)
    y_1 = F_tv.adjust_brightness(x, 0.5)
    y_2 = F_tv.adjust_brightness(x, 2)
    stacked_x = torch.stack((x, x, x))

    y_batch = F.batch_adjust_brightness(stacked_x, torch.tensor((1, 0.5, 2)), True)

    # test B tensor factor
    torch.testing.assert_close(y_batch, torch.stack((y_0, y_1, y_2)))

    # test 1 tensor factor
    torch.testing.assert_close(
        F.batch_adjust_brightness(stacked_x, torch.tensor(0.5)),
        torch.stack([F_tv.adjust_brightness(x, 0.5)] * 3),
    )
    torch.testing.assert_close(
        F.batch_adjust_brightness(stacked_x, torch.tensor([0.5])),
        torch.stack([F_tv.adjust_brightness(x, 0.5)] * 3),
    )

    # test float factor
    torch.testing.assert_close(
        F.batch_adjust_brightness(stacked_x, 0.5),
        torch.stack([F_tv.adjust_brightness(x, 0.5)] * 3),
    )

    # test negative factor
    with pytest.raises(
        ValueError,
        match="brightness_factor is not non-negative.",
    ):
        F.batch_adjust_brightness(stacked_x, -0.5)
    with pytest.raises(
        ValueError,
        match="brightness_factor is not non-negative.",
    ):
        F.batch_adjust_brightness(stacked_x, torch.tensor(-0.5), True)
    with pytest.raises(
        ValueError,
        match="brightness_factor is not non-negative.",
    ):
        F.batch_adjust_brightness(stacked_x, torch.tensor([1.0, -0.5, 2.0]), True)

    # test wrong factor size
    with pytest.raises(
        ValueError,
        match="brightness_factor tensor should contain 1 or B elements.",
    ):
        F.batch_adjust_brightness(stacked_x, torch.tensor([1.0, 2.0]), True)

    # test wrong factor type
    with pytest.raises(
        TypeError,
        match="brightness_factor should be a float or Tensor.",
    ):
        F.batch_adjust_brightness(stacked_x, "ahah", True)

    # Test input is not tensor.
    with pytest.raises(
        TypeError, match="Object should be a tensor. Got <class 'list'>."
    ):
        F.batch_adjust_brightness([2, 0, 3], 0.5)

    # Test input without batch dimension.
    with pytest.raises(TypeError, match="Tensor is not a torch batch of images."):
        F.batch_adjust_brightness(torch.rand((4, 12, 12)), 0.5)

    # Test input wrong channel size.
    with pytest.raises(
        TypeError,
        match=r"Input image tensor permitted channel values are \[1, 3\], but found 4.",
    ):
        F.batch_adjust_brightness(torch.rand((4, 4, 12, 12)), 0.5)


def test_batch_adjust_contrast():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x = torch.tensor(x_data, dtype=torch.uint8).reshape(x_shape).permute((2, 1, 0))

    y_0 = F_tv.adjust_contrast(x, 1)
    y_1 = F_tv.adjust_contrast(x, 0.5)
    y_2 = F_tv.adjust_contrast(x, 2)
    stacked_x = torch.stack((x, x, x))

    y_batch = F.batch_adjust_contrast(stacked_x, torch.tensor((1, 0.5, 2)))

    # test B tensor factor
    torch.testing.assert_close(y_batch, torch.stack((y_0, y_1, y_2)))

    # test B tensor factor one channel
    one_c_x = (
        torch.tensor(x_data, dtype=torch.uint8).reshape((4, 3, 1)).permute((2, 1, 0))
    )
    y_0 = F_tv.adjust_contrast(one_c_x, 1)
    y_1 = F_tv.adjust_contrast(one_c_x, 0.5)
    y_2 = F_tv.adjust_contrast(one_c_x, 2)
    stacked_one_c_x = torch.stack((one_c_x, one_c_x, one_c_x))
    y_one_c_batch = F.batch_adjust_contrast(stacked_one_c_x, torch.tensor((1, 0.5, 2)))
    torch.testing.assert_close(y_one_c_batch, torch.stack([y_0, y_1, y_2]))

    # test float factor
    torch.testing.assert_close(
        F.batch_adjust_contrast(stacked_x, 0.5),
        torch.stack([F_tv.adjust_contrast(x, 0.5)] * 3),
    )

    # test negative factor
    with pytest.raises(
        ValueError,
        match="contrast_factor is not non-negative.",
    ):
        F.batch_adjust_contrast(stacked_x, -0.5)
    with pytest.raises(
        ValueError,
        match="contrast_factor is not non-negative.",
    ):
        F.batch_adjust_contrast(stacked_x, torch.tensor(-0.5), True)
    with pytest.raises(
        ValueError,
        match="contrast_factor is not non-negative.",
    ):
        F.batch_adjust_contrast(stacked_x, torch.tensor([1.0, -0.5, 2.0]), True)

    # test wrong factor size
    with pytest.raises(
        ValueError,
        match="contrast_factor tensor should contain 1 or B elements.",
    ):
        F.batch_adjust_contrast(stacked_x, torch.tensor([1.0, 2.0]), True)

    # test wrong factor type
    with pytest.raises(
        TypeError,
        match="contrast_factor should be a float or Tensor.",
    ):
        F.batch_adjust_contrast(stacked_x, "ahah", True)

    # Test input is not tensor.
    with pytest.raises(
        TypeError, match="Object should be a tensor. Got <class 'list'>."
    ):
        F.batch_adjust_contrast([2, 0, 3], 0.5)

    # Test input is not batched tensor.
    with pytest.raises(
        TypeError, match="Object should be a tensor. Got <class 'list'>."
    ):
        F.batch_adjust_contrast([2, 0, 3], 0.5)

    # Test input without batch dimension.
    with pytest.raises(TypeError, match="Tensor is not a torch batch of images."):
        F.batch_adjust_contrast(torch.rand((4, 12, 12)), 0.5)

    # Test input wrong channel size.
    with pytest.raises(
        TypeError,
        match=r"Input image tensor permitted channel values are \[1, 3\], but found 4.",
    ):
        F.batch_adjust_contrast(torch.rand((4, 4, 12, 12)), 0.5)


def test_batch_adjust_hue():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x = torch.tensor(x_data, dtype=torch.uint8).reshape(x_shape).permute((2, 1, 0))

    y_0 = F_tv.adjust_hue(x, 0)
    y_1 = F_tv.adjust_hue(x, 0.25)
    y_2 = F_tv.adjust_hue(x, -0.25)
    stacked_x = torch.stack((x, x, x))

    y_batch = F.batch_adjust_hue(stacked_x, torch.tensor((0, 0.25, -0.25)), True)

    # test B tensor factor
    torch.testing.assert_close(y_batch, torch.stack((y_0, y_1, y_2)))

    # test B tensor factor one channel
    one_c_x = (
        torch.tensor(x_data, dtype=torch.uint8).reshape((4, 3, 1)).permute((2, 1, 0))
    )
    stacked_one_c_x = torch.stack((one_c_x, one_c_x, one_c_x))
    y_one_c_batch = F.batch_adjust_hue(stacked_one_c_x, torch.tensor((0, 0.25, -0.25)))
    torch.testing.assert_close(y_one_c_batch, stacked_one_c_x)

    # test float factor
    torch.testing.assert_close(
        F.batch_adjust_hue(stacked_x, 0.5), torch.stack([F_tv.adjust_hue(x, 0.5)] * 3)
    )

    # test wrong value factor
    with pytest.raises(
        ValueError,
        match="hue_factor is not between -0.5 and 0.5.",
    ):
        F.batch_adjust_hue(stacked_x, -0.75)
    with pytest.raises(
        ValueError,
        match="hue_factor is not between -0.5 and 0.5.",
    ):
        F.batch_adjust_hue(stacked_x, 0.75)
    with pytest.raises(
        ValueError,
        match="hue_factor is not between -0.5 and 0.5.",
    ):
        F.batch_adjust_hue(stacked_x, torch.tensor(-0.75), True)
    with pytest.raises(
        ValueError,
        match="hue_factor is not between -0.5 and 0.5.",
    ):
        F.batch_adjust_hue(stacked_x, torch.tensor(0.75), True)
    with pytest.raises(
        ValueError,
        match="hue_factor is not between -0.5 and 0.5.",
    ):
        F.batch_adjust_hue(stacked_x, torch.tensor([1.0, -0.5, 2.0]), True)

    # test wrong factor size
    with pytest.raises(
        ValueError,
        match="hue_factor tensor should contain 1 or B elements.",
    ):
        F.batch_adjust_hue(stacked_x, torch.tensor([0.25, -0.25]), True)

    # test wrong factor type
    with pytest.raises(
        TypeError,
        match="hue_factor should be a float or Tensor.",
    ):
        F.batch_adjust_hue(stacked_x, "ahah", True)

    # Test input is not tensor.
    with pytest.raises(
        TypeError, match="Object should be a tensor. Got <class 'list'>."
    ):
        F.batch_adjust_hue([2, 0, 3], 0.5)

    # Test input without batch dimension.
    with pytest.raises(TypeError, match="Tensor is not a torch batch of images."):
        F.batch_adjust_hue(torch.rand((4, 12, 12)), 0.5)

    # Test input wrong channel size.
    with pytest.raises(
        TypeError,
        match=r"Input image tensor permitted channel values are \[1, 3\], but found 4.",
    ):
        F.batch_adjust_hue(torch.rand((4, 4, 12, 12)), 0.5)


def test_batch_adjust_saturation():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x = torch.tensor(x_data, dtype=torch.uint8).reshape(x_shape).permute((2, 1, 0))

    y_0 = F_tv.adjust_saturation(x, 1)
    y_1 = F_tv.adjust_saturation(x, 0.5)
    y_2 = F_tv.adjust_saturation(x, 2)
    stacked_x = torch.stack((x, x, x))

    y_batch = F.batch_adjust_saturation(stacked_x, torch.tensor((1, 0.5, 2)))

    # test B tensor factor
    torch.testing.assert_close(y_batch, torch.stack((y_0, y_1, y_2)))

    # test B tensor factor one channel
    one_c_x = (
        torch.tensor(x_data, dtype=torch.uint8).reshape((4, 3, 1)).permute((2, 1, 0))
    )
    stacked_one_c_x = torch.stack((one_c_x, one_c_x, one_c_x))
    y_one_c_batch = F.batch_adjust_saturation(
        stacked_one_c_x, torch.tensor((1, 0.5, 2))
    )
    torch.testing.assert_close(y_one_c_batch, stacked_one_c_x)

    # test float factor
    torch.testing.assert_close(
        F.batch_adjust_saturation(stacked_x, 0.5),
        torch.stack([F_tv.adjust_saturation(x, 0.5)] * 3),
    )

    # test negative factor
    with pytest.raises(
        ValueError,
        match="saturation_factor is not non-negative.",
    ):
        F.batch_adjust_saturation(stacked_x, -0.5)
    with pytest.raises(
        ValueError,
        match="saturation_factor is not non-negative.",
    ):
        F.batch_adjust_saturation(stacked_x, torch.tensor(-0.5), True)
    with pytest.raises(
        ValueError,
        match="saturation_factor is not non-negative.",
    ):
        F.batch_adjust_saturation(stacked_x, torch.tensor([1.0, -0.5, 2.0]), True)

    # test wrong factor size
    with pytest.raises(
        ValueError,
        match="saturation_factor tensor should contain 1 or B elements.",
    ):
        F.batch_adjust_saturation(stacked_x, torch.tensor([1.0, 2.0]), True)

    # test wrong factor type
    with pytest.raises(
        TypeError,
        match="saturation_factor should be a float or Tensor.",
    ):
        F.batch_adjust_saturation(stacked_x, "ahah", True)

    # Test input is not tensor.
    with pytest.raises(
        TypeError, match="Object should be a tensor. Got <class 'list'>."
    ):
        F.batch_adjust_saturation([2, 0, 3], 0.5)

    # Test input without batch dimension.
    with pytest.raises(TypeError, match="Tensor is not a torch batch of images."):
        F.batch_adjust_saturation(torch.rand((4, 12, 12)), 0.5)

    # Test input wrong channel size.
    with pytest.raises(
        TypeError,
        match=r"Input image tensor permitted channel values are \[1, 3\], but found 4.",
    ):
        F.batch_adjust_saturation(torch.rand((4, 4, 12, 12)), 0.5)


def test_batch_gaussian_blur():
    torch.manual_seed(28)

    x_shape = [3, 4, 4]
    x = torch.randint(0, 255, x_shape)

    # test (B,2) tensor factor
    y_0 = F_tv.gaussian_blur(x, [3, 3], 1)
    y_1 = F_tv.gaussian_blur(x, [3, 3], [0.5, 0.75])
    y_2 = F_tv.gaussian_blur(x, [3, 3], 2)
    stacked_x = torch.stack((x, x, x))
    y_batch = F.batch_gaussian_blur(
        stacked_x, [3, 3], torch.tensor([[1, 1], [0.5, 0.75], [2, 2]])
    )
    torch.testing.assert_close(y_batch, torch.stack((y_0, y_1, y_2)))

    # test (B,) tensor sigma
    y_0 = F_tv.gaussian_blur(x, [3, 3], 1.0)
    y_1 = F_tv.gaussian_blur(x, [3, 3], 0.5)
    y_2 = F_tv.gaussian_blur(x, [3, 3], 2)
    stacked_x = torch.stack((x, x, x))
    y_batch = F.batch_gaussian_blur(stacked_x, [3, 3], torch.tensor([1, 0.5, 2]))
    torch.testing.assert_close(y_batch, torch.stack((y_0, y_1, y_2)))

    # test (1,) tensor sigma
    y_0 = F_tv.gaussian_blur(x, [3, 3], 1.0)
    stacked_x = torch.stack((x, x, x))
    y_batch = F.batch_gaussian_blur(stacked_x, [3, 3], torch.tensor(1))
    torch.testing.assert_close(y_batch, torch.stack((y_0, y_0, y_0)))

    # test 1 float sigma
    y_0 = F_tv.gaussian_blur(x, [3, 3], 1.0)
    stacked_x = torch.stack((x, x, x))
    y_batch = F.batch_gaussian_blur(stacked_x, [3, 3], 1)
    torch.testing.assert_close(y_batch, torch.stack((y_0, y_0, y_0)))

    # test 1 element sequence float sigma
    y_0 = F_tv.gaussian_blur(x, [3, 3], 1.0)
    stacked_x = torch.stack((x, x, x))
    y_batch = F.batch_gaussian_blur(stacked_x, [3, 3], [1])
    torch.testing.assert_close(y_batch, torch.stack((y_0, y_0, y_0)))

    # test 2 element sequence float sigma
    y_0 = F_tv.gaussian_blur(x, [3, 3], 1.0)
    stacked_x = torch.stack((x, x, x))
    y_batch = F.batch_gaussian_blur(stacked_x, [3, 3], [1, 1])
    torch.testing.assert_close(y_batch, torch.stack((y_0, y_0, y_0)))

    # test None sigma
    y_0 = F_tv.gaussian_blur(x, [3, 3], None)
    stacked_x = torch.stack((x, x, x))
    y_batch = F.batch_gaussian_blur(stacked_x, [3, 3], None)
    torch.testing.assert_close(y_batch, torch.stack((y_0, y_0, y_0)))

    # test 1 int sequence kernel
    y_0 = F_tv.gaussian_blur(x, [3, 3], 1.0)
    y_1 = F_tv.gaussian_blur(x, [3, 3], 1)
    y_2 = F_tv.gaussian_blur(x, [3, 3], 1)
    stacked_x = torch.stack((x, x, x))
    y_batch = F.batch_gaussian_blur(stacked_x, 3, 1)
    torch.testing.assert_close(y_batch, torch.stack((y_0, y_1, y_2)))

    # test kernel is not int or not sequence of 2 int.
    with pytest.raises(
        TypeError,
        match="kernel_size should be int or a sequence of integers. Got <class 'float'>.",
    ):
        F.batch_gaussian_blur(stacked_x, 3.0, [2, 2])
    with pytest.raises(
        ValueError, match="If kernel_size is a sequence its length should be 2. Got 3."
    ):
        F.batch_gaussian_blur(stacked_x, [3, 3, 3], [2, 2])

    # test kernel does not contain even or zero number.
    with pytest.raises(
        ValueError,
        match=r"kernel_size should have odd and positive integers. Got \[3, 2\].",
    ):
        F.batch_gaussian_blur(stacked_x, [3, 2], [2, 2])
    with pytest.raises(
        ValueError,
        match=r"kernel_size should have odd and positive integers. Got \[3, 0\].",
    ):
        F.batch_gaussian_blur(stacked_x, [3, 0], [2, 2])

    # test sigma int is inferior or equal to 0.
    with pytest.raises(ValueError, match="sigma should have positive values."):
        F.batch_gaussian_blur(stacked_x, [3, 3], 0)
    with pytest.raises(ValueError, match="sigma should have positive values."):
        F.batch_gaussian_blur(stacked_x, [3, 3], -1)

    # test sigma tensor is inferior or equal to 0.
    with pytest.raises(ValueError, match="sigma should have positive values."):
        F.batch_gaussian_blur(stacked_x, [3, 3], torch.tensor(0), True)
    with pytest.raises(ValueError, match="sigma should have positive values."):
        F.batch_gaussian_blur(stacked_x, [3, 3], torch.tensor([0.5, -1, 0.25]), True)
    with pytest.raises(ValueError, match="sigma should have positive values."):
        F.batch_gaussian_blur(
            stacked_x, [3, 3], torch.tensor([[0.5, 0.3], [-1, 1.0], [0.25, 0.25]]), True
        )

    # test sigma sequence int has more than 2 elements.
    with pytest.raises(
        ValueError, match="If sigma is a sequence, its length should be 2. Got 3."
    ):
        F.batch_gaussian_blur(stacked_x, [3, 3], [3, 3, 3])

    # test sigma tensor has more than 2 dimensions.
    with pytest.raises(
        ValueError,
        match="If sigma is a tensor, its dimension should be 0, 1 or 2. Got 3.",
    ):
        F.batch_gaussian_blur(stacked_x, [3, 3], torch.tensor([[[3.0]]]))

    # test sigma tensor does not have 1 or B elements.
    with pytest.raises(
        ValueError,
        match="If sigma is a tensor, it should contain one or batch size elements. Got 4.",
    ):
        F.batch_gaussian_blur(stacked_x, [3, 3], torch.tensor([3.0, 3.0, 3.0, 3.0]))

    # test sigma tensor has more than 2 elements in second dimension.
    with pytest.raises(
        ValueError,
        match="If sigma is a tensor, it should have a feature dim of size one or two. Got 3.",
    ):
        F.batch_gaussian_blur(stacked_x, [3, 3], torch.tensor([[3.0, 3.0, 3.0]]))

    # test sigma sequence float has more than 2 elements.
    with pytest.raises(
        ValueError, match="If sigma is a sequence, its length should be 2. Got 3."
    ):
        F.batch_gaussian_blur(stacked_x, [3, 3], [3.0, 3.0, 3.0])

    # test sigma is of wrong type.
    with pytest.raises(
        TypeError,
        match="sigma should be either int, float or sequence of floats or int or tensor. Got <class 'str'>.",
    ):
        F.batch_gaussian_blur(stacked_x, [3, 3], "ahah")

    # Test input is not tensor.
    with pytest.raises(
        TypeError, match="Object should be a tensor. Got <class 'list'>."
    ):
        F.batch_gaussian_blur([2, 0, 3], [3, 3], [2, 2])

    # Test input without batch dimension.
    with pytest.raises(TypeError, match="Tensor is not a torch batch of images."):
        F.batch_gaussian_blur(torch.rand((4, 12, 12)), [3, 3], [2, 2])
