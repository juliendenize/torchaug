import torch
import torchvision.transforms.functional as F_tv

import torchaug.batch_transforms.functional as F


def test_batch_adjust_brightness():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x = torch.tensor(x_data, dtype=torch.uint8).reshape(x_shape).permute((2, 1, 0))

    # test 0
    y_0 = F_tv.adjust_brightness(x, 1)

    # test 1
    y_1 = F_tv.adjust_brightness(x, 0.5)

    # test 2
    y_2 = F_tv.adjust_brightness(x, 2)

    y_batch = F.batch_adjust_brightness(
        torch.stack((x, x, x)), torch.tensor((1, 0.5, 2))
    )

    torch.testing.assert_close(y_batch, torch.stack((y_0, y_1, y_2)))


def test_batch_adjust_contrast():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x = torch.tensor(x_data, dtype=torch.uint8).reshape(x_shape).permute((2, 1, 0))

    # test 0
    y_0 = F_tv.adjust_contrast(x, 1)

    # test 1
    y_1 = F_tv.adjust_contrast(x, 0.5)

    # test 2
    y_2 = F_tv.adjust_contrast(x, 2)

    y_batch = F.batch_adjust_contrast(torch.stack((x, x, x)), torch.tensor((1, 0.5, 2)))

    torch.testing.assert_close(y_batch, torch.stack((y_0, y_1, y_2)))


def test_batch_adjust_hue():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x = torch.tensor(x_data, dtype=torch.uint8).reshape(x_shape).permute((2, 1, 0))

    # TODO: No ValueError for adjust_hue.
    # with pytest.raises(ValueError):
    #     F.batch_adjust_hue(x, torch.tensor((-0.7)))
    #     F.batch_adjust_hue(x, torch.tensor((1)))

    # test 0
    y_0 = F_tv.adjust_hue(x, 0)

    # test 1
    y_1 = F_tv.adjust_hue(x, 0.25)

    # test 2
    y_2 = F_tv.adjust_hue(x, -0.25)

    y_batch = F.batch_adjust_hue(torch.stack((x, x, x)), torch.tensor((0, 0.25, -0.25)))

    torch.testing.assert_close(y_batch, torch.stack((y_0, y_1, y_2)))


def test_batch_adjust_saturation():
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x = torch.tensor(x_data, dtype=torch.uint8).reshape(x_shape).permute((2, 1, 0))

    # test 0
    y_0 = F_tv.adjust_saturation(x, 1)

    # test 1
    y_1 = F_tv.adjust_saturation(x, 0.5)

    # test 2
    y_2 = F_tv.adjust_saturation(x, 2)

    y_batch = F.batch_adjust_saturation(
        torch.stack((x, x, x)), torch.tensor((1, 0.5, 2))
    )

    torch.testing.assert_close(y_batch, torch.stack((y_0, y_1, y_2)))


def test_batch_gaussian_blur():
    torch.manual_seed(28)

    x_shape = [3, 4, 4]
    x = torch.randint(0, 255, x_shape)

    # test 0
    y_0 = F_tv.gaussian_blur(x, [3, 3], 1)

    # test 1
    y_1 = F_tv.gaussian_blur(x, [3, 3], [0.5, 0.75])

    # test 2
    y_2 = F_tv.gaussian_blur(x, [3, 3], 2)

    y_batch = F.batch_gaussian_blur(
        torch.stack((x, x, x)), [3, 3], torch.tensor([[1, 1], [0.5, 0.75], [2, 2]])
    )

    torch.testing.assert_close(y_batch, torch.stack((y_0, y_1, y_2)))
