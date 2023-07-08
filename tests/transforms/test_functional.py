import torch
import torchvision.transforms.functional as F_tv

import gpu_torchvision.transforms.functional as F


def test_gaussian_blur():
    x_shape = [3, 4, 4]
    x = torch.randint(0, 255, x_shape)

    # test 0
    y_0 = F_tv.gaussian_blur(x, [3, 3], 1)
    y = F.gaussian_blur(x, [3, 3], [1, 1])
    torch.testing.assert_close(y, y_0)

    # test 1
    y_1 = F_tv.gaussian_blur(x, [3, 3], [0.5, 0.75])
    y = F.gaussian_blur(x, [3, 3], [0.5, 0.75])
    torch.testing.assert_close(y, y_1)

    # test 2
    y_2 = F_tv.gaussian_blur(x, [3, 3], 2)
    y = F.gaussian_blur(x, [3, 3], [2, 2])
    torch.testing.assert_close(y, y_2)


def test_normalize():
    torch.manual_seed(28)

    for channels in [1, 3]:
        img = torch.rand(channels, 10, 10)
        mean = [img[c].mean().item() for c in range(channels)]
        std = [img[c].std().item() for c in range(channels)]

    x_shape = [3, 4, 4]
    x = torch.randn(x_shape)

    torch.testing.assert_close(
        F.normalize(x, mean, std, False), F_tv.normalize(x, mean, std, False)
    )

    x_tv = x.clone()
    y = x.clone()
    torch.testing.assert_close(
        F.normalize(y, mean, std, True), F_tv.normalize(x_tv, mean, std, True)
    )


def test_solarize():
    torch.manual_seed(28)

    x_shape = [3, 4, 4]
    x = torch.randint(0, 255, x_shape)

    torch.testing.assert_close(F.solarize(x, 128), F_tv.solarize(x, 128))
