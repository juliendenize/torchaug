import pytest
import torch
import torchvision.transforms.functional as F_tv

import torchaug.transforms.functional as F


def test_gaussian_blur():
    x_shape = [3, 4, 4]
    x = torch.randint(0, 255, x_shape)

    # test value sigma None.
    y_0 = F_tv.gaussian_blur(x, [3, 3], None)
    y = F.gaussian_blur(x, [3, 3], None)
    torch.testing.assert_close(y, y_0)

    # test value sigma 1.
    y_0 = F_tv.gaussian_blur(x, [3, 3], 1)
    y = F.gaussian_blur(x, [3, 3], 1)
    torch.testing.assert_close(y, y_0)

    # test value sigma [1].
    y_0 = F_tv.gaussian_blur(x, [3, 3], 1)
    y = F.gaussian_blur(x, [3, 3], [1])
    torch.testing.assert_close(y, y_0)

    # test value sigma [0.5, 0.75].
    y_1 = F_tv.gaussian_blur(x, [3, 3], [0.5, 0.75])
    y = F.gaussian_blur(x, [3, 3], [0.5, 0.75])
    torch.testing.assert_close(y, y_1)

    # test value sigma 2.
    y_2 = F_tv.gaussian_blur(x, [3, 3], 2.0)
    y = F.gaussian_blur(x, [3, 3], 2.0)
    torch.testing.assert_close(y, y_2)

    # test value sigma Tensor(2.).
    y_2 = F_tv.gaussian_blur(x, [3, 3], 2.0)
    y = F.gaussian_blur(x, [3, 3], torch.tensor(2.0))
    torch.testing.assert_close(y, y_2)

    # test value sigma Tensor([2.]).
    y_2 = F_tv.gaussian_blur(x, [3, 3], 2.0)
    y = F.gaussian_blur(x, [3, 3], torch.tensor([2.0]))
    torch.testing.assert_close(y, y_2)

    # test value sigma Tensor([2., 2.]).
    y_2 = F_tv.gaussian_blur(x, [3, 3], 2.0)
    y = F.gaussian_blur(x, [3, 3], torch.tensor([2.0]))
    torch.testing.assert_close(y, y_2)

    # test kernel has one element.
    torch.testing.assert_close(
        F.gaussian_blur(x, 3, [0.5, 0.75]), F.gaussian_blur(x, [3, 3], [0.5, 0.75])
    )

    # test kernel is not int or not sequence of 2 int.
    with pytest.raises(
        TypeError,
        match="kernel_size should be int or a sequence of integers. Got <class 'float'>.",
    ):
        F.gaussian_blur(x, 3.0, [2, 2])
    with pytest.raises(
        ValueError, match="If kernel_size is a sequence its length should be 2. Got 3."
    ):
        F.gaussian_blur(x, [3, 3, 3], [2, 2])

    # test kernel does not contain even or zero number.
    with pytest.raises(
        ValueError,
        match=r"kernel_size should have odd and positive integers. Got \[3, 2\].",
    ):
        F.gaussian_blur(x, [3, 2], [2, 2])
    with pytest.raises(
        ValueError,
        match=r"kernel_size should have odd and positive integers. Got \[3, 0\].",
    ):
        F.gaussian_blur(x, [3, 0], [2, 2])

    # test sigma int is inferior or equal to 0.
    with pytest.raises(ValueError, match="sigma should have positive values. Got 0."):
        F.gaussian_blur(x, [3, 3], 0)
    with pytest.raises(ValueError, match="sigma should have positive values. Got -1."):
        F.gaussian_blur(x, [3, 3], -1)

    # test sigma sequence int has more than 2 elements.
    with pytest.raises(
        ValueError, match="If sigma is a sequence, its length should be 2. Got 3."
    ):
        F.gaussian_blur(x, [3, 3], [3, 3, 3])

    # test sigma tensor has more than 1 dimension.
    with pytest.raises(
        ValueError, match="If sigma is a tensor, its dimension should be 1. Got 2."
    ):
        F.gaussian_blur(x, [3, 3], torch.tensor([[3.0]]))

    # test sigma tensor has more than 2 elements.
    with pytest.raises(
        ValueError,
        match="If sigma is a tensor of multiple values, its length should be 2. Got 3.",
    ):
        F.gaussian_blur(x, [3, 3], torch.tensor([3.0, 3.0, 3.0]))

    # test sigma sequence float has more than 2 elements.
    with pytest.raises(
        ValueError, match="If sigma is a sequence, its length should be 2. Got 3."
    ):
        F.gaussian_blur(x, [3, 3], [3.0, 3.0, 3.0])

    # test sigma is of wrong type.
    with pytest.raises(
        TypeError,
        match="sigma should be either int, float or sequence of floats or int or tensor. Got <class 'str'>.",
    ):
        F.gaussian_blur(x, [3, 3], "ahah")

    # Test input is not tensor.
    with pytest.raises(
        TypeError, match="Object should be a tensor. Got <class 'list'>."
    ):
        F.gaussian_blur([2, 0, 3], [3, 3], [2, 2])


def test_normalize():
    torch.manual_seed(28)

    for channels in [1, 3]:
        img = torch.rand(channels, 10, 10)
        mean = [img[c].mean().item() for c in range(channels)]
        std = [img[c].std().item() for c in range(channels)]

    x_shape = [3, 4, 4]
    x = torch.rand(x_shape)

    # Test not inplace.
    torch.testing.assert_close(
        F.normalize(x, mean, std, False), F_tv.normalize(x, mean, std, False)
    )

    # Test not inplace and value check that passes.
    torch.testing.assert_close(
        F.normalize(x, mean, std, False, True), F_tv.normalize(x, mean, std, False)
    )

    x_tv = x.clone()
    y = x.clone()

    # Test inplace.
    torch.testing.assert_close(
        F.normalize(y, mean, std, True), F_tv.normalize(x_tv, mean, std, True)
    )

    # Test input is not tensor.
    with pytest.raises(
        TypeError, match="Object should be a tensor. Got <class 'list'>."
    ):
        F.normalize([2, 0, 3], mean, std)

    # Test input is float tensor.
    with pytest.raises(
        TypeError, match="Input tensor should be a float tensor. Got torch.uint8."
    ):
        F.normalize(torch.randint(3, 10, (3, 10, 10), dtype=torch.uint8), mean, std)

    # Test if std contains a 0.
    with pytest.raises(
        ValueError,
        match="std contains a zero leading to division by zero.",
    ):
        F.normalize(x, mean, [0, 0.225, 0.225], False, True)


def test_solarize():
    torch.manual_seed(28)

    x_shape = [3, 4, 4]
    x_uint = torch.randint(0, 255, x_shape, dtype=torch.uint8)
    x_float = torch.rand(x_shape)

    # Test input is not tensor.
    with pytest.raises(
        TypeError, match="Object should be a tensor. Got <class 'list'>."
    ):
        F.solarize([2, 0, 3], 0.5)

    # Test input has fewer than 3 dimensions.
    with pytest.raises(
        TypeError,
        match="Input image tensor should have at least 3 dimensions, but found 2.",
    ):
        F.solarize(torch.randint(0, 255, (10, 10)), 0.5)

    # Test if threshold not float or not tensor.
    with pytest.raises(
        TypeError, match="threshold should be a float or a tensor of one element."
    ):
        F.solarize(x_uint, [0.5])

    # Test threshold int
    torch.testing.assert_close(F.solarize(x_uint, 128), F_tv.solarize(x_uint, 128))

    # Test threshold float
    torch.testing.assert_close(F.solarize(x_float, 0.5), F_tv.solarize(x_float, 0.5))

    # Test threshold value for int
    with pytest.raises(ValueError, match="Threshold should be less than bound of img."):
        F.solarize(x_uint, 256, True)
    F.solarize(x_uint, 130, True)

    # Test threshold value for float
    with pytest.raises(ValueError, match="Threshold should be less than bound of img."):
        F.solarize(x_float, 1.1, True)
    F.solarize(x_float, 0.5, True)
