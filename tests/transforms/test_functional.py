import pytest
import torch
import torchvision.transforms.functional as F_tv
from torchvision.transforms.functional import convert_image_dtype

import torchaug.transforms.functional as F


def test_div_255():
    x_shape = [3, 4, 4]
    x = torch.rand(x_shape)

    out = F.div_255(x, inplace=False)
    torch.testing.assert_close(out, x / 255)

    F.div_255(x, inplace=True)
    torch.testing.assert_close(x, out)


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

    # test sigma GPU transfer (if available)
    if torch.cuda.is_available():
        y_2 = F_tv.gaussian_blur(x, [3, 3], 2.0)
        y = F.gaussian_blur(x.cuda(), [3, 3], torch.tensor([2.0])).cpu()
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


def test_mixup():
    torch.manual_seed(28)

    x_shape = [3, 4, 4]
    x_1 = torch.randn(x_shape)
    x_2 = torch.randn(x_shape)

    # Test lam float
    out = F.mixup(x_1, x_2, 0.5, False)
    expected_out = 0.5 * x_1 + 0.5 * x_2
    torch.testing.assert_close(out, expected_out)

    # Test lam tensor not float
    with pytest.raises(TypeError, match="lam should be float. Got <class 'str'>."):
        out = F.mixup(x_1, x_2, "ahah", False)

    # Test tensor_2 is not float.
    with pytest.raises(
        TypeError, match="Tensors should be float. Got torch.float32 and torch.int32."
    ):
        out = F.mixup(x_1, x_2.to(torch.int32), 0.5, False)

    # Test tensor_1 is not float.
    with pytest.raises(
        TypeError, match="Tensors should be float. Got torch.int32 and torch.float32."
    ):
        out = F.mixup(x_1.to(torch.int32), x_2, 0.5, False)


def test_mul_255():
    x_shape = [3, 4, 4]
    x = torch.rand(x_shape)

    out = F.mul_255(x, inplace=False)
    torch.testing.assert_close(out, x * 255)

    F.mul_255(x, inplace=True)
    torch.testing.assert_close(x, out)


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
        F.normalize(x, mean, std, inplace=False),
        F_tv.normalize(x, mean, std, inplace=False),
    )

    # Test not inplace and value check that passes.
    torch.testing.assert_close(
        F.normalize(x, mean, std, inplace=False, value_check=True),
        F_tv.normalize(x, mean, std, inplace=False),
    )

    x_tv = x.clone()
    y = x.clone()

    # Test inplace.
    torch.testing.assert_close(
        F.normalize(y, mean, std, inplace=True),
        F_tv.normalize(x_tv, mean, std, inplace=True),
    )

    # Test input is not tensor.
    with pytest.raises(
        TypeError, match="Object should be a tensor. Got <class 'list'>."
    ):
        F.normalize([2, 0, 3], mean, std)

    # Test input is int tensor.
    int_tensor = torch.randint(1, 256, (3, 10, 10), dtype=torch.uint8)
    torch.testing.assert_close(
        F.normalize(int_tensor, mean, std, cast_dtype=torch.float32, inplace=False),
        F_tv.normalize(
            convert_image_dtype(int_tensor, dtype=torch.float32),
            mean,
            std,
            inplace=False,
        ),
    )
    with pytest.raises(
        TypeError,
        match="Input tensor should be a float tensor or cast_dtype set to a float dtype. Got torch.uint8 and None.",
    ):
        F.normalize(int_tensor, mean, std, cast_dtype=None)
    with pytest.raises(
        ValueError,
        match="cast_dtype should be a float dtype. Got torch.int32.",
    ):
        F.normalize(int_tensor, mean, std, cast_dtype=torch.int32)

    # Test if std contains a 0.
    with pytest.raises(
        ValueError,
        match="std contains a zero leading to division by zero.",
    ):
        F.normalize(
            x, mean, [0, 0.225, 0.225], cast_dtype=None, inplace=False, value_check=True
        )


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

    # Test if threshold not float or not tensor
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
