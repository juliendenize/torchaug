**Efficient vision data augmentations for CPU/GPU per-sample/batched data.**

***Under active development, subject to API change***

[![PyPI python](https://img.shields.io/pypi/pyversions/torchaug)](https://pypi.org/project/torchaug)
[![PyPI version](https://badge.fury.io/py/torchaug.svg)](https://pypi.org/project/torchaug)
[![documentation](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchaug%2Fjson&query=%24.info.version&colorB=brightgreen)](https://torchaug.readthedocs.io/en/stable/)
[![codecov](https://codecov.io/gh/juliendenize/torchaug/branch/main/graph/badge.svg?token=CA266XDW8D)](https://codecov.io/gh/juliendenize/torchaug)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

<!-- start doc -->

# Torchaug

## Introduction

Torchaug is a data augmentation library for the Pytorch ecosystem. It is meant to deal efficiently with tensors that are either on CPU or GPU and either per-samples or on batches.

It seeks to improve Torchvision performance that has been implemented over Pytorch and Pillow to, among other things, perform data augmentations. However, because it has been implemented first with per-sample CPU data augmentations in mind, it has several drawbacks to make it efficient:

- For data augmentations on GPU, some CPU/GPU synchronizations cannot be avoided.
- For data augmentations applied on batch, the randomness is sampled for the whole batch and not each sample.

Torchaug removes these issues and is meant to be used complimentary with Torchvision. It follows the same nomenclature as Torchvision with *functional* augmentations and *transforms* class wrappers. It is split into two packages:
- **transforms** for per-sample data augmentations
- **batch_transfroms** for batched data augmentations.

More details can be found in the [documentation](https://torchaug.readthedocs.io/en/stable/).

To be sure to retrieve the same data augmentations as Torchvision, it has been tested on each of its components to match Torchvision outputs.

## How to use

0. Install a Pytorch >= 2.0 environment.

1. Install Torchaug.

```bash
pip install torchaug
```

2. Import data augmentations either from `torchaug.transforms` or `torchaug.batch_transforms` packages. To ease with handling multiple sequential augmentations, wrappers have been defined.

```python
from torchaug.transforms import (
    ImageWrapper,
    RandomColorJitter,
    RandomGaussianBlur
)
from torchaug.batch_transforms import (
    BatchImageWrapper,
    BatchRandomColorJitter,
    BatchRandomHorizontalFlip
)

transform = ImageWrapper(
    [RandomColorJitter(...), RandomGaussianBlur(...)],
)

batch_transform = BatchImageWrapper(
    [BatchRandomColorJitter(...), BatchRandomHorizontalFlip(...)],
    inplace=True
)
```

## How to contribute

Feel free to contribute to this library by making issues and/or pull requests. For each feature you implement, add tests to make sure it works. Also, make sure to update the documentation.

## LICENSE

This project is under the Apache license 2.0.

<!-- end doc -->
