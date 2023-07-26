**Efficient vision data augmentations for on GPU and/or batched data.**

***Under active development, subject to API change***

[![PyPI python](https://img.shields.io/pypi/pyversions/torchaug)](https://pypi.org/project/torchaug)
[![PyPI version](https://badge.fury.io/py/torchaug.svg)](https://pypi.org/project/torchaug)
[![codecov](https://codecov.io/gh/juliendenize/torchaug/branch/main/graph/badge.svg?token=CA266XDW8D)](https://codecov.io/gh/juliendenize/torchaug)
[![License](https://img.shields.io/badge/license-CeCILL--C-green.svg)](LICENSE)

# Torchaug

## Introduction

Torchvision has been implemented over Pytorch and Pillow to perform Computer Vision in the Pytorch Ecosystem. One of its core functionality is to perform Data augmentations. However because it has been implemented first for CPU and later with GPU in mind, it has several drawbacks to make it work on GPUs:

- Some CPU/GPU synchronizations cannot be avoided.
- Batch data augmentations are not completely random, meaning that random parameters are sampled for the whole batch and not each unique component.

This library aims to remove these issues. It acts like a complementary library with Torchvision and does not use the same namespace. Still, it follows the same nomenclature as Torchvision with *functional* augmentations and *class* wrappers.

To be sure to retrieve the same data augmentations as Torchvision, it has been tested on each of its augmentations.

See [here](augmentations.md) the implemented data augmentations and [here](speed_comparison.md) the speed comparison with Torchvision.

## How to use

0. Install a Pytorch >= 2.0 environment and Python >= 3.8.

1. Clone the repo.

2. Install Torchaug

```bash
cd torchaug
pip3 install .
```

3. Import data augmentations just as Torchvision

```python
from torchaug.transforms import RandomColorJitter
from torchaug.batch_transforms import BatchRandomColorJitter

transform = RandomColorJitter(...)
batch_transform = BatchRandomColorJitter(...)
```

## How to contribute

Feel free to contribute to this library by making issues and/or pull requests. For each feature you implement, add tests to make sure it works.

## LICENSE

This project is under the CeCILL license 2.1.
