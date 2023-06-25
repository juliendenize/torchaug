# GPU Torchvision

## Introduction

Torchvision has been implemented over Pytorch and Pillow to perform Computer Vision in the Pytorch Ecosystem. One of its core functionality is to perform Data augmentations. However because it has been implemented first for CPU and later with GPU in mind, it has several drawbacks to make it work on GPUs:

- Some CPU/GPU synchronizations cannot be avoided.
- Batch data augmentations are not completly random, meaning that random parameters are sampled for the whole batch and not each unique components.

This library aims to remove these issues. It acts like a complementary library with Torchvision and do not use the same namespace. Still, it follows the same nomenclature as Torchvision with *functional* augmentations and *class* wrappers.

To be sure to retrieve the same data augmentations as Torchvision, it has been tested on each of its augmentations.

## How to use

1. Install a Pytorch environment.

2. Install GPU Torchvision

```bash
pip3 install gpu_torchvision
```

3. Import data augmentations just as Torchvision

```python
from gpu_torchvision.transforms import RandomColorJitter

transform = RandomColorJitter(...)
```

## How to contribute

Feel free to contribute to this library by making and/or pull requests. For each features you implement, add tests to be sure it works.