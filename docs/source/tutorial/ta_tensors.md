# TA Tensors Tutorial

## Introduction

Torchaug tensors are a subclass of [Tensor](#torch.Tensor). They are largely based on the [torchvision tensors](#torchvision.tv_tensors.TVTensor). Their use is described in detail in [this section](#how-tatensors-are-used).

Therefore a [TATensor](#torchaug.ta_tensors.TATensor) can be used in torch operations just like any [Tensor](#torch.Tensor) but some operations need to be considered carefully We recommand that you first have a look at [Torchvision's documentation](https://pytorch.org/vision/stable/auto_examples/transforms/plot_tv_tensors.html#what-can-i-do-with-a-tvtensor).

## All the TATensors

We define several [TATensor](#torchaug.ta_tensors.TATensor) as describe in the next subsections.

### Image

An [Image](#torchaug.ta_tensors.Image) is a tensor used to represent an image, just as a [TV Image](#torchvision.tv_tensors.Image).

Its dimension is at least `3`.

### Video

A [Video](#torchaug.ta_tensors.Video) is a tensor used to represent an video, just as a [TV Video](#torchvision.tv_tensors.Video).

Its dimension is at least `4`.

### BoundingBoxes

A [BoundingBoxes](#torchaug.ta_tensors.BoundingBoxes) is a tensor used to represent bounding boxes, just as a [TVBoundingBoxes](#torchvision.tv_tensors.BoundingBoxes).

It has the following added attributes:
- `canvas_size` the size of the associated tensor image or video.
- `format` a [BoundingBoxFormat](#torchaug.ta_tensors.BoundingBoxFormat)

### Mask

A [Mask](#torchaug.ta_tensors.Mask) is a tensor used to represent bounding boxes, just as a [TV Mask](#torchvision.tv_tensors.Mask).

Its dimension is at least `2`.

### Labels

A [Labels](#torchaug.ta_tensors.Labels) is a tensor used to represent labels.

### BatchImages

A [BatchImages](#torchaug.ta_tensors.BatchImages) is a tensor used to represent a batch of images.

Its dimension is at least `4`.

### BatchVideos

A [BatchVideos](#torchaug.ta_tensors.BatchVideos) is a tensor used to represent a batch of videos.

Its dimension is at least `5`.

### BatchBoundingBoxes

A [BatchBoundingBoxes](#torchaug.ta_tensors.BatchBoundingBoxes) is a tensor used to represent a batch of bounding boxes.

It has the following added attributes:
- `canvas_size` the size of the associated batch of images or videos.
- `format` a [BoundingBoxFormat](#torchaug.ta_tensors.BoundingBoxFormat)
- `samples_ranges` a list of the range of the indices of the bounding boxes for each sample.

It also have specific behavior and defined class methods to handle some cases that are [documented](#torchaug.ta_tensors.BatchBoundingBoxes).

### BatchMasks

A [BatchMasks](#torchaug.ta_tensors.BatchMasks) is a tensor to represent a batch of masks.

It has the following added attributes:
- `samples_ranges` a list of the range of the indices of the masks for each sample.

It also have specific behavior and defined class methods to handle some cases that are [documented](#torchaug.ta_tensors.BatchMasks).

### BatchLabels

A [BatchLabels](#torchaug.ta_tensors.BatchLabels) is a tensor to represent a batch of labels.

It has the following added attributes:
- `samples_ranges` a list of the range of the indices of the labels for each sample.

It also have specific behavior and defined class methods to handle some cases that are [documented](#torchaug.ta_tensors.BatchLabels).

## How TATensors are used

### Internally

Internally, Torchaug uses the same notion of `kernel` for its functionals as in Torchvision. This means that each type of [TATensor](#torchaug.ta_tensors.TATensor) can have different transformations. The [Transforms](#torchaug.transforms.RandomApplyTransform) correctly dispatch the inputs to the functionals.

For example:
- [F.resize](#torchaug.transforms.functional.resize) should work differently for [BatchImages](#torchaug.ta_tensors.BatchImages) and [BatchBoundingBoxes](#torchaug.ta_tensors.BatchBoundingBoxes).
- [F.adjust_brightness_batch](#torchaug.transforms.functional.adjust_brightness_batch) should be defined for [BatchImages](#torchaug.ta_tensors.BatchImages) and not for [BatchBoundingBoxes](#torchaug.ta_tensors.BatchBoundingBoxes)

### In your code

To use [TA tensors](#torchaug.ta_tensors.TATensor), you can simply import the classes and instantiate them as a [TVTensor](#torchvision.tv_tensors.TVTensor):

```python
from torchaug.ta_tensors import Image

image = Image(torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8))
```

To help you to collate TATensors in batches, torchaug provides a [default_collate](#torchaug.data.dataloader.default_collate) to use with [DataLoader](#torch.utils.data.DataLoader).

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchaug.data.dataloader import default_collate
from torchaug.ta_tensors import Image, BatchImages

class CustomDataset(Dataset):
    def __init__(self, len: int = 100):
        self._len = len

    def __getitem__(self, idx):
        return Image(
            torch.randint(0, 256, (3, 224, 224),
            dtype=torch.uint8)
        )

    def __len__(self):
        return self._len

dataloader = DataLoader(CustomDataset(), batch_size=2, collate_fn=default_collate)

batch = next(iter(dataloader))
assert isinstance(batch, BatchImages)
assert list(batch.shape) == [2, 3, 224, 224]
```
