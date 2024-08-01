# Nested Tensors Tutorial

## Introduction

[TANestedTensors](#torchaug.ta_tensors.TANestedTensors) are data structures that store [Tensor](#torch.Tensor) and [TATensor](#torchaug.ta_tensors.TATensor).

They are meant to be used when it is impossible to resize the inputs to collate batches which is particularly interesting during inference for object detections, segmentations, ...

**Torchaug** transforms handle the nested tensors by essentially applying a for loop on each of the samples stored in the nested tensors.

It is not particularly efficient compared to using directly a for loop on batches but it allows one to use Torchaug to transform in the same framework:
1. sample tensors.
2. batched tensors.
3. nested tensors.

## All the nested tensors

We define several [TANestedTensors](#torchaug.ta_tensors.TANestedTensors) as describe in the next subsections.

### NestedTensors

[NestedTensors](#torchaug.ta_tensors.NestedTensors) store [Tensor](#torch.Tensor).

### ImageNestedTensors

[ImageNestedTensors](#torchaug.ta_tensors.ImageNestedTensors) store [Image](#torchaug.ta_tensors.Image).

### VideoNestedTensors

[VideoNestedTensors](#torchaug.ta_tensors.VideoNestedTensors) store [Video](#torchaug.ta_tensors.Video).

### BoundingBoxesNestedTensors

[BoundingBoxesNestedTensors](#torchaug.ta_tensors.BoundingBoxesNestedTensors) store [BoundingBoxes](#torchaug.ta_tensors.BoundingBoxes).

### MaskNestedTensors

[MaskNestedTensors](#torchaug.ta_tensors.MaskNestedTensors) store [Mask](#torchaug.ta_tensors.Mask).

### LabelsNestedTensors

[LabelsNestedTensors](#torchaug.ta_tensors.LabelsNestedTensors) store [Labels](#torchaug.ta_tensors.Labels).


## How NestedTensors are used

### Internally

Internally, Torchaug transforms use nested tensors as a list of tensors. This means that **each sample in the nested tensors are transformed independantly** in a for loop.

**Torchaug automatically detects that nested tensors are used** by looking at the type of its inputs. However, this comes with the cost that Torchaug forces to only use nested tensors for the whole batch if at least one is found.

Torchaug provides also transforms to convert the nested tensors to batches when all samples have the same size or to a list:
- [NestedToList](#torchaug.transforms.NestedToList)
- [NestedToBatch](#torchaug.transforms.NestedToBatch)

### In your code

To use [TANestedTensors](#torchaug.ta_tensors.TANestedTensors), you can simply import the classes and instantiate them:

```python
from torchaug.ta_tensors import NestedImages, Image

images = [
    Image(torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)),
    Image(torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8))
]
nested_images = NestedImages(images)
```

To help you to collate [TANestedTensors](#torchaug.ta_tensors.TANestedTensors) in batches, torchaug provides a [default_nested_collate](#torchaug.data.dataloader.default_nested_collate) to use with [DataLoader](#torch.utils.data.DataLoader):

```python
import torch
from torch.utils.data import DataLoader, Dataset

from torchaug.data.dataloader import default_nested_collate
from torchaug.ta_tensors import Image, ImageNestedTensors


class CustomDataset(Dataset):
    def __init__(self, len: int = 100):
        self._len = len

    def __getitem__(self, idx):
        return Image(torch.randint(0, 256, (3, 224, torch.randint(224, 448, (1,))[0]), dtype=torch.uint8))

    def __len__(self):
        return self._len


dataloader = DataLoader(CustomDataset(), batch_size=2, collate_fn=default_nested_collate)

batch = next(iter(dataloader))
assert isinstance(batch, ImageNestedTensors)
```
