# Speed comparison between Torchvision and Torchaug

## Setting for comparison

We used a [python script](../speed_script.py) that compare the speed between Torchvision and Torchaug on CPU and GPU.

It **does not test** the improvement based on eliminating the CPU/GPU synchronization which should favor Torchaug in comparison with Torchvision.

It was performed on **1 GPU P5000** and **6 logical cores of an Intel(R) Xeon(R) Gold 6126 CPU @ 2.60GHz**.

## Batch Transforms

Input `torch.randn(B, 3, 224, 224)`. In our script B=64.

Using Torchvision, to have random call on each element of a batch, one has to iterate over the elements of the batch as the following:
```python
import torchvision.transforms as transforms

t = transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)
torch.stack([t(img) for img in batch_input])
```
In our implementation, there is no need to iterate over the elements, but some augmentations allow for flexibility between making several random calls for sub-batches and time of computations.


Time in ms averaged on 100 runs for Torchvision and Torchaug.
### CPU

| Augmentation              | Random Calls | Torchvision | Torchaug     |
| ------------------------- | :----------: | :---------: | :------:     |
| BatchRandomColorJitter    | 1            | x           | 819.726      |
| BatchRandomColorJitter    | 8            | x           | 660.376      |
| BatchRandomColorJitter    | B            | 816.436     | **754.716**  |
| BatchRandomGaussianBlur   | B            | 1953.318    | **1570.750** |
| BatchRandomGrayScale      | B            | 88.471      | **20.969**   |
| BatchRandomHorizontalFlip | B            | 73.019      | **18.075**   |
| BatchRandomResizedCrop    | 1            | x           | 117.221      |
| BatchRandomResizedCrop    | 8            | x           | 109.775      |
| BatchRandomResizedCrop    | 16           | x           | 118.750      |
| BatchRandomResizedCrop    | B            | 199.716     | **189.726**  |
| BatchRandomSolarize       | B            | 113.437     | **38.263**   |

### GPU

| Augmentation              | Random Calls | Torchvision | Torchaug   |
| ------------------------- | :----------: | :---------: | :------:   |
| BatchRandomColorJitter    | 1            | x           | 24.430     |
| BatchRandomColorJitter    | 8            | x           | 25.274     |
| BatchRandomColorJitter    | B            | 35.838      | **26.155** |
| BatchRandomGaussianBlur   | B            | 40.826      | **26.870** |
| BatchRandomGrayScale      | B            | 3.140       | **0.718**  |
| BatchRandomHorizontalFlip | B            | 1.485       | **0.562**  |
| BatchRandomResizedCrop    | 1            | x           | 1.568      |
| BatchRandomResizedCrop    | 8            | x           | 1.780      |
| BatchRandomResizedCrop    | 16           | x           | 2.279      |
| BatchRandomResizedCrop    | B            | 7.018       | **6.888**  |
| BatchRandomSolarize       | B            | 1.843       | **0.428**  |

## Transforms

Input `torch.randn(3, 224, 224)`.

Time in ms averaged on 1000 runs for Torchvision and Torchaug.

### CPU
| Augmentation       | Torchvision | Torchaug   |
| ------------------ | :---------: | :------:   |
| Normalize          | 0.696       | **0.398**  |
| RandomColorJitter  | 12.264      | **11.584** |
| RandomGaussianBlur | 30.934      | **30.096** |
| RandomSolarize     | 0.927       | **0.881**  |

### GPU

| Augmentation       | Torchvision | Torchaug  |
| ------------------ | :---------: | :------:  |
| Normalize          | 0.190       | **0.104** |
| RandomColorJitter  | 1.227       | **0.727** |
| RandomGaussianBlur | **0.672**   | 0.699     |
| RandomSolarize     | 0.037       | **0.034** |
