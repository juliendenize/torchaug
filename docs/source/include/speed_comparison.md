# Speed comparison between Torchvision and Torchaug

## Setting for comparison

We used a [python script](../speed_script.py) that compare the speed between Torchvision and Torchaug on CPU and GPU.

It **does not test** the improvement based on eliminating the CPU/GPU synchronization which should favor Torchaug in comparison with Torchvision.

It was performed on **1 GPU A100** and **8 logical cores of an Intel(R) Xeon(R) Silver 4215R CPU @ 3.20GHz**.

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

| Transform                 | num_rand_calls   | Torchvision (ms)    | Torchaug (ms)           |
|---------------------------|------------------|---------------------|-------------------------|
| BatchRandomColorJitter    | 1                |                     | 556.25 +/- 37.48        |
| BatchRandomColorJitter    | 8                |                     | 501.83 +/- 14.29        |
| BatchRandomColorJitter    | -1               | 1632.88 +/- 8444.97 | **1559.92 +/- 6900.16** |
| BatchRandomGaussianBlur   |                  | 383.29 +/- 42.33    | **125.64 +/- 5.01**     |
| BatchRandomGrayScale      |                  | 39.91 +/- 6.48      | **10.41 +/- 3.55**      |
| BatchRandomHorizontalFlip |                  | 16.81 +/- 5.00      | **8.40 +/- 3.39**       |
| BatchRandomResizedCrop    | 1                | 64.16 +/- 12.34     |
| BatchRandomResizedCrop    | 8                |                     | 45.78 +/- 3.23          |
| BatchRandomResizedCrop    | 16               |                     | 53.74 +/- 3.40          |
| BatchRandomResizedCrop    | -1               | 114.29 +/- 4.66     | **110.43 +/- 4.34**     |
| BatchRandomSolarize       |                  | 29.58 +/- 4.46      | **17.98 +/- 3.59**      |


### GPU

| Transform                 | num_rand_calls   | Torchvision (ms) | Torchaug (ms)      |
|---------------------------|------------------|------------------|--------------------|
| BatchRandomColorJitter    | 1                |                  | 4.23 +/- 0.30      |
| BatchRandomColorJitter    | 8                |                  | 8.25 +/- 0.13      |
| BatchRandomColorJitter    | -1               | 29.91 +/- 4.13   | **19.65 +/- 0.41** |
| BatchRandomGaussianBlur   |                  | 8.07 +/- 1.15    | **2.92 +/- 0.04**  |
| BatchRandomGrayScale      |                  | 2.70 +/- 0.30    | **0.19 +/- 0.00**  |
| BatchRandomHorizontalFlip |                  | 1.10 +/- 0.09    | **0.18 +/- 0.00**  |
| BatchRandomResizedCrop    | 1                |                  | 0.56 +/- 0.13      |
| BatchRandomResizedCrop    | 8                |                  | 1.09 +/- 0.015     |
| BatchRandomResizedCrop    | 16               |                  | 2.06 +/- 0.021     |
| BatchRandomResizedCrop    | -1               | 6.44 +/- 0.30    | **6.21 +/- 0.18**  |
| BatchRandomSolarize       |                  | 1.46 +/- 0.14    | **0.22 +/- 0.01**  |
## Transforms

Input `torch.randn(3, 224, 224)`.

Time in ms averaged on 1000 runs for Torchvision and Torchaug.

### CPU
| Transform           | Torchvision (ms)   | Torchaug (ms)       |
|---------------------|--------------------|---------------------|
| Normalize           | 42.69 +/- 86.65    | **0.43 +/- 1.59**   |
| RandomColorJitter   | 49.49 +/- 516.74   | **11.29 +/- 11.58** |
| RandomGaussianBlur  | 15.46 +/- 61.11    | **5.53 +/- 6.05**   |
| RandomSolarize      | **0.51 +/- 1.72**  | 0.53 +/- 1.76       |


### GPU

| Transform          | Torchvision (ms)  | Torchaug (ms)     |
|--------------------|-------------------|-------------------|
| Normalize          | 0.14 +/- 0.11     | **0.07 +/- 0.08** |
| RandomColorJitter  | 1.26 +/- 24.85    | **0.45 +/- 0.45** |
| RandomGaussianBlur | **0.18 +/- 0.20** | 0.25 +/- 0.23     |
| RandomSolarize     | **0.04 +/- 0.02** | 0.04 +/- 0.03     |
