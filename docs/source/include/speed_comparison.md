# Speed comparison between Torchvision and Torchaug

## Setting for comparison

We used a [python script](../speed_script.py) that compare the speed between Torchvision (v0.15.2) and Torchaug on CPU and GPU.

It **does not test** the improvement based on eliminating the CPU/GPU synchronization which should favor Torchaug in comparison with Torchvision.

It was performed on **1 GPU A100** and **8 logical cores of an Intel(R) Xeon(R) Silver 4215R CPU @ 3.20GHz**.

## Batch Transforms

Input `torch.randn(batch_size, 3, 224, 224)`.

Using Torchvision, to have random call on each element of a batch, one has to iterate over the elements of the batch as the following:
```python
import torchvision.transforms as transforms

t = transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)
torch.stack([t(img) for img in batch_input])
```
In our implementation, there is no need to iterate over the elements, but some augmentations allow for flexibility between making several random calls for sub-batches and time of computations.


Time in ms averaged on 100 runs for Torchvision and Torchaug.
### CPU

| Transform                 | Rand Calls   | Torchvision       | Torchaug           | Torchvision     | Torchaug           | Torchvision     | Torchaug           | Torchvision    | Torchaug          |
|---------------------------|--------------|-------------------|--------------------|-----------------|--------------------|-----------------|--------------------|----------------|-------------------|
| **Batch size**            |              | **128**           | **128**            | **64**          | **64**             | **16**          | **16**             | **8**          | **8**
| BatchRandomColorJitter    | 1            |                   | 905.15  ± 44.71    |                 | 336.76  ± 14.02    |                 | 62.27  ± 4.38      |                | 34.39  ± 3.32     |                  | 7.57  ± 8.95     |
| BatchRandomColorJitter    | 8            |                   | 506.70  ± 12.96    |                 | 292.49  ± 5.25     |                 | 130.44  ± 4.18     |                | 67.85  ± 4.03     |
| BatchRandomColorJitter    | -1           | 1475.54  ± 120.92 | **667.68  ± 7.53** | 726.49  ± 76.45 | **374.77  ± 4.12** | 172.52  ± 41.09 | **131.69  ± 3.59** | 85.26  ± 30.58 | **68.11  ± 3.69** |
| BatchRandomGaussianBlur   |              | 753.40  ± 58.82   | **338.49  ± 5.34** | 371.74  ± 46.65 | **145.64  ± 3.91** | 90.75  ± 22.07  | **31.79  ± 1.60**  | 46.71  ± 17.19 | **22.60  ± 2.79** |
| BatchRandomGrayScale      |              | 156.61  ± 5.88    | **47.87  ± 3.81**  | 41.52  ± 4.70   | **11.06  ± 4.44**  | 7.94  ± 2.06    | **2.22  ± 3.16**   | 3.97  ± 3.51   | **2.55  ± 3.27**  |
| BatchRandomHorizontalFlip |              | 138.29  ± 4.64    | **44.62  ± 4.55**  | 25.71  ± 3.57   | **9.56  ± 3.53**   | 3.34  ± 3.46    | **1.59  ± 2.83**   | 1.62  ± 2.81   | **0.64  ± 1.90**  |
| BatchRandomResizedCrop    | 1            |                   | 233.59  ± 21.54    |                 | 72.43  ± 12.90     |                 | 10.36  ± 3.31      |                | 4.95  ± 3.43      |
| BatchRandomResizedCrop    | 8            |                   | 131.33  ± 4.29     |                 | 49.83  ± 3.23      |                 | 16.89  ± 2.28      |                | 16.16  ± 1.23     |
| BatchRandomResizedCrop    | 16           |                   | 133.11  ± 4.21     |                 | 48.70  ± 2.86      |                 | 32.52  ± 2.18      |                | 16.27  ± 1.38     |
| BatchRandomResizedCrop    | -1           | 303.01  ± 4.69    | **299.61  ± 4.04** | 112.53  ± 2.72  | **111.21  ± 1.86** | 26.41  ± 3.25   | **26.18  ± 3.18**  | 13.24  ± 3.29  | **13.01  ± 3.37** |
| BatchRandomSolarize       |              | 185.82  ± 7.84    | **103.43  ± 3.95** | 53.91  ± 5.30   | **23.78  ± 4.10**  | 10.67  ± 3.80   | **3.51  ± 3.51**   | 5.41  ± 3.48   | **1.83  ± 2.96**  |

### GPU


| Transform                 | Rand Calls   | Torchvision   | Torchaug          | Torchvision   | Torchaug          | Torchvision   | Torchaug         | Torchvision   | Torchaug         |
|---------------------------|--------------|---------------|-------------------|---------------|-------------------|---------------|------------------|---------------|------------------|
| **Batch size**            |              | **128**       | **128**           | **64**        | **64**            | **16**        | **16**           | **8**         | **8**            |
| BatchRandomColorJitter    | 1            |               | 2.79  ± 0.24      |               | 1.52  ± 0.01      |               | 1.20  ± 0.01     |               | 1.21  ± 0.09     |
| BatchRandomColorJitter    | 8            |               | 7.30  ± 0.03      |               | 7.39  ± 0.31      |               | 7.39  ± 0.04     |               | 3.87  ± 0.03     |
| BatchRandomColorJitter    | -1           | 61.89  ± 5.92 | **20.47  ± 0.28** | 31.47  ± 3.63 | **17.42  ± 0.07** | 7.79  ± 1.88  | **7.41  ± 0.04** | 4.01  ± 1.38  | **3.88  ± 0.05** |
| BatchRandomGaussianBlur   |              | 15.82  ± 1.39 | **5.43  ± 0.04**  | 7.75  ± 1.01  | **2.91  ± 0.01**  | 1.95  ± 0.44  | **0.99  ± 0.01** | 1.09  ± 0.40  | **0.71  ± 0.01** |
| BatchRandomGrayScale      |              | 5.37  ± 0.36  | **0.32  ± 0.00**  | 2.74  ± 0.28  | **0.19  ± 0.00**  | 0.70  ± 0.14  | **0.17  ± 0.00** | 0.38  ± 0.10  | **0.17  ± 0.00** |
| BatchRandomHorizontalFlip |              | 2.15  ± 0.11  | **0.28  ± 0.00**  | 1.08  ± 0.06  | **0.18  ± 0.00**  | 0.30  ± 0.03  | **0.11  ± 0.00** | 0.18  ± 0.02  | **0.11  ± 0.00** |
| BatchRandomResizedCrop    | 1            |               | 0.99  ± 0.16      |               | 0.55  ± 0.04      |               | 0.25  ± 0.01     |               | 0.22  ± 0.01     |
| BatchRandomResizedCrop    | 8            |               | 1.13  ± 0.02      |               | 1.10  ± 0.01      |               | 1.09  ± 0.01     |               | 1.11  ± 0.02     |
| BatchRandomResizedCrop    | 16           |               | 2.11  ± 0.14      |               | 2.09  ± 0.02      |               | 2.13  ± 0.11     |               | 1.11  ± 0.02     |
| BatchRandomResizedCrop    | -1           | 12.67  ± 0.35 | **12.34  ± 0.16** | 6.31  ± 0.06  | **6.14  ± 0.07**  | 1.62  ± 0.04  | **1.58  ± 0.03** | 0.83  ± 0.01  | **0.82  ± 0.01** |
| BatchRandomSolarize       |              | 3.03  ± 0.23  | **0.41  ± 0.01**  | 1.56  ± 0.25  | **0.24  ± 0.00**  | 0.42  ± 0.09  | **0.15  ± 0.00** | 0.24  ± 0.06  | **0.15  ± 0.00** |




## Transforms

Input `torch.randn(3, 224, 224)`.

Time in ms averaged on 1000 runs for Torchvision and Torchaug.

### CPU

| Transform          | Torchvision      | Torchaug         |
|--------------------|------------------|------------------|
| Normalize          | 0.48  ± 1.74     | **0.41  ± 1.54** |
| RandomColorJitter  | 11.46  ± 11.44   | **7.99  ± 8.22** |
| RandomGaussianBlur | **5.59  ± 6.57** | 5.67  ± 6.13     |
| RandomSolarize     | **0.57  ± 1.82** | 0.59  ± 1.84     |

### GPU

| Transform          | Torchvision      | Torchaug         |
|--------------------|------------------|------------------|
| Normalize          | 0.14  ± 0.11     | **0.07  ± 0.07** |
| RandomColorJitter  | 1.29  ± 25.39    | **0.43  ± 0.42** |
| RandomGaussianBlur | **0.18  ± 0.18** | 0.26  ± 0.24     |
| RandomSolarize     | **0.04  ± 0.02** | 0.04  ± 0.03     |
