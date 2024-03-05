# Speed comparison between Torchvision and Torchaug

## Setting for comparison

We used a [python script](../speed_script.py) that compare the speed between Torchvision (v0.17.1) and Torchaug on GPU.

It **does not test** the improvement based on eliminating the CPU/GPU synchronization which should favor Torchaug in comparison with Torchvision.

## Comparison

Using Torchvision, to have random call on each element of a Input batch sizes [128, 64, 16, 8] with shape [3, 224, 224] on cuda (RTX 2000 Ada).

Input `torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=uint8)`.

| Transform            | Num chunks   | Torchvision   | Torchaug          | Torchvision   | Torchaug         | Torchvision      | Torchaug         | Torchvision      | Torchaug         |
|----------------------|--------------|---------------|-------------------|---------------|------------------|------------------|------------------|------------------|------------------|
| **Batch size**       |              | **128**       | **128**           | **64**        | **64**           | **16**           | **16**           | **8**            | **8**            |
| RandomColorJitter    | 1            |               | 13.09  ± 10.48    |               | 3.92  ± 0.08     |                  | 0.78  ± 0.03     |                  | 0.52  ± 0.05     |
| RandomColorJitter    | 8            |               | 7.19  ± 0.12      |               | 6.46  ± 0.57     |                  | 3.22  ± 0.10     |                  | 1.70  ± 0.09     |
| RandomColorJitter    | -1           | 22.98  ± 2.23 | **8.67  ± 0.20**  | 11.64  ± 1.47 | **6.31  ± 0.11** | **2.87  ± 0.82** | 3.24  ± 0.19     | **1.60  ± 0.53** | 1.69  ± 0.06     |
| RandomGaussianBlur   |              | 17.72  ± 2.77 | **17.28  ± 0.78** | 8.85  ± 1.11  | **8.45  ± 0.11** | 2.20  ± 0.53     | **2.02  ± 0.04** | 1.24  ± 0.35     | **1.06  ± 0.03** |
| RandomGrayscale      |              | 2.35  ± 0.74  | **0.57  ± 0.04**  | 1.09  ± 0.10  | **0.28  ± 0.01** | 0.29  ± 0.04     | **0.11  ± 0.01** | 0.15  ± 0.03     | **0.08  ± 0.01** |
| RandomHorizontalFlip |              | 1.21  ± 0.05  | **0.41  ± 0.03**  | 0.61  ± 0.05  | **0.22  ± 0.01** | 0.16  ± 0.02     | **0.09  ± 0.01** | 0.08  ± 0.01     | **0.06  ± 0.01** |
| RandomResizedCrop    | 1            |               | 2.19  ± 0.16      |               | 1.03  ± 0.08     |                  | 0.19  ± 0.01     |                  | 0.12  ± 0.01     |
| RandomResizedCrop    | 8            |               | 1.97  ± 0.07      |               | 1.04  ± 0.04     |                  | 0.42  ± 0.02     |                  | 0.39  ± 0.02     |
| RandomResizedCrop    | 16           |               | 2.02  ± 0.07      |               | 1.17  ± 0.03     |                  | 0.73  ± 0.04     |                  | 0.40  ± 0.03     |
| RandomResizedCrop    | -1           | 6.17  ± 0.16  | **5.48  ± 0.07**  | 3.05  ± 0.11  | **2.74  ± 0.07** | 0.79  ± 0.04     | **0.73  ± 0.04** | 0.41  ± 0.03     | **0.39  ± 0.03** |
| RandomSolarize       |              | 1.71  ± 0.10  | **0.57  ± 0.05**  | 0.86  ± 0.09  | **0.25  ± 0.01** | 0.22  ± 0.04     | **0.10  ± 0.01** | 0.12  ± 0.02     | **0.07  ± 0.02** |
