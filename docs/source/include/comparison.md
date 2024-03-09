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
| RandomColorJitter    | 1            |               | 13.15  ± 10.03    |               | 3.99  ± 0.10     |                  | 0.79  ± 0.02     |                  | 0.52  ± 0.04     |
| RandomColorJitter    | 8            |               | 7.37  ± 0.08      |               | 6.14  ± 0.14     |                  | 3.13  ± 0.10     |                  | 1.69  ± 0.08     |
| RandomColorJitter    | -1           | 22.93  ± 1.68 | **8.46  ± 0.61**  | 11.22  ± 1.53 | **6.15  ± 0.23** | **2.85  ± 0.62** | 3.11  ± 0.11     | **1.48  ± 0.50** | 1.63  ± 0.05     |
| RandomGaussianBlur   |              | 18.13  ± 2.91 | **17.36  ± 0.19** | 8.73  ± 1.28  | **8.54  ± 0.09** | 2.40  ± 0.53     | **2.04  ± 0.05** | 1.20  ± 0.40     | **1.04  ± 0.02** |
| RandomGrayscale      |              | 2.16  ± 0.14  | **0.52  ± 0.01**  | 1.09  ± 0.08  | **0.28  ± 0.01** | 0.29  ± 0.05     | **0.11  ± 0.01** | 0.15  ± 0.03     | **0.08  ± 0.01** |
| RandomHorizontalFlip |              | 1.22  ± 0.06  | **0.42  ± 0.03**  | 0.61  ± 0.04  | **0.22  ± 0.01** | 0.16  ± 0.02     | **0.08  ± 0.01** | 0.10  ± 0.02     | **0.06  ± 0.01** |
| RandomResizedCrop    | 1            |               | 2.19  ± 0.18      |               | 1.04  ± 0.08     |                  | 0.20  ± 0.01     |                  | 0.12  ± 0.01     |
| RandomResizedCrop    | 8            |               | 2.00  ± 0.07      |               | 1.07  ± 0.04     |                  | 0.44  ± 0.02     |                  | 0.39  ± 0.02     |
| RandomResizedCrop    | 16           |               | 2.03  ± 0.08      |               | 1.19  ± 0.04     |                  | 0.74  ± 0.03     |                  | 0.39  ± 0.02     |
| RandomResizedCrop    | -1           | 6.21  ± 0.18  | **5.54  ± 0.11**  | 3.09  ± 0.14  | **2.80  ± 0.09** | 0.80  ± 0.05     | **0.74  ± 0.04** | 0.41  ± 0.03     | **0.39  ± 0.03** |
| RandomSolarize       |              | 1.96  ± 0.86  | **0.56  ± 0.05**  | 0.92  ± 0.30  | **0.27  ± 0.03** | 0.25  ± 0.05     | **0.10  ± 0.01** | 0.13  ± 0.03     | **0.07  ± 0.01** |
