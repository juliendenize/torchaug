import time

import torch
from torchvision import transforms as T

from torchaug import batch_transforms as BF
from torchaug import transforms as F

torch.set_printoptions(precision=3)

runs = 1000


torchvision_transforms = [
    T.Normalize([225], [225]),
    T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.1)], 0.5),
    T.RandomApply([T.GaussianBlur([23, 23], [0.1, 2.0])], 0.5),
    T.RandomSolarize(0.5, 0.5),
]

cpu_transforms = [
    F.Normalize([225], [225]),
    F.RandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5),
    F.RandomGaussianBlur([23, 23], [0.1, 2.0], 0.5),
    F.RandomSolarize(0.5, 0.5),
]

cuda_transforms = [
    F.Normalize([225], [225]).cuda(),
    F.RandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5).cuda(),
    F.RandomGaussianBlur([23, 23], [0.1, 2.0], 0.5).cuda(),
    F.RandomSolarize(0.5, 0.5).cuda(),
]

input = torch.randn(3, 224, 224)
cuda_input = input.cuda()


for t in torchvision_transforms:
    results = torch.zeros(runs)
    for i in range(runs):
        torch.cuda.synchronize()
        start = time.process_time()
        t(input)
        torch.cuda.synchronize()
        end = time.process_time()
        results[i] = end - start
    results *= 1000
    print(f"CPU {type(t)} in {results.mean()} +/- {torch.std(results)} ms")

for t in torchvision_transforms:
    results = torch.zeros(runs)
    for i in range(runs):
        torch.cuda.synchronize()
        start = time.process_time()
        t(cuda_input)
        torch.cuda.synchronize()
        end = time.process_time()
        results[i] = end - start
    results *= 1000
    print(f"CUDA {type(t)} in {results.mean()} +/- {torch.std(results)} ms")

for t in cpu_transforms:
    results = torch.zeros(runs)
    for i in range(runs):
        torch.cuda.synchronize()
        start = time.process_time()
        t(input)
        torch.cuda.synchronize()
        end = time.process_time()
        results[i] = end - start
    results *= 1000
    print(f"CPU {type(t)} in {results.mean()} +/- {torch.std(results)} ms")


for t in cuda_transforms:
    results = torch.zeros(runs)
    for i in range(runs):
        torch.cuda.synchronize()
        start = time.process_time()
        t(cuda_input)
        torch.cuda.synchronize()
        end = time.process_time()
        results[i] = end - start
    results *= 1000
    print(f"CUDA {type(t)} in {results.mean()} +/- {torch.std(results)} ms")


batch_input = torch.randn(64, 3, 224, 224)
batch_input_cuda = batch_input.cuda()

batch_torchvision_transforms = [
    T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.1)], 0.5),
    T.RandomApply([T.GaussianBlur([23, 23], [0.1, 2.0])], 0.5),
    T.RandomApply([T.Grayscale(num_output_channels=3)], 0.5),
    T.RandomHorizontalFlip(0.5),
    T.RandomResizedCrop([224, 224], antialias=True),
    T.RandomSolarize(0.5, 0.5),
]

cpu_batch_transforms = [
    BF.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 1, inplace=True),
    BF.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 8, inplace=True),
    BF.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, -1, inplace=True),
    BF.BatchRandomGaussianBlur([23, 23], [0.1, 2], 0.5, inplace=True),
    BF.BatchRandomGrayScale(0.5, inplace=True),
    BF.BatchRandomHorizontalFlip(0.5, inplace=True),
    BF.BatchRandomResizedCrop([224, 224], num_rand_calls=1),
    BF.BatchRandomResizedCrop([224, 224], num_rand_calls=8),
    BF.BatchRandomResizedCrop([224, 224], num_rand_calls=16),
    BF.BatchRandomResizedCrop([224, 224], num_rand_calls=-1),
]

cuda_batch_transforms = [
    BF.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 1, inplace=True).cuda(),
    BF.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 8, inplace=True).cuda(),
    BF.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, -1, inplace=True).cuda(),
    BF.BatchRandomGaussianBlur([23, 23], [0.1, 2], 0.5, inplace=True).cuda(),
    BF.BatchRandomGrayScale(0.5, inplace=True).cuda(),
    BF.BatchRandomHorizontalFlip(0.5, inplace=True).cuda(),
    BF.BatchRandomResizedCrop([224, 224], num_rand_calls=1).cuda(),
    BF.BatchRandomResizedCrop([224, 224], num_rand_calls=8).cuda(),
    BF.BatchRandomResizedCrop([224, 224], num_rand_calls=16).cuda(),
    BF.BatchRandomResizedCrop([224, 224], num_rand_calls=-1).cuda(),
]

runs = 100

# for t in batch_torchvision_transforms:
# 	results = torch.zeros(runs)
# 	for i in range(runs):
# 		torch.cuda.synchronize()
# 		start = time.process_time()
# 		t(batch_input)
# 		torch.cuda.synchronize()
# 		end = time.process_time()
# 		results[i] = end - start
# 	results *= 1000
# 	print(f"all CPU {type(t)} in {results.mean()} +/- {torch.std(results)} ms")

# for t in batch_torchvision_transforms:
# 	results = torch.zeros(runs)
# 	for i in range(runs):
# 		torch.cuda.synchronize()
# 		start = time.process_time()
# 		t(batch_input_cuda)
# 		torch.cuda.synchronize()
# 		end = time.process_time()
# 		results[i] = end - start
# 	results *= 1000
# 	print(f"all CUDA {type(t)} in {results.mean()} +/- {torch.std(results)} ms")

for t in batch_torchvision_transforms:
    results = torch.zeros(runs)
    for i in range(runs):
        torch.cuda.synchronize()
        start = time.process_time()
        torch.stack([t(img) for img in batch_input])
        torch.cuda.synchronize()
        end = time.process_time()
        results[i] = end - start
    results *= 1000
    print(f"each CPU {type(t)} in {results.mean()} +/- {torch.std(results)} ms")

for t in batch_torchvision_transforms:
    results = torch.zeros(runs)
    for i in range(runs):
        torch.cuda.synchronize()
        start = time.process_time()
        torch.stack([t(img) for img in batch_input_cuda])
        torch.cuda.synchronize()
        end = time.process_time()
        results[i] = end - start
    results *= 1000
    print(f"each CUDA {type(t)} in {results.mean()} +/- {torch.std(results)} ms")


for t in cpu_batch_transforms:
    results = torch.zeros(runs)
    for i in range(runs):
        torch.cuda.synchronize()
        start = time.process_time()
        t(batch_input)
        torch.cuda.synchronize()
        end = time.process_time()
        results[i] = end - start
    results *= 1000
    print(f"CPU {type(t)} in {results.mean()} +/- {torch.std(results)} ms")


for t in cuda_batch_transforms:
    results = torch.zeros(runs)
    for i in range(runs):
        torch.cuda.synchronize()
        start = time.process_time()
        t(batch_input_cuda)
        torch.cuda.synchronize()
        end = time.process_time()
        results[i] = end - start
    results *= 1000
    print(f"CUDA {type(t)} in {results.mean()} +/- {torch.std(results)} ms")
