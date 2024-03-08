from __future__ import annotations

import argparse
import time
from typing import Callable

import tabulate
import torch

from torchaug import transforms as F
from torchvision.transforms import v2 as T

torch.set_printoptions(precision=3)


def synchro_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    return


def time_transform(n_runs: int, tensor: torch.Tensor, transform: Callable):
    results = torch.zeros(n_runs)
    for i in range(n_runs):
        synchro_if_cuda(tensor.device)
        start = time.process_time()

        transform(tensor)

        synchro_if_cuda(tensor.device)
        end = time.process_time()

        results[i] = end - start
    results *= 1000

    return results


def time_batch_transform_tv(n_runs: int, tensor: torch.Tensor, transform: Callable):
    results = torch.zeros(n_runs)
    for i in range(n_runs):
        synchro_if_cuda(tensor.device)
        start = time.process_time()

        torch.stack([transform(img) for img in tensor])

        synchro_if_cuda(tensor.device)
        end = time.process_time()

        results[i] = end - start
    results *= 1000

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speed comparison between Torchvision and Torchaug.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-runs-single",
        required=False,
        type=int,
        default=1000,
        help="Number of runs to compute metrics on single transforms.",
    )
    parser.add_argument(
        "--n-runs-batch",
        required=False,
        type=int,
        default=100,
        help="Number of runs to compute metrics on batch transforms.",
    )
    parser.add_argument(
        "--run-single",
        required=False,
        help="Whether to run single transforms.",
        action="store_true",
    )
    parser.add_argument(
        "--run-batch",
        required=False,
        help="Whether to run batch transforms.",
        action="store_true",
    )
    parser.add_argument(
        "--shape",
        required=False,
        help="Shape of the input. Batch not included.",
        type=int,
        nargs="+",
        default=[3, 224, 224],
    )
    parser.add_argument(
        "--batch-sizes",
        required=False,
        help="The batch sizes for batch input.",
        type=int,
        nargs="+",
        default=[8, 16, 64, 128],
    )
    parser.add_argument(
        "--device", required=False, help="Device for input.", type=str, default="cpu"
    )
    parser.add_argument(
        "--dtype", required=False, help="Device for input.", type=str, default="int"
    )

    args = parser.parse_args()

    n_runs_single: int = args.n_runs_single
    n_runs_batch: int = args.n_runs_batch
    run_single: bool = args.run_single
    run_batch: bool = args.run_batch
    shape: list[int] = args.shape
    batch_sizes: list[int] = sorted(args.batch_sizes, reverse=True)
    device = args.device

    print(args)

    if run_single:
        mono_transforms = [
            (
                "Normalize",
                T.Normalize([225], [225]),
                F.Normalize([225], [225]).to(device=device),
            ),
            (
                "RandomColorJitter",
                T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.1)], 0.5),
                F.RandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5).to(device=device),
            ),
            (
                "RandomGaussianBlur",
                T.RandomApply([T.GaussianBlur([23, 23], [0.1, 2.0])], 0.5),
                F.RandomGaussianBlur([23, 23], [0.1, 2.0], 0.5).to(device=device),
            ),
            (
                "RandomSolarize",
                T.RandomSolarize(0.5, 0.5),
                F.RandomSolarize(0.5, 0.5).to(device=device),
            ),
        ]

        if args.dtype == "int":
            input = torch.randint(0, 256, shape, device=device, dtype=torch.uint8)
        elif args.dtype == "float":
            input = torch.rand(shape, device=device, dtype=torch.float32)

        rows = []
        print("Testing mono transforms")
        for name, torchvision_transform, torchaug_transform in mono_transforms:
            print(name)

            results_torchvision = time_transform(
                n_runs_single, input, torchvision_transform
            )
            mean_torchvision = float(results_torchvision.mean())
            std_torchvision = float(torch.std(results_torchvision))

            format_torchvision = f"{mean_torchvision:.2f}  ± {std_torchvision:.2f}"

            results_torchaug = time_transform(n_runs_single, input, torchaug_transform)
            mean_torchaug = float(results_torchaug.mean())
            std_torchaug = float(torch.std(results_torchaug))

            format_torchaug = f"{mean_torchaug:.2f}  ± {std_torchaug:.2f}"

            if mean_torchvision < mean_torchaug:
                format_torchvision = "**" + format_torchvision + "**"
            elif mean_torchvision > mean_torchaug:
                format_torchaug = "**" + format_torchaug + "**"

            rows.append(
                [
                    name,
                    format_torchvision,
                    format_torchaug,
                ]
            )

        header = [
            "Transform",
            "Torchvision",
            "Torchaug",
        ]
        print("\n\n")
        print(f"Input shape {shape} on {device}.")
        print(tabulate.tabulate(rows, header, tablefmt="github"))
        print("\n\n")

        header = ["Transform", "Torchvision", "Torchaug"]

    if run_batch:
        batch_transforms = [
            (
                "RandomColorJitter",
                "1",
                None,
                F.RandomColorJitter(
                    0.5,
                    0.5,
                    0.5,
                    0.1,
                    0.5,
                    num_chunks=1,
                    batch_inplace=True,
                    batch_transform=True,
                ).to(device=device),
            ),
            (
                "RandomColorJitter",
                "8",
                None,
                F.RandomColorJitter(
                    0.5,
                    0.5,
                    0.5,
                    0.1,
                    0.5,
                    num_chunks=18,
                    batch_inplace=True,
                    batch_transform=True,
                ).to(device=device),
            ),
            (
                "RandomColorJitter",
                "-1",
                T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.1)], 0.5),
                F.RandomColorJitter(
                    0.5,
                    0.5,
                    0.5,
                    0.1,
                    0.5,
                    num_chunks=-1,
                    batch_inplace=True,
                    batch_transform=True,
                ).to(device=device),
            ),
            (
                "RandomGaussianBlur",
                "",
                T.RandomApply([T.GaussianBlur([23, 23], [0.1, 2.0])], 0.5),
                F.RandomGaussianBlur(
                    [23, 23], [0.1, 2], 0.5, batch_inplace=True, batch_transform=True
                ).to(device=device),
            ),
            (
                "RandomGrayscale",
                "",
                T.RandomApply([T.Grayscale(num_output_channels=3)], 0.5),
                F.RandomGrayscale(0.5, batch_inplace=True, batch_transform=True).to(
                    device=device
                ),
            ),
            (
                "RandomHorizontalFlip",
                "",
                T.RandomHorizontalFlip(0.5),
                F.RandomHorizontalFlip(
                    0.5, batch_inplace=True, batch_transform=True
                ).to(device=device),
            ),
            (
                "RandomResizedCrop",
                "1",
                None,
                F.RandomResizedCrop([224, 224], num_chunks=1, batch_transform=True).to(
                    device=device
                ),
            ),
            (
                "RandomResizedCrop",
                "8",
                None,
                F.RandomResizedCrop([224, 224], num_chunks=8, batch_transform=True).to(
                    device=device
                ),
            ),
            (
                "RandomResizedCrop",
                "16",
                None,
                F.RandomResizedCrop([224, 224], num_chunks=16, batch_transform=True).to(
                    device=device
                ),
            ),
            (
                "RandomResizedCrop",
                "-1",
                T.RandomResizedCrop([224, 224], antialias=True),
                F.RandomResizedCrop([224, 224], num_chunks=-1, batch_transform=True).to(
                    device=device
                ),
            ),
            (
                "RandomSolarize",
                "",
                T.RandomSolarize(0.5, 0.5),
                F.RandomSolarize(0.5, 0.5, batch_inplace=True, batch_transform=True).to(
                    device=device
                ),
            ),
        ]

        print("Testing batch transforms")
        rows = [["**Batch size**", ""]]
        for batch_size in batch_sizes:
            rows[0].extend([f"**{batch_size}**"] * 2)

        for (
            name,
            rand_calls,
            torchvision_transform,
            torchaug_transform,
        ) in batch_transforms:
            print(name, rand_calls)
            row = [name, rand_calls]

            for batch_size in batch_sizes:
                print("Batch size", batch_size)

                if args.dtype == "int":
                    input = torch.randint(
                        0, 256, (batch_size, *shape), device=device, dtype=torch.uint8
                    )
                elif args.dtype == "float":
                    input = torch.rand(
                        (batch_size, *shape), device=device, dtype=torch.float32
                    )

                do_tv = torchvision_transform is not None
                if do_tv:
                    results_torchvision = time_batch_transform_tv(
                        n_runs_batch, input, torchvision_transform
                    )
                    mean_torchvision = float(results_torchvision.mean())
                    std_torchvision = float(torch.std(results_torchvision))

                format_torchvision = (
                    f"{mean_torchvision:.2f}  ± {std_torchvision:.2f}" if do_tv else ""
                )

                results_torchaug = time_transform(
                    n_runs_batch, input, torchaug_transform
                )
                mean_torchaug = float(results_torchaug.mean())
                std_torchaug = float(torch.std(results_torchaug))

                format_torchaug = f"{mean_torchaug:.2f}  ± {std_torchaug:.2f}"

                if do_tv and mean_torchvision < mean_torchaug:
                    format_torchvision = "**" + format_torchvision + "**"
                elif do_tv and mean_torchvision > mean_torchaug:
                    format_torchaug = "**" + format_torchaug + "**"

                row.extend(
                    [
                        format_torchvision,
                        format_torchaug,
                    ]
                )
            rows.append(row)
        header = [
            "Transform",
            "Num chunks",
        ]

        for i in range(len(batch_sizes)):
            header.extend(["Torchvision", "Torchaug"])

        print("\n\n")
        print(f"Input batch sizes {batch_sizes} with shape {shape} on {device}.")
        print(tabulate.tabulate(rows, header, tablefmt="github"))
        print("\n\n")
