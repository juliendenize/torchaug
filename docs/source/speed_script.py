from __future__ import annotations

import time
from typing import Callable

import tabulate
import torch
from torchvision import transforms as T

from torchaug import batch_transforms as BF
from torchaug import transforms as F

torch.set_printoptions(precision=3)

import argparse


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
        "--batch-size",
        required=False,
        help="Batch size for batch input.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--device", required=False, help="Device for input.", type=str, default="cpu"
    )

    args = parser.parse_args()

    n_runs_single: int = args.n_runs_single
    n_runs_batch: int = args.n_runs_batch
    run_single: bool = args.run_single
    run_batch: bool = args.run_batch
    shape: list[int] = args.shape
    batch_size: int = args.batch_size
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

        input = torch.randn(shape, device=device)

        rows = []
        print("Testing mono transforms")
        for name, torchvision_transform, torchaug_transform in mono_transforms:
            print(name)

            results_torchvision = time_transform(
                n_runs_single, input, torchvision_transform
            )
            mean_torchvision = float(results_torchvision.mean())
            std_torchvision = float(torch.std(results_torchvision))

            format_torchvision = f"{mean_torchvision:.2f} +/- {std_torchvision:.2f}"

            results_torchaug = time_transform(n_runs_single, input, torchaug_transform)
            mean_torchaug = float(results_torchaug.mean())
            std_torchaug = float(torch.std(results_torchaug))

            format_torchaug = f"{mean_torchaug:.2f} +/- {std_torchaug:.2f}"

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
            f"Transform (input {device}: {input.shape})",
            "Torchvision (ms)",
            "Torchaug (ms)",
        ]
        print("\n\n")
        print(tabulate.tabulate(rows, header, tablefmt="github"))
        print("\n\n")

        header = ["Transform", "Torchvision", "Torchaug"]

    if run_batch:
        batch_transforms = [
            (
                "BatchRandomColorJitter",
                "1",
                None,
                BF.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 1, inplace=True).to(
                    device=device
                ),
            ),
            (
                "BatchRandomColorJitter",
                "8",
                None,
                BF.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, 8, inplace=True).to(
                    device=device
                ),
            ),
            (
                "BatchRandomColorJitter",
                "-1",
                T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.1)], 0.5),
                BF.BatchRandomColorJitter(0.5, 0.5, 0.5, 0.1, 0.5, -1, inplace=True).to(
                    device=device
                ),
            ),
            (
                "BatchRandomGaussianBlur",
                "",
                T.RandomApply([T.GaussianBlur([23, 23], [0.1, 2.0])], 0.5),
                BF.BatchRandomGaussianBlur([23, 23], [0.1, 2], 0.5, inplace=True).to(
                    device=device
                ),
            ),
            (
                "BatchRandomGrayScale",
                "",
                T.RandomApply([T.Grayscale(num_output_channels=3)], 0.5),
                BF.BatchRandomGrayScale(0.5, inplace=True).to(device=device),
            ),
            (
                "BatchRandomHorizontalFlip",
                "",
                T.RandomHorizontalFlip(0.5),
                BF.BatchRandomHorizontalFlip(0.5, inplace=True).to(device=device),
            ),
            (
                "BatchRandomResizedCrop",
                "1",
                None,
                BF.BatchRandomResizedCrop([224, 224], num_rand_calls=1).to(
                    device=device
                ),
            ),
            (
                "BatchRandomResizedCrop",
                "8",
                None,
                BF.BatchRandomResizedCrop([224, 224], num_rand_calls=8).to(
                    device=device
                ),
            ),
            (
                "BatchRandomResizedCrop",
                "16",
                None,
                BF.BatchRandomResizedCrop([224, 224], num_rand_calls=16).to(
                    device=device
                ),
            ),
            (
                "BatchRandomResizedCrop",
                "-1",
                T.RandomResizedCrop([224, 224], antialias=True),
                BF.BatchRandomResizedCrop([224, 224], num_rand_calls=-1).to(
                    device=device
                ),
            ),
            (
                "BatchRandomSolarize",
                "",
                T.RandomSolarize(0.5, 0.5),
                BF.BatchRandomSolarize(0.5, 0.5, inplace=True).to(device=device),
            ),
        ]

        input = torch.randn(batch_size, *shape, device=device)

        print("Testing batch transforms")
        rows = []
        for (
            name,
            rand_calls,
            torchvision_transform,
            torchaug_transform,
        ) in batch_transforms:
            print(name, rand_calls)

            do_tv = torchvision_transform is not None
            if do_tv:
                results_torchvision = time_batch_transform_tv(
                    n_runs_batch, input, torchvision_transform
                )
                mean_torchvision = float(results_torchvision.mean())
                std_torchvision = float(torch.std(results_torchvision))

            format_torchvision = (
                f"{mean_torchvision:.2f} +/- {std_torchvision:.2f}" if do_tv else ""
            )

            results_torchaug = time_transform(n_runs_batch, input, torchaug_transform)
            mean_torchaug = float(results_torchaug.mean())
            std_torchaug = float(torch.std(results_torchaug))

            format_torchaug = f"{mean_torchaug:.2f} +/- {std_torchaug:.2f}"

            if do_tv and mean_torchvision < mean_torchaug:
                format_torchvision = "**" + format_torchvision + "**"
            elif do_tv and mean_torchvision > mean_torchaug:
                format_torchaug = "**" + format_torchaug + "**"

            rows.append(
                [
                    name,
                    rand_calls,
                    format_torchvision,
                    format_torchaug,
                ]
            )
        header = [
            f"Transform (input {device}: {input.shape})",
            "num_rand_calls",
            "Torchvision (ms)",
            "Torchaug (ms)",
        ]

        print("\n\n")
        print(tabulate.tabulate(rows, header, tablefmt="github"))
        print("\n\n")
