"""Reproduce the main fair-comparison and resolution tables.

Existing metrics are reused by default.  Use --overwrite to rerun experiments.
Run from the repository root: python scripts/reproduce_main.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = "data/burgers_nu0.01_200.npz"


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def needs_run(path: str, overwrite: bool) -> bool:
    return overwrite or not (ROOT / path / "metrics.json").exists()


def ensure_data(path: str, file_id: int, n_samples: int, nu: float) -> None:
    if (ROOT / path).exists():
        return
    run(
        "scripts/download_subset.py",
        "--file-id",
        str(file_id),
        "--n-samples",
        str(n_samples),
        "--nu",
        str(nu),
        "--out",
        path,
    )


def train_baseline(model: str, seed: int, overwrite: bool) -> None:
    out = f"results/runs/{model}40_seed{seed}"
    if not needs_run(out, overwrite):
        return
    args = [
        "-m",
        "pde_burgers.train",
        "--data",
        DATA,
        "--model",
        model,
        "--out",
        out,
        "--epochs",
        "40",
        "--batch-size",
        "32",
        "--max-train-pairs",
        "10000",
        "--patience",
        "10",
        "--seed",
        str(seed),
    ]
    if model == "fno":
        args += ["--width", "32", "--modes", "16", "--n-layers", "4"]
    else:
        args += ["--base", "16"]
    run(*args)


def train_finetune(kind: str, seed: int, overwrite: bool) -> None:
    if kind == "data":
        out = "results/runs/finetune_data" + ("" if seed == 0 else f"_seed{seed}")
    elif kind == "pde":
        out = "results/runs/finetune_pde1e-3" + ("" if seed == 0 else f"_seed{seed}")
    else:
        out = "results/runs/seq_first_data" + ("" if seed == 0 else f"_seed{seed}")
    if not needs_run(out, overwrite):
        return
    args = [
        "-m",
        "pde_burgers.train",
        "--data",
        DATA,
        "--model",
        "fno",
        "--out",
        out,
        "--epochs",
        "10",
        "--lr",
        "1e-4",
        "--width",
        "32",
        "--modes",
        "16",
        "--n-layers",
        "4",
        "--init",
        f"results/runs/fno40_seed{seed}/model.pt",
        "--max-train-pairs",
        "6000",
        "--max-val-pairs",
        "1000",
        "--patience",
        "5",
        "--seed",
        str(seed),
    ]
    if kind == "seq":
        args += ["--batch-size", "16", "--seq-len", "5", "--seq-data-loss", "first"]
    else:
        args += ["--batch-size", "32"]
        if kind == "pde":
            args += ["--pde-weight", "1e-3"]
    run(*args)


def evaluate_persistence(overwrite: bool) -> None:
    configs = [
        (DATA, "test", 16, "results/persistence/id_test"),
        ("data/burgers_nu0.001_100.npz", "all", 20, "results/persistence/nu0.001"),
        ("data/burgers_nu0.1_100.npz", "all", 20, "results/persistence/nu0.1"),
    ]
    for data, split, n_samples, out in configs:
        if needs_run(out, overwrite):
            run(
                "scripts/evaluate_persistence.py",
                "--data",
                data,
                "--split",
                split,
                "--max-samples",
                str(n_samples),
                "--out",
                out,
            )


def evaluate_resolutions(overwrite: bool) -> None:
    for model in ["fno", "unet"]:
        for seed in range(3):
            for resolution in [1024, 512, 256]:
                out = f"results/resolution/{model}_seed{seed}_x{resolution}"
                if not needs_run(out, overwrite):
                    continue
                run(
                    "scripts/evaluate_ood.py",
                    "--run",
                    f"results/runs/{model}40_seed{seed}",
                    "--data",
                    DATA,
                    "--split",
                    "test",
                    "--max-samples",
                    "16",
                    "--resolution",
                    str(resolution),
                    "--out",
                    out,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Rerun even when metrics exist.")
    args = parser.parse_args()

    ensure_data(DATA, 281363, 200, 0.01)
    ensure_data("data/burgers_nu0.001_100.npz", 268190, 100, 0.001)
    ensure_data("data/burgers_nu0.1_100.npz", 268185, 100, 0.1)
    run("-m", "unittest", "discover", "-s", "tests", "-v")

    for seed in range(3):
        train_baseline("fno", seed, args.overwrite)
        train_baseline("unet", seed, args.overwrite)
        for kind in ["data", "pde", "seq"]:
            train_finetune(kind, seed, args.overwrite)

    evaluate_persistence(args.overwrite)
    run("scripts/summarize_fair_comparison.py")
    evaluate_resolutions(args.overwrite)
    run("scripts/summarize_resolution.py")

    figures = {
        "results/fair_comparison/fair_comparison.png": "docs/figures/fair_comparison.png",
        "results/fair_comparison/rollout_error_comparison.png": "docs/figures/rollout_error_comparison.png",
        "results/resolution_summary/resolution_comparison.png": "docs/figures/resolution_comparison.png",
    }
    for source, target in figures.items():
        shutil.copyfile(ROOT / source, ROOT / target)
    print("\nMain tables and figures reproduced.", flush=True)


if __name__ == "__main__":
    main()
