"""Download a small local subset of PDEBench 1D Burgers via HTTP Range.

The full file is ~7.7 GB.  For the first baseline we only need a few hundred
trajectories.

    python scripts/download_subset.py --n-samples 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pde_burgers.remote import fetch_burgers_subset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--file-id", type=int, default=281363, help="PDEBench file id for Nu=0.01")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--out", type=str, default="data/burgers_nu0.01_200.npz")
    p.add_argument("--timeout", type=float, default=300.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = fetch_burgers_subset(
        file_id=args.file_id,
        n_samples=args.n_samples,
        nu=args.nu,
        out_path=args.out,
        timeout=args.timeout,
        verbose=True,
    )
    print("done:", result["path"])


if __name__ == "__main__":
    main()
