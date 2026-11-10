"""Fast model/data smoke test (no network, no full dataset)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pde_burgers.data import BurgersPairs, split_samples
from pde_burgers.models import FNO1d, UNet1d


def main() -> None:
    torch.manual_seed(0)
    n, t, x = 8, 6, 1024
    u = np.random.default_rng(0).normal(size=(n, t, x)).astype(np.float32)
    train, val, test = split_samples(n, val_frac=0.2, test_frac=0.2, seed=0)
    print("split", train, val, test)

    ds = BurgersPairs(u, train, mean=float(u[train].mean()), std=float(u[train].std()))
    xb, yb = ds[0]
    print("pair shapes", xb.shape, yb.shape)

    fno = FNO1d(width=16, modes=8, n_layers=2)
    unet = UNet1d(base=8)
    out_fno = fno(xb[None])
    with torch.no_grad():
        out_unet = unet(xb[None])
    print("fno", out_fno.shape, "unet", out_unet.shape)
    print("fno params", sum(p.numel() for p in fno.parameters()))
    print("unet params", sum(p.numel() for p in unet.parameters()))

    loss = torch.nn.functional.mse_loss(out_fno, yb[None])
    loss.backward()
    print("backward ok, loss", loss.item())
    print("smoke test finished")


if __name__ == "__main__":
    main()
