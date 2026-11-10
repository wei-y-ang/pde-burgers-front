"""Plot OOD relative-L2 and front-speed errors."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    df = pd.read_csv("results/ood/summary.csv")
    # Use only ID and Nu=0.001/0.1 evaluations.
    keep = {
        "baseline_id": "ID (nu=0.01)",
        "pde_id": "ID (nu=0.01)",
        "baseline_nu0.001": "OOD (nu=0.001)",
        "pde_nu0.001": "OOD (nu=0.001)",
        "baseline_nu0.1": "OOD (nu=0.1)",
        "pde_nu0.1": "OOD (nu=0.1)",
    }
    df = df[df["eval"].isin(keep)].copy()
    df["label"] = df["eval"].map(keep)
    df["model"] = np.where(df["run"].str.contains("pde", case=False), "FNO + PDE", "FNO baseline")

    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in [
        (axes[0], "rel_l2_mean", "rollout mean relative L2"),
        (axes[1], "front_speed_mae", "front speed MAE"),
    ]:
        pivot = df.pivot_table(index="label", columns="model", values=col, aggfunc="mean")
        pivot = pivot.loc[["ID (nu=0.01)", "OOD (nu=0.001)", "OOD (nu=0.1)"]]
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "ood_metrics.png", dpi=160)
    plt.close(fig)
    print("wrote", out / "ood_metrics.png")


if __name__ == "__main__":
    main()
