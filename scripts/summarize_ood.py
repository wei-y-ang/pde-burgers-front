"""Collect OOD/in-distribution evaluation JSON files into one table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path("results/ood")
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "eval": path.parent.name,
                "run": Path(m["run"]).name,
                "nu": m["nu"],
                "rel_l2_mean": m["rel_l2"]["mean"],
                "rel_l2_10": m["rel_l2"].get("at_10"),
                "rel_l2_50": m["rel_l2"].get("at_50"),
                "rel_l2_100": m["rel_l2"].get("at_100"),
                "front_pos_mae_aligned": m["front"]["position_mae_aligned"]["mean"],
                "front_speed_mae": m["front"]["speed_abs_error"]["mean"],
            }
        )
    df = pd.DataFrame(rows).sort_values(["run", "nu"])
    df.to_csv(root / "summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"wrote {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
