# pde-burgers-front

![Python >=3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Neural operators for PDEBench 1D Burgers front prediction.**

> On 200 public Burgers trajectories, FNO1D reaches a mean 100-step rollout
> relative L2 error of 0.077, compared with 0.433 for U-Net1D. One-step PDE
> fine-tuning reduces the FNO error to 0.061. The project also reports
> out-of-distribution limits and negative results rather than hiding them.

This repository learns a PDE surrogate from **public** PDEBench data, with a
clear path from a supervised baseline to physics-informed fine-tuning. The
midterm report is complete, and the main FNO baseline uses three random seeds.

- [Chinese report (Markdown)](docs/report.md) · [PDF](docs/report.pdf) · [LaTeX source](docs/report.tex)
- [Dataset selection record](docs/data_check.md)
- [Experiment log](docs/experiment_log.md)
- [Reproducible baseline checkpoint](checkpoints/fno40_seed0/)

The data contains propagating viscous **shock fronts**:

\[
u_t + u u_x = \nu u_{xx}
\]

Strictly speaking, these are traveling shock/front solutions rather than a
single permanent periodic traveling wave.  The project uses the phrase
"front" deliberately.

---

## 1. Why this dataset

We checked PDEBench 1D ReactionDiffusion first.  Across six parameter
combinations (`Nu=0.5/1.0`, `Rho=1/2/5/10`), the field relaxes to a spatial
constant by the final time; there is no persistent front.  See
[`docs/data_check.md`](docs/data_check.md).

PDEBench **1D Burgers** is different: the `u(x,t)` heatmaps show diagonal
bands, i.e. propagating shocks/fronts.  That makes it suitable for a
"traveling front" project.

---

## 2. Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

### Download only 200 trajectories

The full PDEBench Burgers file is ~7.7 GB.  This script uses HTTP Range
requests to read only the first `N` trajectories and saves a small local
`.npz`:

```bash
python scripts/download_subset.py \
  --file-id 281363 \
  --n-samples 200 \
  --nu 0.01 \
  --out data/burgers_nu0.01_200.npz
```

The resulting file is roughly 140 MB (20 trajectories are about 14 MB).

### Smoke test

```bash
python scripts/smoke_test.py
python -m unittest discover -s tests -v
```

On Unix-like systems with `make`, `make reproduce` runs the unit tests,
downloads the 200-trajectory subset, trains one FNO baseline, and plots a
rollout. The individual commands below work on Windows without `make`.

### Train FNO1D

```bash
python -m pde_burgers.train \
  --data data/burgers_nu0.01_200.npz \
  --model fno \
  --out results/runs/fno \
  --epochs 40 \
  --batch-size 32 \
  --width 32 \
  --modes 16 \
  --n-layers 4
```

### Train U-Net1D baseline

```bash
python -m pde_burgers.train \
  --data data/burgers_nu0.01_200.npz \
  --model unet \
  --out results/runs/unet \
  --epochs 40 \
  --batch-size 32 \
  --base 16
```

### PDE fine-tuning

Start from a trained checkpoint and add the Burgers residual:

```bash
python -m pde_burgers.train \
  --data data/burgers_nu0.01_200.npz \
  --model fno \
  --out results/runs/fno_pde \
  --epochs 10 \
  --batch-size 32 \
  --width 32 \
  --modes 16 \
  --n-layers 4 \
  --init checkpoints/fno40_seed0/model.pt \
  --pde-weight 1e-3 \
  --lr 1e-4
```

The PDE residual is normalized as `R * dt / u_std` before squaring, so the
PDE loss is comparable to the normalized data loss.

### Sequence-level PDE (experimental)

```bash
python -m pde_burgers.train \
  --data data/burgers_nu0.01_200.npz \
  --model fno \
  --out results/runs/fno_seq_pde \
  --epochs 10 \
  --batch-size 16 \
  --width 32 \
  --modes 16 \
  --n-layers 4 \
  --init checkpoints/fno40_seed0/model.pt \
  --seq-len 5 \
  --seq-data-loss first \
  --pde-weight 1e-4 \
  --lr 1e-4
```

In our seed-0 runs, sequence-level PDE loss was **not** better than one-step
PDE fine-tuning: the unrolled PDE residual amplified prediction errors.  The
best sequence variant was data-only first-step supervision (rollout mean
0.0504), but this is a single-seed result and needs more runs.

We also tried `H=2` with the PDE tail detached (`--detach-seq-pde`).  Both the
data-only control and the PDE variant were around `0.101` rollout mean, much
worse than one-step PDE fine-tuning (`0.0607`).  This line is currently a
negative result; the next step is OOD evaluation and front-position metrics,
not more sequence-PDE tuning.

### Plot rollout

```bash
python scripts/plot_rollout.py \
  --npz results/runs/fno/rollout_example.npz \
  --out-dir results/figures/fno
```

### Aggregate seeds

```bash
python scripts/summarize_seeds.py \
  --run-glob "results/runs/fno40_seed*" \
  --out-dir results/summary_fno40
```

---

## 3. Data

PDEBench 1D Burgers:

- file: `1D/Burgers/Train/1D_Burgers_Sols_Nu0.01.hdf5`
- file id: `281363`
- DOI: [10.18419/darus-2986](https://doi.org/10.18419/darus-2986)
- tensor shape: `(10000, 201, 1024)` = `(sample, time, x)`
- `x ∈ [0, 1]`, `t ∈ [0, 2.01]`
- `u` range: about `[-1.35, 1.41]`

The local subset stores:

```text
u: (N, T, X)
x: (X,)
t: (T,)
nu: scalar
source: string
```

Splits are by **trajectory**, not by random space-time points, to avoid
leakage.

---

## 4. Models

- **FNO1D**: spectral convolution + pointwise residual block.
- **U-Net1D**: strong convolutional baseline.

Both take a normalized profile `u(t)` of shape `(B, 1, X)` and predict
`u(t+1)`.

Current training is one-step teacher forcing.  Rollout is autoregressive and
is reported separately.

---

## 5. Current results

40 epochs, 200 trajectories, CPU.  FNO is run with 3 seeds; U-Net with 1 seed.
Fine-tuning starts from each seed's checkpoint and runs 10 more epochs at
`lr=1e-4`.

| Model | Params | One-step rel L2 | Rollout mean rel L2 (100 steps) |
|---|---:|---:|---:|
| FNO1D (w=32, m=16, 4 blocks) | 72,001 | 0.00184 ± 0.00003 | 0.0772 ± 0.0080 |
| FNO + data-only FT | 72,001 | 0.00171 ± 0.00021 | 0.0650 ± 0.0151 |
| FNO + PDE FT (λ=1e-3) | 72,001 | 0.00153 ± 0.00010 | 0.0607 ± 0.0150 |
| U-Net1D (base=16) | 46,001 | 0.00579 | 0.4327 |

The FNO prediction captures the diagonal front structure but smooths sharp
shocks.  PDE fine-tuning gives a small improvement, but the seed variance is
still large; treat it as promising rather than conclusive.

![FNO rollout across seeds](docs/figures/fno40_rollout_mean_std.png)

![Fine-tuning comparison](docs/figures/finetune_rollout_comparison.png)

### OOD and front metrics

The small seed-0 baseline checkpoint is included in
`checkpoints/fno40_seed0/` together with its normalization statistics and model
configuration. Extract two OOD subsets and evaluate it directly:

```bash
python scripts/download_subset.py --file-id 268190 --n-samples 100 --nu 0.001 \
  --out data/burgers_nu0.001_100.npz
python scripts/download_subset.py --file-id 268185 --n-samples 100 --nu 0.1 \
  --out data/burgers_nu0.1_100.npz

python scripts/evaluate_ood.py --run checkpoints/fno40_seed0 \
  --data data/burgers_nu0.001_100.npz --split all --max-samples 20 \
  --out results/ood/baseline_nu0.001
```

Front metrics track the steepest front (`max |u_x|`) and report both raw and
circularly aligned position error, because Burgers data can contain multiple
shocks. The speed is a linear fit to the tracked front position.

| Model | ν | rollout mean rel L2 | aligned position MAE | speed MAE |
|---|---:|---:|---:|---:|
| FNO baseline | 0.01 (ID) | 0.0765 | 0.0081 | 0.0329 |
| FNO baseline | 0.001 (OOD) | 0.1591 | 0.0068 | 0.0267 |
| FNO baseline | 0.1 (OOD) | 0.7897 | 0.0533 | 0.1620 |
| FNO + PDE | 0.01 (ID) | 0.0521 | 0.0052 | 0.0158 |
| FNO + PDE | 0.001 (OOD) | 0.1474 | 0.0048 | 0.0180 |
| FNO + PDE | 0.1 (OOD) | 0.7901 | 0.0524 | 0.1537 |

The OOD failure is asymmetric: the model degrades much more on the smoother
`ν=0.1` regime than on the sharper `ν=0.001` regime.

![OOD metrics](docs/figures/ood_metrics.png)

### Multi-ν conditional training (negative)

```bash
python -m pde_burgers.train \
  --data data/burgers_nu0.001_100.npz data/burgers_nu0.01_200.npz \
  --conditional --exclude-nu 0.1 \
  --model fno --out results/runs/cond \
  --epochs 25 --batch-size 32 --width 32 --modes 16 --n-layers 4
```

This appends a constant normalized `log ν` channel to the input.  It underfits:
joint-test rollout mean `0.391`, held-out `ν=0.1` relative L2 `8.60`.  The
constant-channel conditioning is not enough; FiLM or per-ν normalization is the
next thing to try.

## 6. Metrics

- one-step MSE (normalized);
- one-step relative \(L_2\);
- rollout relative \(L_2\) at steps 10/50/100;
- PDE residual (finite difference) is implemented in
  [`pde_burgers/losses.py`](pde_burgers/losses.py) and used by
  `--pde-weight` during fine-tuning.

---

## 7. Project structure

```text
pde-burgers-front/
├─ README.md
├─ LICENSE
├─ Makefile
├─ requirements.txt
├─ pyproject.toml
├─ .github/workflows/ci.yml
├─ checkpoints/fno40_seed0/
├─ data/
├─ docs/
│  ├─ report.md / report.tex / report.pdf
│  ├─ data_check.md
│  ├─ experiment_log.md
│  └─ figures/
├─ pde_burgers/
│  ├─ data.py
│  ├─ remote.py
│  ├─ models.py
│  ├─ losses.py
│  ├─ front.py
│  ├─ train.py
│  └─ utils.py
├─ scripts/
│  ├─ download_subset.py
│  ├─ smoke_test.py
│  ├─ plot_rollout.py
│  ├─ evaluate_ood.py
│  ├─ summarize_ood.py
│  └─ plot_ood.py
├─ tests/
└─ results/
```

---

## 8. Roadmap

Completed: FNO/U-Net baselines, rollout evaluation, one-step PDE fine-tuning,
OOD evaluation, front metrics, and the two negative-result studies above.

Next steps:

1. Replace constant-channel conditioning with FiLM or per-`Nu` normalization.
2. Run more seeds before drawing stronger conclusions about PDE fine-tuning.
3. Study the asymmetric failure on smoother `Nu=0.1` data.

---

## 9. Notes

- The full data file is large; always extract a small subset first.
- Do not split by random space-time points; split by trajectory.
- Report Monte Carlo/noise? Not applicable here; but report seed variance.
- The `.deps/` folder is a local fallback for this machine and is git-ignored.
  On a clean machine, install `h5py` from `requirements.txt`.
