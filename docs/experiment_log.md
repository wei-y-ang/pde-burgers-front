# Experiment log

## 2026-11-03 — data and initial baseline

- **Data**: PDEBench 1D Burgers, `Nu=0.01`, first 200 trajectories extracted by HTTP Range.
- **Shape**: `(200, 201, 1024)`; local file 139 MB.
- **Split**: 160 train / 20 val / 20 test, by trajectory.
- **Short baseline** (8 epochs):
  - FNO: one-step rel L2 `0.0090`; 50-step rollout mean `0.189`.
  - U-Net: one-step rel L2 `0.0265`; 50-step rollout mean `0.673`.

## 2026-11-04 — 40-epoch, 3-seed FNO

- **Training**: 40 epochs, batch 32, width 32, modes 16, 4 blocks, AdamW `1e-3`.
- **FNO per seed**:

| seed | one-step rel L2 | rollout mean | 10 steps | 50 steps | 100 steps |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.001805 | 0.06797 | 0.02136 | 0.07287 | 0.10789 |
| 1 | 0.001864 | 0.08242 | 0.02829 | 0.07592 | 0.17426 |
| 2 | 0.001860 | 0.08130 | 0.03779 | 0.08945 | 0.11373 |
| **mean ± std** | **0.001843 ± 0.000033** | **0.07723 ± 0.00804** | **0.02915 ± 0.00825** | **0.07941 ± 0.00882** | **0.13196 ± 0.03675** |

- **U-Net, 1 seed, 40 epochs**:
  - one-step rel L2 `0.005786`
  - 100-step rollout mean `0.4327`
  - 10/50/100-step errors: `0.0791 / 0.5048 / 0.7293`
- **Observation**: longer training clearly helps FNO; the remaining error is
  concentrated at sharp shocks.

## 2026-11-04 — PDE residual fine-tuning

- Start from each `fno40_seed{0,1,2}` checkpoint.
- Fine-tune 10 epochs, `lr=1e-4`, batch 32.
- PDE residual: `R = u_t + u u_x - nu u_xx`; normalized as `R * dt / u_std`.
- Results (3 seeds):

| Variant | one-step rel L2 | rollout mean |
|---|---:|---:|
| data-only FT | 0.00171 ± 0.00021 | 0.0650 ± 0.0151 |
| PDE FT, λ=1e-3 | 0.00153 ± 0.00010 | 0.0607 ± 0.0150 |

- **Observation**: PDE FT improves the mean slightly, but the seed error bars
  overlap. λ=1e-2 was worse on seed 0. The current PDE loss is a first-order
  time-difference penalty, not yet a strong rollout constraint.

## 2026-11-04 — sequence-level PDE loss

- Unroll `H=5` steps during training; two data-loss modes:
  - `all`: supervise all 5 predicted steps;
  - `first`: supervise only the first step, but still apply PDE residual to the
    whole unrolled sequence.
- Seed 0, 10 epochs, lr `1e-4`, init from `fno40_seed0`.

| Variant | one-step | rollout mean |
|---|---:|---:|
| seq data, all steps | 0.001784 | 0.0708 |
| seq data, first step | 0.001529 | **0.0504** |
| seq + PDE, all steps, λ=1e-3 | 0.002200 | 0.1014 |
| seq + PDE, first step, λ=1e-3 | 0.001775 | 0.0905 |
| seq + PDE, first step, λ=1e-4 | 0.001705 | 0.0687 |

- **Observation**: sequence-level PDE loss with unrolled predictions
  destabilizes training; the best sequence variant is data-only first-step
  supervision, but this is only one seed.

## 2026-11-04 — H=2 with detached PDE tail

- `seq-len=2`, `seq-data-loss=first`, `detach-seq-pde` (the second predicted
  step is detached when computing the PDE residual).
- Seed 0, 10 epochs, lr `1e-4`, init from `fno40_seed0`.

| Variant | one-step | rollout mean |
|---|---:|---:|
| H=2 detach, data-only | 0.002246 | 0.1010 |
| H=2 detach, PDE λ=1e-3 | 0.002266 | 0.1023 |

- **Observation**: detaching the second step did not rescue sequence training;
  both variants are much worse than one-step PDE fine-tuning (`0.0607`).
- **Conclusion**: do not spend more time on this form. Move to OOD and
  front-position metrics; sequence PDE can be listed as a negative result.

## 2026-11-04 — OOD and front metrics

- Extracted 100 trajectories each for `Nu=0.001` and `Nu=0.1`.
- Evaluated 20 trajectories, 100 rollout steps.
- Front tracking: `argmax |u_x|`, 64-point search window, periodic unwrap.
- Aligned position error removes a constant circular offset caused by locking
  onto a different shock.

| Model | Nu | rel L2 | aligned pos MAE | speed MAE |
|---|---:|---:|---:|---:|
| FNO baseline | 0.01 | 0.0765 | 0.0081 | 0.0329 |
| FNO baseline | 0.001 | 0.1591 | 0.0068 | 0.0267 |
| FNO baseline | 0.1 | 0.7897 | 0.0533 | 0.1620 |
| FNO + PDE | 0.01 | 0.0521 | 0.0052 | 0.0158 |
| FNO + PDE | 0.001 | 0.1474 | 0.0048 | 0.0180 |
| FNO + PDE | 0.1 | 0.7901 | 0.0524 | 0.1537 |

- **Observation**: OOD failure is asymmetric; smoother `Nu=0.1` is much harder
  than sharper `Nu=0.001`. PDE fine-tuning helps ID and `Nu=0.001`, but not
  `Nu=0.1`.
- **Next**: multi-`Nu` training or parameter-conditioned operator.

## 2026-11-04 — multi-Nu conditional FNO (negative)

- Train `Nu=0.001` + `Nu=0.01`, hold out `Nu=0.1`.
- Condition: constant channel with normalized `log Nu`.
- 25 epochs, 5000 pairs, width 32, modes 16, 4 blocks.

| Eval | one-step | rollout mean |
|---|---:|---:|
| joint test (`0.001/0.01`) | 0.00908 | 0.3907 |
| held-out `Nu=0.1` | — | 8.60 |

- **Observation**: constant-channel conditioning underfits and fails OOD
  catastrophically. It is not usable as a final model.
- **Next**: FiLM conditioning, per-`Nu` normalization, or MoE.
