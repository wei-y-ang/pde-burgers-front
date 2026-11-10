# Dataset selection record

## Why not 1D ReactionDiffusion

Before choosing Burgers, we inspected PDEBench 1D ReactionDiffusion trajectories
for six parameter combinations: `Nu=0.5/1.0` and `Rho=1/2/5/10`. In the checked
`u(x,t)` heatmaps, the field relaxed to a spatial constant by the final time.
These configurations therefore did not provide the persistent propagating front
needed for this project, so they were not used for model training.

## Why 1D Burgers

The PDEBench 1D Burgers heatmaps contain diagonal bands corresponding to
propagating viscous shocks/fronts. This makes the dataset suitable for studying
long-horizon rollout, front tracking, and generalization across viscosity.

The experiments use `1D_Burgers_Sols_Nu0.01.hdf5` from PDEBench
([DOI: 10.18419/darus-2986](https://doi.org/10.18419/darus-2986)). The training
subset contains the first 200 trajectories and is obtained with HTTP Range
requests rather than downloading the full 7.7 GB file. See
[`scripts/download_subset.py`](../scripts/download_subset.py) for the extraction
code and the project README for the exact commands.
