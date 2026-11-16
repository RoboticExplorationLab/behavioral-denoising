# Implementation map

| Method | Implementation |
|---|---|
| BD-mean / BD-sample | `src/policies/behavioral_denoising.py` |
| BD-KL | `src/policies/bd_kl.py` |
| BD-mean-χ² | `src/policies/bd_mean_chi2.py` |
| BD-χ² | `src/policies/bd_chi2.py` |
| Select-DPC | `src/policies/select_dpc.py` and chunk execution for Push-T in `select_dpc_chunks.py` |
| Windowed DeePC / DeePC | `src/policies/windowed_deepc.py` / `deepc.py` |

Behavior windows are constructed within episodes. Histories and action
chunks are standardized using demonstration statistics. BD methods condition on
the configured nearest histories and use normalized Gaussian weights. BD-sample
uses the finite sigma-Euler sampler, and BD-KL projects each denoising-step posterior
and reports infeasibility when its history constraints cannot be satisfied.

BD-mean-χ² uses a weighted affine fit. BD-χ² initializes a robust affine fit from
a diffusion sample, carries that fit between decisions, and reselects a mode when
its retained mass is too small. Its leverage limit restricts extrapolation.
Covariance solves add the configured small ridge (1e-6). Nearest-window counts and diffusion schedules are set in the method configurations.

DPC objectives and constraints are in `src/optimization/deepc_qp.py`. Controller
parameters, including regularization, horizons, neighborhood sizes, and solver
limits, are explicit in `config/`. OSQP uses `adaptive_rho_interval=25`.
Applied inputs are fed back to input-conditioned policies after actuator clipping.
All Rocket policies use a shared touchdown controller after first leg contact.

MPC experts are custom implementations: Cartpole and Reacher use CasADi/IPOPT and
Rocket uses CVXPY/OSQP. Push-T uses human demonstrations from Diffusion Policy.

For Push-T BD-χ², `src/policies/pusht_coordinates.py` expresses the pusher relative
to the block while retaining the block's absolute position and orientation.
Each demonstration action chunk uses the block frame at the final history
observation. The predicted chunk is decoded once using the current block pose;
its world-space targets then remain fixed throughout execution. Standardization
uses the transformed demonstration windows. Select-DPC retains world coordinates.

`src/benchmarks/runner.py` handles fixed protocols, resumable per-start records,
and aggregation. `reporting.py` computes metrics from saved records. Plotting and
timing tools are under `scripts/` and read user-generated runs.

The [l4dc branch](https://github.com/RoboticExplorationLab/behavioral-denoising/tree/l4dc)
preserves the submission implementation and paper reproduction materials.
