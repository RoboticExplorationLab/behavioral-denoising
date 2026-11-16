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

For Push-T, `src/tasks/pusht_coordinates.py` provides shared world and
block-relative representations for BD-χ² and Select-DPC. Both methods default to
block-relative coordinates. Relative observations
express the pusher in the block's frame while retaining the block's absolute
pose. Every action in a planning window uses the final history block pose.
Decoded world targets remain fixed throughout chunk execution. Each controller
retains its normalization convention. Select-DPC transforms its input cost and
bounds to retain their physical meaning. The simulator and saved physical
trajectories use world coordinates. See the
[coordinate configuration](config/README.md#push-t-coordinates).

`src/benchmarks/runner.py` handles fixed protocols, resumable per-start records,
and aggregation. `reporting.py` computes metrics from saved records. Plotting and
timing tools are under `scripts/` and read user-generated runs.

The [l4dc branch](https://github.com/RoboticExplorationLab/behavioral-denoising/tree/l4dc)
preserves the submission implementation and paper reproduction materials.
