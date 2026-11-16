# Experiment configurations

Each task has an `experiment.yaml` describing the evaluation and one YAML per
method. Edit these files to change the controller or evaluation settings.

| Task | Demonstrations | Evaluation |
|---|---|---|
| Cartpole | 200 × 100 MPC samples | 10 starts, 300 steps at 25 Hz |
| Rocket Landing | 58 MPC episodes, 17,705 samples | 10 starts, at most 1,000 steps at 60 Hz |
| Reacher | 200 × 200 MPC samples | 10 starts, at most 500 steps at 50 Hz; stop at reference five |
| Push-T | 206 human demonstrations | 50 starts, at most 300 steps at 10 Hz |

All methods share a task's demonstration bank and evaluation starts. No measurement
noise is added during evaluation. Starts and Reacher references are specified in
`data/<task>/test_starts.json`.

The `*_upstream.yaml` files use parameters from the Select-DPC benchmarks. BD-KL
and Cartpole BD-sample use untuned settings. Push-T retains the scalar parameters
selected on eight validation starts. BD-χ² now uses block-relative pusher
coordinates and action targets; Select-DPC continues to use world coordinates.
The updated representation was evaluated on the same 50 starts as the earlier
version, rather than on a new held-out set.

Reacher success requires reaching all five references without a height-limit
violation. Cost statistics include successful controller trials only, normalized
by each start's saved expert cost.

Isomap fits are cached in `.cache/isomap/` and recomputed when absent. Use a new
output directory after changing a configuration.

For the original configurations and paper results, use the
[l4dc branch](https://github.com/RoboticExplorationLab/behavioral-denoising/tree/l4dc).
