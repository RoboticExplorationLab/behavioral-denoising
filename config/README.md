# Experiment configurations

Each task has an `experiment.yaml` describing the evaluation and one YAML per
method. Edit these files to change the controller or evaluation settings.

| Task | Demonstrations | Evaluation |
|---|---|---|
| Cartpole | 200 × 100 MPC samples | 10 starts, 300 steps at 25 Hz |
| Rocket Landing | 58 MPC episodes, 17,705 samples | 10 starts, at most 1,000 steps at 60 Hz |
| Reacher | 200 × 200 MPC samples | 10 starts, at most 500 steps at 50 Hz, stopping at reference five |
| Push-T | 206 human demonstrations | 50 starts, at most 300 steps at 10 Hz |

All methods share a task's demonstration bank and evaluation starts. No measurement
noise is added during evaluation. Starts and Reacher references are specified in
`data/<task>/test_starts.json`.

The `*_upstream.yaml` files use parameters from the Select-DPC benchmarks. BD-KL
and Cartpole BD-sample use untuned settings. Push-T retains the scalar parameters
selected on eight validation starts. Both BD-χ² and Select-DPC default to
block-relative pusher coordinates and action targets. Representation comparisons
reuse the same 50 starts as the earlier version.

Reacher success requires reaching all five references without a height-limit
violation. Cost statistics include successful controller trials only, normalized
by each start's saved expert cost.

Isomap fits are cached in `.cache/isomap/` and recomputed when absent. Use a new
output directory after changing a configuration.

## Push-T coordinates

Both BD-χ² and Select-DPC use the same representation setting. The task default
in `pusht/experiment.yaml` is:

```yaml
task:
  representation: block_relative  # or world
```

Both method YAMLs inherit the task default. A top-level `representation: world`
or `representation: block_relative` can override it in either method's YAML,
alongside `id` and `parameters`. An explicit CLI option
`--representation world` or `--representation block_relative` overrides both
the task and method settings for the run. The older `bd_coordinates` setting is
still read for BD-χ² when neither new setting is specified.

`block_relative` expresses the pusher position relative to the block, rotated
into the block's frame. The block's absolute position and cosine/sine orientation
remain in the observation. All action targets in a window, including past
inputs, use the block pose at the last history observation. The plan is decoded
once, so subsequent block motion does not move queued world targets.

The shared conversion leaves demonstration files and simulation in world
coordinates. Each controller retains its normalization convention. Select-DPC
transforms its input cost and workspace bounds to preserve their physical
meaning. Records include the effective representation, normalization, policy
features, and planning-frame anchor alongside the physical trajectories.

Controllers accept physical observations through `Policy.act()`.
`PushTTask.policy_observation()` exposes transformed features for inspection.
Do not transform observations before passing them to `act()`.

For the original configurations and paper results, use the
[l4dc branch](https://github.com/RoboticExplorationLab/behavioral-denoising/tree/l4dc).
