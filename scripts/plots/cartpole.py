"""Plot Cartpole trajectories and tracking errors from saved results.

Run from the project root:
    .venv/bin/python scripts/plots/cartpole.py

The first figure shows all ten trials, their pointwise median and central 50%.
The available-trial counts and explanatory text are retained in the CSV/README. No values are imputed after a run
ends. The second figure and optional movie replay one matched initial state.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.plots.common import PAPER_RESULTS, REPOSITORY, paper_font, replay_input_path
from scripts.plots.common import verify_file_hash, verify_rendering_hash

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


METHODS = ("bd_chi2", "select_dpc")
LABELS = {"bd_chi2": "BD-χ²", "select_dpc": "Select-DPC"}
COLORS = {"bd_chi2": "#2ca02c", "select_dpc": "#ff7f0e"}
REFERENCE = np.array([0., 0., 1., 0., 0.])
DT = .04
HORIZON = 12.
ANGLE_LIMIT = 15.
POSITION_LIMIT = 1.
INPUT_HASHES = {}
FONT_FAMILY = paper_font()
POSE_TIMES = (0., .8, 3.6)


@dataclass
class Trial:
    directory: Path
    method: str
    seed: int
    time: np.ndarray
    states: np.ndarray
    result: dict

    @property
    def angle_error(self):
        angle = self.states[:, 1]
        return np.rad2deg(np.abs(np.arctan2(np.sin(angle), np.cos(angle))))

    @property
    def position_error(self):
        return np.abs(self.states[:, 0])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    path = path.resolve()
    value = digest(path)
    previous = INPUT_HASHES.setdefault(str(path), value)
    if previous != value:
        raise ValueError(f"Input changed while plotting: {path}")
    return path


def read_json(path):
    return json.loads(pin(path).read_text())


def load_trial(directory, method):
    metadata = read_json(directory / "metadata.json")
    result = read_json(directory / "result.json")
    archive = pin(directory / "trajectory.npz")
    if metadata["trajectory_sha256"] != digest(archive):
        raise ValueError(f"Trajectory hash does not match its metadata: {directory}")
    with np.load(archive, allow_pickle=False) as saved:
        states = saved["states"].copy()
        time = saved["timestamps"].copy()
        observations = saved["observations"]
        references = saved["references"]
        steps = len(saved["actions"])
        expected_observations = np.column_stack([
            states[:, 0], np.sin(states[:, 1]), np.cos(states[:, 1]), states[:, 2:]])
        if (states.shape != (steps + 1, 4) or time.shape != (steps + 1,)
                or not np.isfinite(states).all()
                or not np.allclose(time, np.arange(steps + 1) * DT, atol=1e-10, rtol=0)
                or not np.allclose(observations, expected_observations, atol=1e-12, rtol=0)
                or not np.array_equal(references, np.broadcast_to(REFERENCE, references.shape))
                or observations.shape != (steps + 1, 5)
                or references.shape != (steps, 5)
                or int(result["steps"]) != steps):
            raise ValueError(f"Unexpected state, reference or time alignment: {directory}")
    return Trial(directory, method, int(result["seed"]), time, states, result)


def load_trials(root):
    trials = {method: [load_trial(path, method) for path in
              sorted((root / "cartpole" / method).glob("seed_*"))] for method in METHODS}
    if any(len(group) != 10 for group in trials.values()):
        raise ValueError("Expected exactly ten saved starts per method")
    if [t.seed for t in trials[METHODS[0]]] != [t.seed for t in trials[METHODS[1]]]:
        raise ValueError("The methods do not share the same ten seeds")
    for first, second in zip(trials[METHODS[0]], trials[METHODS[1]]):
        if not np.array_equal(first.states[0], second.states[0]):
            raise ValueError(f"Initial states differ for seed {first.seed}")
    return trials


def balanced_intervals(trial):
    balanced = ((trial.angle_error <= ANGLE_LIMIT)
                & (trial.position_error <= POSITION_LIMIT)
                & (np.abs(trial.states[:, 3]) <= 1.))
    edges = np.diff(np.r_[False, balanced, False].astype(int))
    return [(float(trial.time[start]), float(trial.time[end - 1]))
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def choose_pair(trials, seed):
    pairs = list(zip(trials[METHODS[0]], trials[METHODS[1]]))
    if seed is None:
        candidates = [(first, second) for first, second in pairs
                      if first.result["success"] and not second.result["success"]
                      and any(end - start >= 2. for start, end in balanced_intervals(second))]
        if not candidates:
            raise ValueError("No failed Select-DPC trial balanced for at least two seconds")
        pair = candidates[0]
    else:
        matches = [pair for pair in pairs if pair[0].seed == seed]
        if not matches:
            raise ValueError(f"Seed {seed} is not in the finalized ten starts")
        pair = matches[0]
        if (not pair[0].result["success"] or pair[1].result["success"]
                or not any(end - start >= 2. for start, end in balanced_intervals(pair[1]))):
            raise ValueError("Choose a BD-success/Select-failure pair with at least two seconds of temporary balance")
    return dict(zip(METHODS, pair))


def summarize(trials):
    times = np.arange(round(HORIZON / DT) + 1) * DT
    summaries = {}
    for method, group in trials.items():
        for metric in ("angle_error", "position_error"):
            values = np.full((len(group), len(times)), np.nan)
            for row, trial in enumerate(group):
                values[row, :len(trial.time)] = getattr(trial, metric)
            count = np.sum(np.isfinite(values), axis=0)
            if np.any(count == 0):
                raise ValueError("No recorded states remain at part of the plot horizon")
            summaries[method, metric] = dict(
                count=count, median=np.nanmedian(values, axis=0),
                q25=np.nanquantile(values, .25, axis=0, method="linear"),
                q75=np.nanquantile(values, .75, axis=0, method="linear"),
                mean=np.nanmean(values, axis=0), sample_sd=np.nanstd(values, axis=0, ddof=1))
    return times, summaries


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dddddd", linewidth=.6)
    axis.set_axisbelow(True)


def save_figure(figure, output, name):
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=240, facecolor="white")
    plt.close(figure)


def method_handles():
    return [Line2D([], [], color=COLORS[method], linewidth=2.8, label=LABELS[method])
            for method in METHODS]


def draw_tracking_axes(axes, trials, times, summaries):
    for axis, metric, label, threshold in zip(
            axes, ("angle_error", "position_error"),
            ("Pole-angle error (deg)", "Cart-position error (m)"), (ANGLE_LIMIT, POSITION_LIMIT)):
        style_axis(axis)
        axis.axvspan(10, 12, color="#888888", alpha=.075, linewidth=0)
        axis.axhline(threshold, color="#777777", linestyle=":", linewidth=1.)
        for method in METHODS:
            color = COLORS[method]
            for trial in trials[method]:
                error = getattr(trial, metric)
                axis.plot(trial.time, error, color=color, alpha=.23, linewidth=.75)
                if trial.time[-1] < HORIZON - DT / 2:
                    axis.plot(trial.time[-1], error[-1], marker="x", color=color,
                              markersize=6, markeredgewidth=1.3)
            summary = summaries[method, metric]
            axis.fill_between(times, summary["q25"], summary["q75"], color=color, alpha=.18, linewidth=0)
            full_cohort = summary["count"] == 10
            axis.plot(times, np.where(full_cohort, summary["median"], np.nan),
                      color=color, linewidth=2.4)
            # Counts and the survivor-conditional interpretation are in the CSV/README.
            conditional = ~full_cohort
            if conditional.any():
                conditional[max(0, np.flatnonzero(conditional)[0] - 1)] = True
                axis.plot(times, np.where(conditional, summary["median"], np.nan),
                          color=color, linewidth=2.4, linestyle="--")
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0)
    axes[0].set_ylim(0, 185)
    axes[0].set_yticks([0, 45, 90, 135, 180])
    axes[1].set(xlabel="Time (s)", xlim=(0, HORIZON))


def tracking_figure(trials, times, summaries, output):
    figure, axes = plt.subplots(2, 1, figsize=(8., 5.5), sharex=True)
    figure.subplots_adjust(left=.12, right=.98, bottom=.12, top=.9, hspace=.16)
    draw_tracking_axes(axes, trials, times, summaries)
    figure.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=(.55, .99),
                  ncol=2, frameon=False, fontsize=11)
    save_figure(figure, output, "tracking_error")


def load_pole_geometry(trial):
    metadata = read_json(trial.directory / "metadata.json")
    rendering = read_json(trial.directory / "rendering.json")
    verify_rendering_hash(trial.directory, metadata)
    model = pin(trial.directory / rendering["model_xml"])
    verify_file_hash(model, rendering.get("model_xml_sha256"), rendering.get("model_sha256"),
                     metadata.get("recording", {}).get("assets", {}).get("model.xml"))
    element = ET.parse(model).find(".//geom[@name='cpole']")
    return np.fromstring(element.attrib["fromto"], sep=" ").reshape(2, 3)


def pole_geometry(pair):
    geometries = [load_pole_geometry(trial) for trial in pair.values()]
    if not np.array_equal(*geometries):
        raise ValueError("The paired records use different pole geometries")
    return geometries[0]


def pose_coordinates(state, geometry):
    position, angle = state[:2]
    x = position + geometry[:, 0] * np.cos(angle) + geometry[:, 2] * np.sin(angle)
    height = -geometry[:, 0] * np.sin(angle) + geometry[:, 2] * np.cos(angle)
    return x, height


def draw_pose(axis, state, geometry, color, alpha=1., zorder=3):
    x, height = pose_coordinates(state, geometry)
    axis.add_patch(Rectangle((state[0] - .12, -.06), .24, .12, facecolor="none",
                             edgecolor=color, linewidth=.8, alpha=alpha, zorder=zorder))
    axis.plot(x, height, color=color, linewidth=2.2, alpha=alpha,
              solid_capstyle="round", zorder=zorder + .1)
    axis.plot(x[-1], height[-1], "o", color=color, markersize=3.5,
              alpha=alpha, zorder=zorder + .2)


def scene_axis(axis, limit):
    axis.set(xlim=(-limit, limit), ylim=(-.77, .77), aspect="equal")
    axis.axhline(0, color="#aaaaaa", linewidth=1.1, zorder=0)
    axis.axvline(0, color="#dddddd", linewidth=.7, linestyle=":", zorder=0)
    axis.set_yticks([])
    axis.spines[["top", "left", "right"]].set_visible(False)
    axis.spines["bottom"].set_color("#dddddd")
    axis.tick_params(axis="x", labelsize=8, color="#bbbbbb")


def scene_limit(pair, geometry):
    radius = np.linalg.norm(geometry[:, [0, 2]], axis=1).max()
    return max(1.4, max(np.max(np.abs(t.states[:, 0])) for t in pair.values()) + radius + .12)


def trajectory_pose_indices(trial):
    """Three shared absolute times plus the method's actual final saved pose."""
    indices = {round(time / DT) for time in POSE_TIMES if time <= trial.time[-1]}
    indices.add(len(trial.time) - 1)
    return sorted(indices)


def draw_trajectory_axes(axes, pair, geometry):
    limit = scene_limit(pair, geometry)
    for axis, method in zip(axes, METHODS):
        trial = pair[method]
        color = COLORS[method]
        axis.set(xlim=(-limit, limit), ylim=(-.75, .75), aspect="equal", ylabel="Height (m)")
        axis.set_title(LABELS[method], color=color, fontsize=12, pad=8)
        axis.set_yticks([-.6, -.3, 0., .3, .6])
        style_axis(axis)
        axis.axhline(0, color="#aaaaaa", linewidth=.8, zorder=0)
        axis.axvline(0, color="#dddddd", linewidth=.6, linestyle=":", zorder=0)
        # The dashed path uses every saved tip position, including the terminal
        # state. This does not extrapolate the shorter Select-DPC trajectory.
        angles = trial.states[:, 1]
        tip_x = trial.states[:, 0] + geometry[-1, 0] * np.cos(angles) + geometry[-1, 2] * np.sin(angles)
        tip_height = -geometry[-1, 0] * np.sin(angles) + geometry[-1, 2] * np.cos(angles)
        axis.plot(tip_x, tip_height, color=color, linewidth=1., linestyle="--", alpha=.65, zorder=1)
        for order, index in enumerate(trajectory_pose_indices(trial)):
            alpha = .10 + .80 * trial.time[index] / HORIZON
            draw_pose(axis, trial.states[index], geometry, color, alpha=alpha, zorder=2 + order)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("x (m)")


def trajectory_figure(pair, geometry, output):
    figure, axes = plt.subplots(2, 1, figsize=(4.4, 5.2), sharex=True, sharey=True)
    figure.subplots_adjust(left=.16, right=.98, bottom=.11, top=.93, hspace=.28)
    draw_trajectory_axes(axes, pair, geometry)
    save_figure(figure, output, "trajectory_comparison")


def combined_figure(trials, times, summaries, pair, geometry, output):
    figure = plt.figure(figsize=(10., 4.9))
    grid = figure.add_gridspec(2, 2, width_ratios=[1., 1.5], left=.07, right=.985,
                              bottom=.12, top=.9, hspace=.28, wspace=.36)
    trajectory_top = figure.add_subplot(grid[0, 0])
    trajectory_bottom = figure.add_subplot(grid[1, 0], sharex=trajectory_top, sharey=trajectory_top)
    error_top = figure.add_subplot(grid[0, 1])
    error_bottom = figure.add_subplot(grid[1, 1], sharex=error_top)
    draw_trajectory_axes([trajectory_top, trajectory_bottom], pair, geometry)
    draw_tracking_axes([error_top, error_bottom], trials, times, summaries)
    error_top.tick_params(labelbottom=False)
    right_box = error_top.get_position()
    figure.legend(handles=method_handles(), loc="upper center",
                  bbox_to_anchor=((right_box.x0 + right_box.x1) / 2, .99),
                  ncol=2, frameon=False, fontsize=10)
    save_figure(figure, output, "cartpole_combined")


def trajectory_movie(pair, geometry, output):
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("ffmpeg is required for the movie; use --skip-video for static plots only")
    figure = plt.figure(figsize=(12, 6.4))
    grid = figure.add_gridspec(2, 2, height_ratios=[2.3, 1], left=.07, right=.98,
                              bottom=.1, top=.86, hspace=.28, wspace=.18)
    scenes = [figure.add_subplot(grid[0, column]) for column in range(2)]
    history = figure.add_subplot(grid[1, :])
    style_axis(history)
    history.set(xlim=(0, HORIZON), ylim=(0, 185), xlabel="Time (s)", ylabel="Angle error (deg)")
    history.axhline(ANGLE_LIMIT, color="#777777", linewidth=.8, linestyle=":")
    history.set_yticks([0, 90, 180])
    clock = figure.suptitle("", y=.965, fontsize=16)
    artists = {}
    limit = scene_limit(pair, geometry)
    for axis, method in zip(scenes, METHODS):
        scene_axis(axis, limit)
        axis.set_xlabel("Cart position (m)")
        axis.set_title(LABELS[method], color=COLORS[method], fontsize=14, weight="bold")
        cart = Rectangle((0, -.06), .24, .12, color=COLORS[method], zorder=3)
        axis.add_patch(cart)
        pole, = axis.plot([], [], color=COLORS[method], linewidth=5, solid_capstyle="round")
        tip, = axis.plot([], [], "o", color=COLORS[method], markersize=6)
        label = axis.text(.5, .02, "", transform=axis.transAxes, ha="center", fontsize=10)
        ended = axis.text(.5, .55, "", transform=axis.transAxes, ha="center", va="center", color="#777777", fontsize=13,
                          bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
        curve, = history.plot([], [], color=COLORS[method], linewidth=2, label=LABELS[method])
        artists[method] = (cart, pole, tip, label, ended, curve)
    history.legend(loc="upper right", frameon=False, fontsize=9)
    cursor = history.axvline(0, color="#999999", linewidth=.8)
    writer = FFMpegWriter(fps=25, metadata={"title": f"Cartpole seed {pair['bd_chi2'].seed}: saved-state replay"},
                         extra_args=["-threads", "1", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"])
    with writer.saving(figure, str(output / "closed_loop_comparison.mp4"), dpi=100):
        for index in range(round(HORIZON / DT) + 1):
            time = index * DT
            clock.set_text(f"Cartpole · seed {pair['bd_chi2'].seed} · t = {time:.2f} s")
            cursor.set_xdata([time, time])
            for method in METHODS:
                trial = pair[method]
                cart, pole, tip, label, ended, curve = artists[method]
                available = index < len(trial.time)
                for artist in (cart, pole, tip):
                    artist.set_visible(available)
                if available:
                    state = trial.states[index]
                    x, height = pose_coordinates(state, geometry)
                    cart.set_xy((state[0] - .12, -.06))
                    pole.set_data(x, height)
                    tip.set_data([x[-1]], [height[-1]])
                    label.set_text(f"x = {state[0]:+.2f} m   |θ| = {trial.angle_error[index]:.1f}°   ω = {state[3]:+.2f} rad/s")
                    ended.set_text("")
                else:
                    label.set_text("")
                    ended.set_text(f"Terminated at {trial.time[-1]:.2f} s\nNo later states recorded")
                stop = min(index + 1, len(trial.time))
                curve.set_data(trial.time[:stop], trial.angle_error[:stop])
            writer.grab_frame(facecolor="white")
    plt.close(figure)


def export_data(trials, times, summaries, output):
    with (output / "tracking_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "metric", "time_s", "n_recorded", "median", "q25", "q75", "mean", "sample_sd"])
        for (method, metric), summary in summaries.items():
            for index, time in enumerate(times):
                writer.writerow([method, metric, time] + [summary[k][index] for k in
                                ("count", "median", "q25", "q75", "mean", "sample_sd")])
    with (output / "tracking_trials.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "seed", "success", "time_s", "angle_error_deg", "position_error_m"])
        for method, group in trials.items():
            for trial in group:
                writer.writerows((method, trial.seed, trial.result["success"], *row) for row in
                                 zip(trial.time, trial.angle_error, trial.position_error))


def write_documentation(root, trials, pair, output, include_video):
    selected = pair["select_dpc"]
    interval = max(balanced_intervals(selected), key=lambda span: span[1] - span[0])
    first_end = min(t.time[-1] for group in trials.values() for t in group)
    ending = (f"terminates at **{selected.time[-1]:.2f} s** (raw angle {selected.states[-1, 1]:.5f} rad crosses the simulator's 7 rad cutoff)"
              if abs(selected.states[-1, 1]) > 7 else
              f"fails the benchmark after {selected.time[-1]:.2f} recorded seconds (reason: {selected.result['terminal_reason']})")
    text = f"""# Cartpole comparison plots

![Combined trajectory and tracking figure](cartpole_combined.png)

[Combined PDF](cartpole_combined.pdf) · [Combined SVG](cartpole_combined.svg)

The combined figure places the BD-χ² and Select-DPC trajectory panels vertically on the left, and the angle and position errors vertically on the right. The standalone figures are retained below.

![Tracking errors](tracking_error.png)

All ten shared starts are included. Green is BD-χ²; orange is Select-DPC.
Angle error is |atan2(sin θ, cos θ)| in degrees, so complete rotations do not create artificial error and opposite signs cannot cancel. Position error is |x| in meters. The reference is upright at x=0.

Thin curves show individual trials; thick curves show the pointwise median and shading the interquartile range (linear empirical quantiles). This band describes variation across recorded trials, not uncertainty in a population mean. Mean and sample SD are also exported in `tracking_summary.csv`.

**Early termination.** All ten trials contribute only through {first_end:.2f} s. Five Select-DPC trials terminate early; five remain recorded at 12 s, including one that fails stabilization. The dashed median after the first termination summarizes only remaining recorded trials. Crosses mark early endpoints, and `tracking_summary.csv` includes sample counts. No values are imputed after termination. BD-χ² succeeds on 10/10 starts; Select-DPC on 4/10.

The dotted lines indicate the angle (15°) and position (1 m) tolerances. These alone do not define success: the benchmark also requires |angular velocity|≤1 rad/s throughout the final 50 steps and a complete 12 s episode. The final two seconds are lightly shaded.

![Vertically stacked pole-tip trajectories](trajectory_comparison.png)

Seed **{selected.seed}** illustrates temporary balance followed by failure: Select-DPC balances from **{interval[0]:.2f} to {interval[1]:.2f} s**, then {ending}. BD-χ² completes 12 s. The default selection is the lowest seed where BD-χ² succeeds and failed Select-DPC maintains all three balance tolerances for at least two consecutive sampled seconds. This is a deliberately selected failure illustration, not a typical-trial claim.

The trajectory panels show BD-χ² above Select-DPC on identical axes with equal physical scaling. Dashed lines trace recorded pole-tip positions. Solid cart-and-pole poses show 0, 0.8, 3.6 s and each actual endpoint. Fainter poses are earlier on a shared 12 s time scale. Pole endpoints use the model geometry; the cart is schematic. The optional movie replays the saved states.

Figures use Times New Roman when available, otherwise DejaVu Serif. Set `PAPER_FONT` to choose another installed font.

{"[Side-by-side saved-state movie](closed_loop_comparison.mp4) · " if include_video else ""}[Tracking PDF](tracking_error.pdf) · [Tracking SVG](tracking_error.svg) · [Trajectory PDF](trajectory_comparison.pdf) · [Trajectory SVG](trajectory_comparison.svg)

Regenerate from the project root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/plots/cartpole.py
```

Use `--skip-video` for static figures only; `--seed` selects another paired trial. Defaults read `results/paper` and write this directory. `--root` and `--output` override those paths. Raw tracking errors and summary statistics are exported as CSV.
"""
    (output / "README.md").write_text(text)
    for path, expected in INPUT_HASHES.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Input changed during rendering: {path}")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_root=str(root),
                    script_path=str(Path(__file__).resolve()), script_sha256=digest(Path(__file__)), input_sha256=INPUT_HASHES,
                    selected_seed=selected.seed, colors=COLORS, timestep_s=DT,
                    seeds=[t.seed for t in trials[METHODS[0]]],
                    tracking_error="Absolute principal-angle error (degrees) and absolute cart displacement (m)",
                    summary="Pointwise available-trial median and linear 25th/75th empirical quantiles",
                    missing_data="No imputation; counts retained in CSV; summaries with n<10 are survivor-conditional",
                    trajectory_plot={"layout": "BD-chi2 above Select-DPC; shared physical axes; trajectories left and errors right in combined figure",
                                     "shared_pose_times_s": list(POSE_TIMES), "include_terminal_states": True,
                                     "pole_tip_path": "Dashed line through all saved states; no extrapolation",
                                     "alpha_range": [.10, .90], "common_time_scale_s": [0, HORIZON]},
                    font_family=FONT_FAMILY,
                    common_all_trial_interval_s=[0., float(first_end)],
                    early_termination_seeds={m: [t.seed for t in group if t.time[-1] < HORIZON - DT / 2]
                                             for m, group in trials.items()},
                    selected_balance_intervals={m: balanced_intervals(t) for m, t in pair.items()},
                    simulations_run=0, numerical_controller_calls=0,
                    software={"numpy": np.__version__, "matplotlib": matplotlib.__version__},
                    output_sha256={p.name: digest(p) for p in sorted(output.iterdir())
                                   if p.is_file() and p.name != "manifest.json"})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def main():
    default_root = (Path(__file__).resolve().parents[2] / "results/paper")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "figures/cartpole").resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Explicit registration also handles a Matplotlib cache made before the
    # requested fonts were installed. Font binaries remain outside the repository.
    local_fonts = Path.home() / ".local/share/fonts/msttcorefonts"
    for font_path in sorted(local_fonts.glob("*.TTF")):
        font_manager.fontManager.addfont(str(pin(font_path)))
    try:
        font_path = Path(font_manager.findfont(FONT_FAMILY, fallback_to_default=False))
    except ValueError as error:
        raise RuntimeError("Install Times New Roman before generating these figures") from error
    pin(font_path)
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": 10,
                         "axes.labelsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"})
    trials = load_trials(root)
    pair = choose_pair(trials, args.seed)
    geometry = pole_geometry(pair)
    times, summaries = summarize(trials)
    tracking_figure(trials, times, summaries, output)
    trajectory_figure(pair, geometry, output)
    combined_figure(trials, times, summaries, pair, geometry, output)
    export_data(trials, times, summaries, output)
    if not args.skip_video:
        trajectory_movie(pair, geometry, output)
    write_documentation(root, trials, pair, output, not args.skip_video)
    print(json.dumps({"output": str(output), "selected_seed": pair["bd_chi2"].seed,
                      "paired_starts": 10, "simulations_run": 0}), flush=True)


if __name__ == "__main__":
    main()
