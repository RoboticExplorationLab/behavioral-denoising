"""Plot Rocket landing trajectories and tracking errors from saved results.

Run: .venv/bin/python scripts/plots/rocket.py
No controller, physics simulator, or experiment is executed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.plots.common import PAPER_RESULTS, REPOSITORY, paper_font, replay_input_path
from scripts.plots.common import verify_rendering_hash

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np


METHODS = ("bd_chi2", "select_dpc")
LABELS = {"bd_chi2": "BD-χ²", "select_dpc": "Select-DPC"}
COLORS = {"bd_chi2": "#2ca02c", "select_dpc": "#ff7f0e"}
FONT_FAMILY = paper_font()
DT = 1 / 60
POSE_TIMES = (0., .5, 1.)
INPUT_HASHES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    path = path.resolve()
    value = digest(path)
    if INPUT_HASHES.setdefault(str(path), value) != value:
        raise ValueError(f"Input changed while plotting: {path}")
    return path


def read_json(path):
    return json.loads(pin(path).read_text())


@dataclass
class Trial:
    directory: Path
    method: str
    seed: int
    states: np.ndarray
    time: np.ndarray
    references: np.ndarray
    result: dict
    rendering: dict
    bodies: dict

    @property
    def angle_error(self):
        angle = self.states[:, 4]
        return np.rad2deg(np.abs(np.arctan2(np.sin(angle), np.cos(angle))))


def load_trial(directory, method):
    metadata = read_json(directory / "metadata.json")
    result = read_json(directory / "result.json")
    rendering = read_json(directory / "rendering.json")
    trajectory = pin(directory / "trajectory.npz")
    render_state = pin(directory / "render_state.npz")
    if (metadata["trajectory_sha256"] != digest(trajectory)
            or metadata["recording"]["render_state_sha256"] != digest(render_state)):
        raise ValueError(f"Saved-data hashes differ from metadata: {directory}")
    verify_rendering_hash(directory, metadata, required=True)
    with np.load(trajectory, allow_pickle=False) as saved:
        states, times = saved["states"].copy(), saved["timestamps"].copy()
        references = saved["references"].copy()
        steps = len(saved["actions"])
        if (states.shape != (steps + 1, 6) or references.shape != (steps, 6)
                or not np.array_equal(states, saved["observations"])
                or not np.array_equal(states, saved["raw_observations"][:, :6])
                or not np.isfinite(states).all() or not np.isfinite(references).all()
                or not np.allclose(times, np.arange(steps + 1) * DT, atol=1e-10, rtol=0)
                or int(result["steps"]) != steps):
            raise ValueError(f"Unexpected Rocket state/time/reference alignment: {directory}")
    with np.load(render_state, allow_pickle=False) as saved:
        bodies = {key: saved[key].copy() for key in saved.files}
    names = bodies["body_names"].tolist()
    lander = names.index("lander")
    if (not np.array_equal(bodies["body_position"][:, lander], states[:, :2])
            or not np.array_equal(bodies["body_angle"][:, lander], states[:, 4])
            or not np.array_equal(bodies["time"], times)
            or not np.isclose(rendering["timestep"], DT)
            or metadata["config"]["task"]["mode"] != "landing"):
        raise ValueError(f"Body poses or protocol differ from the recorded state: {directory}")
    return Trial(directory, method, int(result["seed"]), states, times, references, result, rendering, bodies)


def load_trials(root):
    trials = {method: [load_trial(path, method) for path in
              sorted((root / "rocket" / method).glob("seed_*"))] for method in METHODS}
    if any(len(group) != 10 for group in trials.values()):
        raise ValueError("Expected ten finalized starts per method")
    for first, second in zip(trials[METHODS[0]], trials[METHODS[1]]):
        if first.seed != second.seed or not np.array_equal(first.states[0], second.states[0]):
            raise ValueError("Paired seeds or initial states differ")
    # The task has a lower landing reference and an optional approach
    # reference exactly 5 m higher. Use the shared fixed landing goal for this plot.
    references = np.vstack([trial.references for group in trials.values() for trial in group])
    if len(np.unique(references[:, 0])) != 1 or np.any(references[:, 2:] != 0):
        raise ValueError("Expected one fixed horizontal target and upright/zero-speed references")
    goal = np.array([references[0, 0], references[:, 1].min()])
    allowed_heights = np.array([goal[1], goal[1] + 5.])
    if not np.isin(references[:, 1], allowed_heights).all():
        raise ValueError("Reference heights do not match the audited landing/approach protocol")
    return trials, goal


def choose_pair(trials, seed):
    pairs = [(first, second) for first, second in zip(trials[METHODS[0]], trials[METHODS[1]])
             if first.result["success"] and not second.result["success"]]
    if seed is not None:
        pairs = [pair for pair in pairs if pair[0].seed == seed]
    if not pairs:
        raise ValueError("Choose a recorded BD-success/Select-failure pair")
    return dict(zip(METHODS, pairs[0]))


def errors(trial, goal):
    return {"angle_error_deg": trial.angle_error,
            "landing_distance_m": np.linalg.norm(trial.states[:, :2] - goal, axis=1)}


def summarize(trials, goal):
    frames = max(len(trial.time) for group in trials.values() for trial in group)
    times = np.arange(frames) * DT
    summaries = {}
    for method, group in trials.items():
        for metric in ("angle_error_deg", "landing_distance_m"):
            values = np.full((10, frames), np.nan)
            for row, trial in enumerate(group):
                values[row, :len(trial.time)] = errors(trial, goal)[metric]
            summary = {key: np.full(frames, np.nan) for key in ("median", "q25", "q75", "mean", "sample_sd")}
            summary["count"] = np.isfinite(values).sum(axis=0)
            for index in range(frames):
                observed = values[:, index][np.isfinite(values[:, index])]
                if len(observed):
                    summary["median"][index] = np.median(observed)
                    summary["q25"][index], summary["q75"][index] = np.quantile(observed, [.25, .75], method="linear")
                    summary["mean"][index] = np.mean(observed)
                if len(observed) > 1:
                    summary["sample_sd"][index] = np.std(observed, ddof=1)
            summaries[method, metric] = summary
    return times, summaries


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dddddd", linewidth=.6)
    axis.set_axisbelow(True)


def method_handles():
    return [Line2D([], [], color=COLORS[method], linewidth=2.8, label=LABELS[method]) for method in METHODS]


def draw_errors(axes, trials, goal, times, summaries):
    for axis, metric, label in zip(axes, ("angle_error_deg", "landing_distance_m"),
                                    ("Tilt error (deg)", "Distance to landing point (m)")):
        style_axis(axis)
        for method in METHODS:
            color = COLORS[method]
            for trial in trials[method]:
                values = errors(trial, goal)[metric]
                axis.plot(trial.time, values, color=color, alpha=.23, linewidth=.75)
                marker = "o" if trial.result["success"] else "x"
                axis.plot(trial.time[-1], values[-1], marker=marker, color=color,
                          markerfacecolor="none", markersize=4.5, markeredgewidth=1.)
            summary = summaries[method, metric]
            axis.fill_between(times, summary["q25"], summary["q75"], color=color, alpha=.18, linewidth=0)
            complete = summary["count"] == 10
            axis.plot(times, np.where(complete, summary["median"], np.nan), color=color, linewidth=2.4)
            conditional = ~complete
            if conditional.any():
                conditional[max(0, np.flatnonzero(conditional)[0] - 1)] = True
                axis.plot(times, np.where(conditional, summary["median"], np.nan),
                          color=color, linewidth=2.4, linestyle="--")
        axis.set(ylabel=label, ylim=(0, None), xlim=(0, np.ceil(times[-1])))
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("Time (s)")


def world_vertices(trial, body, index):
    body_index = trial.bodies["body_names"].tolist().index(body["name"])
    angle = trial.bodies["body_angle"][index, body_index]
    position = trial.bodies["body_position"][index, body_index]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for fixture in body["fixtures"]:
        if fixture["type"] != "polygon":
            raise ValueError("Expected the recorded Rocket polygon fixtures")
        yield np.asarray(fixture["vertices"]) @ rotation.T + position


def scene_bounds(pair):
    minimum, maximum = np.full(2, np.inf), np.full(2, -np.inf)
    for trial in pair.values():
        for index in range(len(trial.time)):
            for body in trial.rendering["rocket_bodies"]:
                for vertices in world_vertices(trial, body, index):
                    minimum = np.minimum(minimum, vertices.min(axis=0))
                    maximum = np.maximum(maximum, vertices.max(axis=0))
    return np.floor((minimum - 1.) * 2) / 2, np.ceil((maximum + 1.) * 2) / 2


def pose_indices(trial):
    indices = {round(time / DT) for time in POSE_TIMES if time <= trial.time[-1]}
    indices.add(len(trial.time) - 1)
    return sorted(indices)


def draw_trajectories(axes, pair):
    minimum, maximum = scene_bounds(pair)
    opacity_horizon = max(trial.time[-1] for trial in pair.values())
    for axis, method in zip(axes, METHODS):
        trial = pair[method]
        axis.set(xlim=(minimum[0], maximum[0]), ylim=(minimum[1], maximum[1]),
                 aspect="equal", ylabel="Height (m)")
        axis.set_title(LABELS[method], color=COLORS[method], fontsize=12, pad=8)
        style_axis(axis)
        barge = next(body for body in trial.rendering["rocket_bodies"] if body["name"] == "barge")
        barge_index = trial.bodies["body_names"].tolist().index("barge")
        positions = trial.bodies["body_position"][:, barge_index]
        angles = trial.bodies["body_angle"][:, barge_index]
        if (not np.allclose(positions, positions[0], atol=1e-5, rtol=0)
                or not np.allclose(angles, angles[0], atol=1e-6, rtol=0)):
            raise ValueError("A moving barge needs a time-resolved background")
        for vertices in world_vertices(trial, barge, 0):
            axis.add_patch(Polygon(vertices, facecolor="#eeeeee", edgecolor="#777777", linewidth=.8, zorder=1))
        axis.plot(trial.states[:, 0], trial.states[:, 1], linestyle="--", linewidth=1.,
                  color=COLORS[method], alpha=.65, zorder=2)
        for order, index in enumerate(pose_indices(trial)):
            alpha = .25 + .65 * trial.time[index] / opacity_horizon
            for body in trial.rendering["rocket_bodies"]:
                if body["name"] == "barge":
                    continue
                for vertices in world_vertices(trial, body, index):
                    axis.add_patch(Polygon(vertices, facecolor="none", edgecolor=COLORS[method],
                                           linewidth=1., alpha=alpha, zorder=3 + order))
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("x (m)")


def save_figure(figure, output, name):
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=240, facecolor="white")
    plt.close(figure)


def make_figures(trials, goal, times, summaries, pair, output):
    figure = plt.figure(figsize=(10., 6.2))
    grid = figure.add_gridspec(2, 2, width_ratios=[1., 1.9], left=.07, right=.985,
                              bottom=.10, top=.91, hspace=.28, wspace=.35)
    trajectory_top = figure.add_subplot(grid[0, 0])
    trajectory_bottom = figure.add_subplot(grid[1, 0], sharex=trajectory_top, sharey=trajectory_top)
    error_top = figure.add_subplot(grid[0, 1])
    error_bottom = figure.add_subplot(grid[1, 1], sharex=error_top)
    draw_trajectories([trajectory_top, trajectory_bottom], pair)
    draw_errors([error_top, error_bottom], trials, goal, times, summaries)
    box = error_top.get_position()
    figure.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=((box.x0 + box.x1) / 2, .99),
                  ncol=2, frameon=False, fontsize=10)
    save_figure(figure, output, "rocket_combined")

    figure, axes = plt.subplots(2, 1, figsize=(8., 5.5), sharex=True)
    figure.subplots_adjust(left=.13, right=.98, bottom=.12, top=.9, hspace=.16)
    draw_errors(axes, trials, goal, times, summaries)
    figure.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=(.55, .99),
                  ncol=2, frameon=False, fontsize=11)
    save_figure(figure, output, "tracking_error")

    figure, axes = plt.subplots(2, 1, figsize=(4.4, 6.2), sharex=True, sharey=True)
    figure.subplots_adjust(left=.16, right=.98, bottom=.10, top=.93, hspace=.25)
    draw_trajectories(axes, pair)
    save_figure(figure, output, "trajectory_comparison")


def export_data(trials, goal, times, summaries, output):
    with (output / "tracking_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "metric", "time_s", "n_recorded", "median", "q25", "q75", "mean", "sample_sd"])
        for (method, metric), summary in summaries.items():
            for index, time in enumerate(times):
                values = [summary[key][index] for key in ("count", "median", "q25", "q75", "mean", "sample_sd")]
                writer.writerow([method, metric, time] + [value if np.isfinite(value) else "" for value in values])
    with (output / "tracking_trials.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "seed", "success", "time_s", "x_m", "y_m", "angle_error_deg", "landing_distance_m"])
        for method, group in trials.items():
            for trial in group:
                values = errors(trial, goal)
                for index, time in enumerate(trial.time):
                    writer.writerow([method, trial.seed, trial.result["success"], time, *trial.states[index, :2],
                                     values["angle_error_deg"][index], values["landing_distance_m"][index]])


def write_documentation(root, trials, goal, times, pair, output):
    selected = pair["select_dpc"]
    first_end = min(trial.time[-1] for group in trials.values() for trial in group)
    text = f"""# Rocket landing comparison

![Combined figure](rocket_combined.png)

[Combined PDF](rocket_combined.pdf) · [Combined SVG](rocket_combined.svg) · [Tracking PDF](tracking_error.pdf) · [Trajectory PDF](trajectory_comparison.pdf)

The figure places BD-χ² (green) above Select-DPC (orange) on the left, and two error panels over ten shared starts on the right.

**Errors.** Tilt is |atan2(sin θ, cos θ)| in degrees. Position error is the Euclidean distance from the rocket body center to the fixed landing reference **({goal[0]:.8f}, {goal[1]:.8f}) m**. This y-coordinate is the desired rocket-center height, not the barge surface. Successful landing can leave a nonzero position error. The goal is recovered from the lower saved reference stage and verified against every record. The active controller reference can initially be 5 m higher; therefore this figure's landing distance differs from the reference error used in the running cost.

**Aggregation and endpoints.** Thin curves show every saved trial. Thick lines show pointwise available-trial medians; bands show linear empirical 25th–75th percentiles, not confidence intervals. Circles mark successful endpoints and crosses failed endpoints. The median becomes dashed once fewer than ten records remain. Both successful and failed trials leave the sample when their recorded rollout ends; dashed summaries are conditional on remaining recorded trials, not just survivors of failures. Sample counts, means and sample SD are retained in `tracking_summary.csv`. At n=1 the IQR is zero and sample SD is undefined; at n=0 summaries are blank. No landed or failed trajectory is carried forward. All twenty trajectories exist only through **{first_end:.5f} s**; the latest recorded endpoint is **{times[-1]:.5f} s**, below the 16.6667 s evaluation cap. BD-χ² succeeds on 10/10 trials, Select-DPC on 6/10.

**Selected trajectory.** Seed **{selected.seed}** shows BD-χ² success and Select-DPC failure. The default selects the lowest shared seed with this outcome pair. Their initial physical states match exactly. BD ends at **{pair['bd_chi2'].time[-1]:.5f} s** with `{pair['bd_chi2'].result['terminal_reason']}`; Select ends at **{selected.time[-1]:.5f} s** with `{selected.result['terminal_reason']}`. In the default seed 3005, Select crosses the right position boundary without ground contact; its recorded `crashed` flag does not indicate a ground impact. This is a selected failure illustration, not a representative-trial claim.

**Geometry and time.** Dashed paths follow the recorded rocket-body center. Body, legs and nozzle use their fixture polygons and recorded positions and angles; the gray barge uses its initial pose. Both panels have equal physical scaling and identical limits, including the out-of-bounds endpoint. Four poses show 0, 0.5, 1.0 s and each actual endpoint. Fainter poses are earlier on a shared time scale ending at {max(t.time[-1] for t in pair.values()):.5f} s.

Regenerate from the project root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/plots/rocket.py
```

`--seed`, `--root` and `--output` override the selected pair and paths. Figures use Times New Roman when available, otherwise DejaVu Serif; `PAPER_FONT` chooses another installed font. `tracking_trials.csv` contains per-state errors and `tracking_summary.csv` contains aggregate statistics.
"""
    (output / "README.md").write_text(text)
    for path, expected in INPUT_HASHES.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Input changed while plotting: {path}")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_root=str(root),
                    script_path=str(Path(__file__).resolve()), script_sha256=digest(Path(__file__)), input_sha256=INPUT_HASHES,
                    selected_seed=selected.seed, fixed_landing_reference_m=goal.tolist(),
                    font_family=FONT_FAMILY, colors=COLORS, timestep_s=DT,
                    seeds=[trial.seed for trial in trials[METHODS[0]]],
                    tracking_error="Absolute wrapped tilt; Euclidean distance to fixed landing reference",
                    summary="Pointwise available-trial median/IQR; successful and failed endpoints both remove observations",
                    no_post_termination_padding=True,
                    common_all_trial_interval_s=[0., float(first_end)],
                    trajectory=dict(path="saved lander-body center", line_style="dashed", physical_aspect="equal",
                                    shared_pose_times_s=list(POSE_TIMES), include_terminal_states=True,
                                    barge_pose="Initial saved transform, after checking translation drift<=1e-5m per coordinate and angle drift<=1e-6rad",
                                    alpha_range=[.25, .90], common_time_scale_s=[0., float(max(t.time[-1] for t in pair.values()))],
                                    displayed_pose_times_s={method: trial.time[pose_indices(trial)].tolist() for method, trial in pair.items()}),
                    outcomes={method: {"successes": sum(bool(t.result["success"]) for t in group),
                                       "recorded_durations_s": {str(t.seed): float(t.time[-1]) for t in group}}
                              for method, group in trials.items()},
                    simulations_run=0, numerical_controller_calls=0,
                    software={"numpy": np.__version__, "matplotlib": matplotlib.__version__},
                    output_sha256={path.name: digest(path) for path in sorted(output.iterdir())
                                   if path.is_file() and path.name != "manifest.json"})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=(Path(__file__).resolve().parents[2] / "results/paper"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "figures/rocket").resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted((Path.home() / ".local/share/fonts/msttcorefonts").glob("*.TTF")):
        font_manager.fontManager.addfont(str(pin(path)))
    try:
        pin(Path(font_manager.findfont(FONT_FAMILY, fallback_to_default=False)))
    except ValueError as error:
        raise RuntimeError("Install Times New Roman before generating these figures") from error
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": 10, "axes.labelsize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    trials, goal = load_trials(root)
    pair = choose_pair(trials, args.seed)
    times, summaries = summarize(trials, goal)
    make_figures(trials, goal, times, summaries, pair, output)
    export_data(trials, goal, times, summaries, output)
    write_documentation(root, trials, goal, times, pair, output)
    print(json.dumps({"output": str(output), "selected_seed": pair[METHODS[0]].seed,
                      "paired_starts": 10, "simulations_run": 0}), flush=True)


if __name__ == "__main__":
    main()
