"""Plot Reacher trajectories, reference completion, and constraint violations.

Run from the project root: .venv/bin/python scripts/plots/reacher.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np


METHODS = ("bd_chi2", "select_dpc")
LABELS = {"bd_chi2": "BD-χ²", "select_dpc": "Select-DPC"}
COLORS = {"bd_chi2": "#2ca02c", "select_dpc": "#ff7f0e"}
FONT_FAMILY = paper_font()
DT = .02
MAX_STEPS = 500
TARGET_COUNT = 5
TARGET_TOLERANCE = .005
HEIGHT_LIMIT = .1
POSE_TIMES = (0., .8, 2.)
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
    observations: np.ndarray
    time: np.ndarray
    targets: np.ndarray
    completion_steps: np.ndarray
    points: np.ndarray
    result: dict

    @property
    def steps(self):
        return len(self.time) - 1

    @property
    def violation_mm(self):
        return 1000 * np.maximum(self.observations[:, 5] - HEIGHT_LIMIT, 0.)


def load_trial(directory, method):
    metadata = read_json(directory / "metadata.json")
    result = read_json(directory / "result.json")
    rendering = read_json(directory / "rendering.json")
    trajectory_path = pin(directory / "trajectory.npz")
    render_path = pin(directory / "render_state.npz")
    xml_path = pin(directory / "rendering_assets/model.xml")
    if (metadata["trajectory_sha256"] != digest(trajectory_path)
            or metadata["recording"]["render_state_sha256"] != digest(render_path)):
        raise ValueError(f"Saved-data hashes differ from metadata: {directory}")
    verify_rendering_hash(directory, metadata)
    verify_file_hash(xml_path, metadata["recording"]["assets"]["model.xml"],
                     rendering.get("model_xml_sha256"))
    if (not np.isclose(metadata["control_timestep_s"], DT)
            or not np.isclose(rendering["timestep"], DT)
            or not np.isclose(rendering["target_tolerance"], TARGET_TOLERANCE)
            or not np.isclose(metadata["config"]["height_limit"], HEIGHT_LIMIT)):
        raise ValueError(f"Unexpected Reacher protocol: {directory}")
    with np.load(trajectory_path, allow_pickle=False) as saved:
        observations = saved["observations"].copy()
        times = saved["timestamps"].copy()
        targets = saved["targets"].copy()
        counts = saved["target_completion_counts"].copy()
        completion_steps = saved["target_completion_steps"].copy()
        references = saved["references"].copy()
        steps = len(saved["actions"])
        if (observations.shape != (steps + 1, 8) or targets.shape != (TARGET_COUNT, 2)
                or counts.shape != (steps + 1,) or references.shape != (steps, 8)
                or not np.isfinite(observations).all() or not np.isfinite(targets).all()
                or not np.array_equal(saved["states"], observations)
                or not np.array_equal(saved["time"], times)
                or not np.allclose(times, np.arange(steps + 1) * DT, atol=1e-9, rtol=0)
                or int(result["steps"]) != steps or not 0 < steps <= MAX_STEPS):
            raise ValueError(f"Unexpected saved state/time alignment: {directory}")
    # Independently recover ordered target hits from post-transition world positions.
    # One target can be attained per state; the fifth hit ends the task.
    next_target = 0
    recovered_steps = []
    recovered_counts = np.zeros(steps + 1, dtype=int)
    for index, fingertip in enumerate(observations[:, 4:6]):
        if next_target < TARGET_COUNT:
            distance = np.linalg.norm(fingertip - targets[next_target])
            if distance < TARGET_TOLERANCE:
                recovered_steps.append(index)
                next_target += 1
        recovered_counts[index] = next_target
    if (counts[0] != 0 or not np.array_equal(counts, recovered_counts)
            or not np.array_equal(completion_steps, recovered_steps)
            or not np.array_equal(references[:, 4:6], targets[recovered_counts[:-1]])
            or bool(result["success"]) != (next_target == TARGET_COUNT)):
        raise ValueError(f"Stored target events differ from ordered 5 mm hits: {directory}")
    if next_target == TARGET_COUNT:
        if (completion_steps[-1] != steps or result["first_five_completion_step"] != steps
                or not np.isclose(result["first_five_completion_time_s"], steps * DT)):
            raise ValueError(f"Successful trial does not stop at its fifth hit: {directory}")
    elif result["first_five_completion_step"] is not None:
        raise ValueError(f"Failed trial has a fifth-target completion step: {directory}")
    violating = observations[:, 5] > HEIGHT_LIMIT
    if violating[0] or np.count_nonzero(violating[1:]) != result["violations_strict"]:
        raise ValueError(f"Saved constraint counts or initial feasibility differ: {directory}")
    with np.load(render_path, allow_pickle=False) as saved:
        body_positions = saved["body_xpos"].copy()
        if not np.array_equal(saved["time"], times):
            raise ValueError(f"Saved render and trajectory clocks differ: {directory}")
    world = ET.parse(xml_path).getroot().find("worldbody")
    if world is None:
        raise ValueError(f"Archived model has no world body: {xml_path}")
    names = ["world"] + [body.get("name") for body in world.iter("body")]
    if body_positions.shape != (steps + 1, len(names), 3):
        raise ValueError(f"Saved body transforms do not match the archived XML: {directory}")
    indices = [names.index(name) for name in ("body0", "body1", "fingertip")]
    points = body_positions[:, indices, :2]
    if (not np.isfinite(points).all()
            or not np.allclose(points[:, -1], observations[:, 4:6], atol=1e-12, rtol=0)):
        raise ValueError(f"Saved world fingertip differs from task observations: {directory}")
    return Trial(directory, method, int(result["seed"]), observations, times,
                 targets, completion_steps, points, result)


def load_trials(root):
    trials = {method: [load_trial(path, method) for path in
              sorted((root / "reacher" / method).glob("seed_*"))] for method in METHODS}
    if any(len(group) != 10 for group in trials.values()):
        raise ValueError("Expected ten finalized starts per method")
    for first, second in zip(trials[METHODS[0]], trials[METHODS[1]]):
        if (first.seed != second.seed
                or not np.array_equal(first.observations[0], second.observations[0])
                or not np.array_equal(first.targets, second.targets)):
            raise ValueError("Paired seeds, initial observations or targets differ")
    return trials


def choose_pair(trials, seed):
    pairs = [(first, second) for first, second in zip(trials[METHODS[0]], trials[METHODS[1]])
             if first.result["success"] and not second.result["success"]]
    if seed is not None:
        pairs = [pair for pair in pairs if pair[0].seed == seed]
    if not pairs:
        raise ValueError("Choose a recorded BD-success/Select-failure pair")
    return dict(zip(METHODS, pairs[0]))


def progress(trials):
    grid = np.arange(MAX_STEPS + 1)
    # Achieved events remain achieved after stopping. This extends cumulative
    # event counts, never physical trajectories or tracking errors.
    counts = {method: np.vstack([np.searchsorted(trial.completion_steps, grid, side="right")
                                for trial in group]) for method, group in trials.items()}
    return grid * DT, counts


def constraint_series(trials):
    series = {}
    for method, group in trials.items():
        magnitude = np.full((len(group), MAX_STEPS + 1), np.nan)
        peak_so_far = np.zeros_like(magnitude)
        for row, trial in enumerate(group):
            values = trial.violation_mm
            magnitude[row, :len(values)] = values
            peaks = np.maximum.accumulate(values)
            peak_so_far[row, :len(values)] = peaks
            # The running peak persists; instantaneous physical magnitudes
            # remain unavailable after a recorded rollout ends.
            peak_so_far[row, len(values):] = peaks[-1]
        series[method] = {"magnitude_mm": magnitude, "peak_so_far_mm": peak_so_far}
    return series


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dddddd", linewidth=.6)
    axis.set_axisbelow(True)


def method_handles():
    return [Line2D([], [], color=COLORS[method], linewidth=2.8,
                   label=LABELS[method]) for method in METHODS]


def draw_progress(axes, times, counts):
    for axis in axes:
        style_axis(axis)
        axis.set_xlim(0, MAX_STEPS * DT)
    for method in METHODS:
        color = COLORS[method]
        for row in counts[method]:
            axes[0].step(times, row, where="post", color=color, alpha=.19, linewidth=.8)
        axes[0].step(times, counts[method].mean(axis=0), where="post", color=color, linewidth=2.4)
        completed_percent = 100 * (counts[method] == TARGET_COUNT).mean(axis=0)
        axes[1].step(times, completed_percent, where="post", color=color, linewidth=2.4)
    axes[0].set(ylabel="References reached", yticks=np.arange(6), ylim=(-.12, 5.16))
    axes[0].tick_params(labelbottom=False)
    axes[1].set(xlabel="Time (s)", ylabel="All references reached (%)",
                yticks=np.arange(0, 101, 20), ylim=(-3, 104))


def draw_violation_rate(axis, times, constraints):
    style_axis(axis)
    for method in METHODS:
        percentage = 100 * (constraints[method]["peak_so_far_mm"] > 0).mean(axis=0)
        axis.step(times, percentage, where="post", color=COLORS[method], linewidth=2.4)
    axis.set(xlabel="Time (s)", ylabel="Trials ever violating (%)",
             xlim=(0, MAX_STEPS * DT), ylim=(-3, 104), yticks=np.arange(0, 101, 20))


def draw_violation_magnitude(axis, trials, zoom_end):
    style_axis(axis)
    peak = 0.
    for method in METHODS:
        for trial in trials[method]:
            values = trial.violation_mm
            positive = values > 0
            peak = max(peak, float(values.max()))
            axis.plot(trial.time, values, color=COLORS[method], linewidth=1., alpha=.6)
            axis.scatter(trial.time[positive], values[positive], color=COLORS[method],
                         marker="x", s=19, linewidths=1., zorder=3)
    axis.set(xlabel="Time (s)", ylabel="Violation (mm)", xlim=(0, zoom_end),
             ylim=(-.025 * max(peak, 1), 1.08 * max(peak, 1)))


def violation_zoom_end(trials):
    event_times = [trial.time[trial.violation_mm > 0]
                   for group in trials.values() for trial in group]
    latest = max((float(times[-1]) for times in event_times if len(times)), default=0.)
    return min(MAX_STEPS * DT, max(.3, np.ceil((latest + .04) * 10) / 10))


def pose_indices(trial):
    shared = [int(round(time / DT)) for time in POSE_TIMES if time < trial.time[-1]]
    return sorted(set(shared + [trial.steps]))


def draw_trajectories(axes, pair):
    for axis, method in zip(axes, METHODS):
        trial = pair[method]
        color = COLORS[method]
        axis.axhspan(HEIGHT_LIMIT, .23, color="#eeeeee", zorder=0)
        axis.axhline(HEIGHT_LIMIT, color="#aaaaaa", linewidth=.7, zorder=1)
        axis.plot(trial.points[:, -1, 0], trial.points[:, -1, 1], linestyle="--",
                  color=color, linewidth=1.15, alpha=.8, zorder=2)
        violating_points = trial.points[trial.violation_mm > 0, -1]
        axis.scatter(violating_points[:, 0], violating_points[:, 1], marker="x",
                     color="#c62828", s=23, linewidths=1.2, zorder=8)
        for index in pose_indices(trial):
            alpha = .25 + .65 * trial.time[index] / (MAX_STEPS * DT)
            points = trial.points[index]
            axis.plot(points[:, 0], points[:, 1], color=color, alpha=alpha,
                      linewidth=4., solid_capstyle="round", zorder=3)
            axis.scatter(points[:, 0], points[:, 1], s=[15, 15, 11],
                         color=color, alpha=alpha, zorder=4)
        for number, target in enumerate(trial.targets, start=1):
            axis.add_patch(Circle(target, TARGET_TOLERANCE, facecolor="white",
                                  edgecolor="#555555", linewidth=.75, zorder=5))
            axis.plot(*target, marker=".", markersize=2, color="#444444", zorder=6)
            offset = (-12, 5) if number == 1 else (5, 5)
            axis.annotate(str(number), target, xytext=offset, textcoords="offset points",
                          fontsize=9, color="#333333", zorder=7)
        axis.scatter(*trial.points[0, 0], s=19, color="#555555", zorder=6)
        axis.set(xlim=(-.23, .23), ylim=(-.23, .23), aspect="equal",
                 ylabel="y (m)", title=LABELS[method],
                 xticks=[-.2, 0, .2], yticks=[-.2, 0, .2])
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(LABELS[method], fontsize=11, pad=5)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("x (m)")


def save_figure(figure, output, name):
    for extension in ("pdf", "png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(figure)


def make_figures(trials, times, counts, constraints, pair, output):
    figure = plt.figure(figsize=(10, 6.2))
    grid = figure.add_gridspec(2, 2, width_ratios=[1, 1.5], left=.07, right=.985,
                              bottom=.10, top=.91, wspace=.36, hspace=.25)
    trajectory_axes = [figure.add_subplot(grid[row, 0]) for row in range(2)]
    progress_axes = [figure.add_subplot(grid[row, 1]) for row in range(2)]
    draw_trajectories(trajectory_axes, pair)
    draw_progress(progress_axes, times, counts)
    progress_axes[0].legend(handles=method_handles(), loc="lower center",
                            bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False, fontsize=11)
    save_figure(figure, output, "reacher_combined")

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 4.9), sharex=True)
    figure.subplots_adjust(left=.15, right=.98, bottom=.12, top=.89, hspace=.20)
    draw_progress(axes, times, counts)
    figure.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=(.55, .99),
                  ncol=2, frameon=False, fontsize=11)
    save_figure(figure, output, "target_progress")

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.2))
    figure.subplots_adjust(left=.15, right=.98, bottom=.11, top=.89, hspace=.38)
    draw_violation_magnitude(axes[0], trials, violation_zoom_end(trials))
    draw_violation_rate(axes[1], times, constraints)
    figure.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=(.55, .99),
                  ncol=2, frameon=False, fontsize=11)
    save_figure(figure, output, "constraint_violations")

    figure, axes = plt.subplots(2, 1, figsize=(4.4, 6.2), sharex=True, sharey=True)
    figure.subplots_adjust(left=.16, right=.98, bottom=.10, top=.93, hspace=.25)
    draw_trajectories(axes, pair)
    save_figure(figure, output, "trajectory_comparison")


def export_data(trials, times, counts, output):
    with (output / "target_events.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "seed", "target_number", "target_x_m", "target_y_m",
                         "reached", "completion_step", "completion_time_s", "hit_distance_m"])
        for method, group in trials.items():
            for trial in group:
                for index, target in enumerate(trial.targets):
                    reached = index < len(trial.completion_steps)
                    step = int(trial.completion_steps[index]) if reached else None
                    distance = np.linalg.norm(trial.observations[step, 4:6] - target) if reached else ""
                    writer.writerow([method, trial.seed, index + 1, *target, reached,
                                     step if reached else "", step * DT if reached else "", distance])
    with (output / "progress_trials.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "seed", "time_s", "targets_reached", "all_five_completed",
                         "state_recorded_at_this_time"])
        for method, group in trials.items():
            for trial, row in zip(group, counts[method]):
                for step, (time, count) in enumerate(zip(times, row)):
                    writer.writerow([method, trial.seed, time, count, count == TARGET_COUNT,
                                     step <= trial.steps])
    with (output / "progress_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "time_s", "n_trials", "mean_targets_reached",
                         "sample_sd_targets_reached", "all_five_completed", "all_five_completed_percent"])
        for method, values in counts.items():
            for index, time in enumerate(times):
                column = values[:, index]
                completed = int(np.count_nonzero(column == TARGET_COUNT))
                writer.writerow([method, time, len(column), column.mean(), column.std(ddof=1),
                                 completed, 100 * completed / len(column)])


def export_constraints(trials, times, constraints, output):
    with (output / "violation_trials.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "seed", "time_s", "state_recorded", "violation_mm",
                         "peak_violation_so_far_mm", "ever_violated"])
        for method, group in trials.items():
            for row, trial in enumerate(group):
                for step, time in enumerate(times):
                    value = constraints[method]["magnitude_mm"][row, step]
                    peak = constraints[method]["peak_so_far_mm"][row, step]
                    writer.writerow([method, trial.seed, time, np.isfinite(value),
                                     value if np.isfinite(value) else "", peak, peak > 0])
    with (output / "violation_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "time_s", "n_trials", "n_states_recorded",
                         "trials_ever_violated", "trials_ever_violated_percent", "max_violation_so_far_mm"])
        for method, series in constraints.items():
            for index, time in enumerate(times):
                peaks = series["peak_so_far_mm"][:, index]
                violated = np.count_nonzero(peaks > 0)
                observed = np.count_nonzero(np.isfinite(series["magnitude_mm"][:, index]))
                writer.writerow([method, time, len(peaks), observed, violated,
                                 100 * violated / len(peaks), peaks.max()])
    with (output / "violation_by_start.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "seed", "post_action_samples", "violating_samples",
                         "peak_violation_mm", "first_violation_time_s", "last_violation_time_s",
                         "target_completion_success", "strict_clean_success"])
        for method, group in trials.items():
            for trial in group:
                events = np.flatnonzero(trial.violation_mm > 0)
                success = bool(trial.result["success"])
                writer.writerow([method, trial.seed, trial.steps, len(events), trial.violation_mm.max(),
                                 events[0] * DT if len(events) else "",
                                 events[-1] * DT if len(events) else "", success, success and not len(events)])


def write_documentation(root, trials, pair, output):
    selected = pair[METHODS[0]].seed
    outcomes = {method: {"successes": sum(bool(trial.result["success"]) for trial in group),
                         "final_mean_targets_reached": sum(len(trial.completion_steps) for trial in group) / len(group),
                         "trials_with_violations": sum(bool(np.any(trial.violation_mm > 0)) for trial in group),
                         "violating_samples": sum(int(np.count_nonzero(trial.violation_mm > 0)) for trial in group),
                         "maximum_violation_mm": max(float(trial.violation_mm.max()) for trial in group)}
                for method, group in trials.items()}
    text = f"""# Reacher comparison

![Combined figure](reacher_combined.png)

[Combined PDF](reacher_combined.pdf) · [Combined SVG](reacher_combined.svg) · [Violation PDF](constraint_violations.pdf) · [Progress PDF](target_progress.pdf) · [Trajectory PDF](trajectory_comparison.pdf)

The figure uses ten paired starts, five ordered targets per start, 5 mm strict fingertip-distance threshold, and 10 s budget. A run stops immediately when the fifth target is reached. Green is BD-χ²; orange is Select-DPC. Figures use Times New Roman when available, otherwise DejaVu Serif.

**Progress.** Top right: thin step curves show each trial's cumulative number of targets reached; thick curves show the arithmetic mean over all ten trials. Bottom right: the percentage of all ten trials that have completed the fifth target by that time. Its final values are {outcomes['bd_chi2']['successes'] * 10}% and {outcomes['select_dpc']['successes'] * 10}%, respectively. Mean final counts are {outcomes['bd_chi2']['final_mean_targets_reached']:g} and {outcomes['select_dpc']['final_mean_targets_reached']:g}/5. These are empirical summaries, not confidence intervals. Targets differ across starts, but paired methods receive identical ordered target lists and initial states. Progress compares task completion without averaging distances to different active targets.

Both progress panels retain all ten trials throughout 0–10 s: completed events remain completed after a run stops. This does not extend physical trajectories or impute tracking errors. Failed runs retain only targets actually reached; they are never dropped from the denominator. Event times use the saved post-transition state at which the ordered target is first within 5 mm.

**Violations.** The separate [violation figure](constraint_violations.pdf), lower panel: the percentage of all ten trials with at least one recorded fingertip y > 0.1 m sample by time t. This is cumulative incidence: once a trial violates, it stays counted even after recovery or termination. It is not the percentage currently violating. The denominator is always ten, ending at **{outcomes['bd_chi2']['trials_with_violations'] * 10}% for BD-χ²** and **{outcomes['select_dpc']['trials_with_violations'] * 10}% for Select-DPC**. BD has {outcomes['bd_chi2']['violating_samples']} violating samples, with maximum excess **{outcomes['bd_chi2']['maximum_violation_mm']:.5f} mm**; Select has {outcomes['select_dpc']['violating_samples']}. There is no numerical or 1 mm tolerance in this definition. These statistics describe recorded control instants, not a continuous-time feasibility certificate.

The companion [violation figure](constraint_violations.pdf) shows individual instantaneous excesses **1000 max(0, y_tip − 0.1) mm**, with crosses at violating samples, above the same cumulative incidence curves. Its top panel zooms to **0–{violation_zoom_end(trials):.2f} s**, chosen to include every observed excursion plus context. In these records all violations occur at 0.12–0.22 s; every trial is still recorded throughout the zoom. No averaging or median hides isolated violations. Physical magnitude traces stop at each actual endpoint; no zero is inserted after termination. The selected trajectory seed {selected} has no violations in either method, while this aggregate view includes every start. Red crosses mark violating fingertip samples in a trajectory panel if present.

**Trajectory selection.** The default is the lowest shared seed with BD-χ² completion and Select-DPC noncompletion, here **{selected}**. BD-χ² reaches {len(pair['bd_chi2'].completion_steps)}/5 references and ends at **{pair['bd_chi2'].steps * DT:.2f} s**; Select-DPC reaches {len(pair['select_dpc'].completion_steps)}/5 and ends at **{pair['select_dpc'].steps * DT:.2f} s**. This is an illustrative pair, not a representative sample. Both panels use identical limits and equal physical scaling. Dashed paths follow recorded fingertip positions; solid arm poses show 0, 0.8, 2.0 s and each actual endpoint. Fainter poses are earlier on a shared 10 s clock. Numbered circles mark the five references in visitation order with radius 5 mm.

**Marked region.** Gray denotes fingertip y > 0.1 m, a task constraint on the fingertip only, not a physical obstacle or whole-arm collision region. The demonstration MPC constrains predicted fingertip height. Exploratory action noise is added afterward, so the demonstrations are not guaranteed strictly feasible. The environment does not terminate a rollout when this boundary is crossed; constraint violations and five-target completion are separate recorded metrics. Benchmark success requires both five-reference completion and no recorded height violation.

Regenerate from the project root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/plots/reacher.py
```

`--seed`, `--root` and `--output` override the selected pair and paths; `PAPER_FONT` chooses another installed font. CSV outputs contain reference coordinates and completion events, per-trial progress, aggregate completion statistics, and instantaneous and cumulative constraint violations.
"""
    (output / "README.md").write_text(text)
    for path, expected in INPUT_HASHES.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Input changed while plotting: {path}")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_root=str(root),
                    script_path=str(Path(__file__).resolve()), script_sha256=digest(Path(__file__)), input_sha256=INPUT_HASHES,
                    selected_seed=selected, font_family=FONT_FAMILY, colors=COLORS,
                    timestep_s=DT, budget_s=MAX_STEPS * DT,
                    target_tolerance_m=TARGET_TOLERANCE, target_count=TARGET_COUNT,
                    seeds=[trial.seed for trial in trials[METHODS[0]]],
                    progress="Cumulative ordered target hits; fixed denominator of ten trials; event counts persist after stopping",
                    no_physical_trajectory_padding=True, constraint_applies_to="fingertip y only",
                    violations=dict(threshold_m=HEIGHT_LIMIT, comparison="strict >; no tolerance",
                                    samples="Post-action control instants; all initial states checked feasible",
                                    magnitude_mm="1000 max(0, fingertip_y - height_limit)",
                                    figure="constraint_violations",
                                    lower_panel="Percentage of all ten trials with at least one violation by time t",
                                    fixed_denominator=10, carry_historical_events_after_stop=True,
                                    pad_physical_magnitude_after_stop=False,
                                    magnitude_zoom_s=[0., violation_zoom_end(trials)]),
                    trajectory=dict(path="saved world fingertip", line_style="dashed", physical_aspect="equal",
                                    shared_pose_times_s=list(POSE_TIMES), include_terminal_states=True,
                                    alpha_range=[.25, .9], common_time_scale_s=[0., MAX_STEPS * DT],
                                    displayed_pose_times_s={method: [index * DT for index in pose_indices(trial)]
                                                            for method, trial in pair.items()}),
                    outcomes=outcomes, simulations_run=0, numerical_controller_calls=0,
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
    output = (args.output or root / "figures/reacher").resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted((Path.home() / ".local/share/fonts/msttcorefonts").glob("*.TTF")):
        font_manager.fontManager.addfont(str(pin(path)))
    try:
        pin(Path(font_manager.findfont(FONT_FAMILY, fallback_to_default=False)))
    except ValueError as error:
        raise RuntimeError("Install Times New Roman before generating these figures") from error
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": 10, "axes.labelsize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    trials = load_trials(root)
    pair = choose_pair(trials, args.seed)
    times, counts = progress(trials)
    constraints = constraint_series(trials)
    make_figures(trials, times, counts, constraints, pair, output)
    export_data(trials, times, counts, output)
    export_constraints(trials, times, constraints, output)
    write_documentation(root, trials, pair, output)
    print(json.dumps({"output": str(output), "selected_seed": pair[METHODS[0]].seed,
                      "paired_starts": 10, "simulations_run": 0}), flush=True)


if __name__ == "__main__":
    main()
