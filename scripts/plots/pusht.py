"""Make the Push-T paper figures using saved final-test records only.

Run from the repository root:
    .venv-pusht/bin/python scripts/plots/pusht.py

No policy or simulator is imported. Instantaneous coverage is never extended
past a recorded endpoint; the median/IQR use only available observations.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.plots.common import PAPER_RESULTS, REPOSITORY, paper_font, replay_input_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
import numpy as np
from shapely.geometry import Polygon as GeometryPolygon
from shapely.ops import unary_union


METHODS = ("bd_chi2", "select_dpc")
LABELS = {"bd_chi2": "BD-χ²", "select_dpc": "Select-DPC"}
COLORS = {"bd_chi2": "#2ca02c", "select_dpc": "#ff7f0e"}
FONT_FAMILY = paper_font()
COVERAGE_FONT_SCALE = 2.5
BASE_LEFT_SEEDS = (300013, 300021, 300052)
BASE_RIGHT_SEEDS = (300015, 300023, 300053)
LEFT_SEEDS = (300013, 300021, 300052, 300044, 300022)
RIGHT_SEEDS = (300015, 300023, 300053, 300028, 300051)
FIGURE_SIZE = (6.8, 8.2)
DT, HORIZON, THRESHOLD = 0.1, 30.0, 0.95
INPUT_HASHES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    path = path.resolve()
    value = digest(path)
    if INPUT_HASHES.setdefault(str(path), value) != value:
        raise ValueError(f"Input changed during plotting: {path}")
    return path


def read_json(path):
    return json.loads(pin(path).read_text())


def transform(vertices, pose):
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return np.asarray(vertices) @ rotation.T + pose[:2]


@dataclass
class Trial:
    method: str
    seed: int
    time: np.ndarray
    states: np.ndarray
    coverage: np.ndarray
    goal_pose: np.ndarray
    geometry: dict
    outline: np.ndarray
    result: dict
    geometry_error: float

    @property
    def centers(self):
        # Recorded position is the Pymunk body origin, not its center of mass.
        center = np.asarray(self.geometry["block_center_of_gravity"])
        return np.array([transform(center, pose) for pose in self.states[:, 2:]])


def load_trial(directory, method, seed, protocol):
    metadata = read_json(directory / "metadata.json")
    if protocol.get("release_protocol"):
        hashes = metadata["payload_sha256"]
        filenames = ("trajectory.npz", "result.json")
    else:
        hashes = read_json(directory / "record.json")["sha256"]
        filenames = ("trajectory.npz", "result.json", "metadata.json")
    for filename in filenames:
        if digest(pin(directory / filename)) != hashes[filename]:
            raise ValueError(f"Record hash mismatch: {directory / filename}")
    result = read_json(directory / "result.json")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in
                  ("states", "time", "coverage", "goal_pose", "actions", "observations", "rewards")}
    count = result["steps"]
    geometry = metadata["geometry"]
    shapes = {"states": (count + 1, 5), "time": (count + 1,), "coverage": (count + 1,),
              "goal_pose": (3,), "actions": (count, 2), "observations": (count + 1, 6), "rewards": (count,)}
    if any(arrays[name].shape != shape or not np.isfinite(arrays[name]).all()
           for name, shape in shapes.items()):
        raise ValueError(f"Malformed saved arrays: {directory}")
    if protocol.get("release_protocol"):
        expected_methods = {entry["id"]: entry for entry in protocol["methods"]}
        protocol_matches = (metadata["protocol_sha256"] == protocol["sha256"]
                            and metadata["method"] == expected_methods[method]
                            and protocol["paper_protocol"])
    else:
        protocol_matches = (metadata["phase"] == "final"
                            and metadata["protocol_sha256"] == protocol["parent_protocol_sha256"]
                            and metadata["method"] == protocol["selected_candidates"][method]["method"])
    if (result["method"] != method or result["seed"] != seed or not protocol_matches
            or geometry["control_hz"] != 10 or geometry["success_threshold"] != THRESHOLD
            or not np.array_equal(arrays["goal_pose"], geometry["goal_pose"])):
        raise ValueError(f"Unexpected method or task protocol: {directory}")
    np.testing.assert_allclose(arrays["time"], np.arange(count + 1) * DT, atol=1e-12, rtol=0)
    coverage = arrays["coverage"]
    if np.any(coverage < 0) or np.any(coverage > 1 + 1e-12):
        raise ValueError(f"Coverage outside [0, 1]: {directory}")
    success = bool(np.any(coverage[1:] > THRESHOLD))
    if (success != result["success"] or abs(result["max_coverage"] - coverage[1:].max()) > 1e-12
            or abs(result["final_coverage"] - coverage[-1]) > 1e-12
            or (success and (coverage[-1] <= THRESHOLD or np.any(coverage[1:-1] > THRESHOLD)))
            or (not success and (count != 300 or result["termination_reason"] != "timeout"))):
        raise ValueError(f"Coverage/outcome mismatch: {directory}")
    local_shape = unary_union([GeometryPolygon(vertices) for vertices in geometry["block_polygons"]])
    if local_shape.geom_type != "Polygon" or local_shape.interiors:
        raise ValueError("Expected a single T-shaped outline without holes")
    outline = np.asarray(local_shape.exterior.coords)
    goal = GeometryPolygon(transform(outline, arrays["goal_pose"]))
    error = 0.0
    for index in sorted({0, count // 2, count}):
        block = GeometryPolygon(transform(outline, arrays["states"][index, 2:]))
        error = max(error, abs(goal.intersection(block).area / goal.area - coverage[index]))
    if error > 1e-12:
        raise ValueError(f"Saved block geometry disagrees with coverage: {directory}")
    return Trial(method, seed, arrays["time"], arrays["states"], coverage,
                 arrays["goal_pose"], geometry, outline, result, error)


def load_trials(root):
    protocol = load_protocol(root)
    seeds = protocol["test_seeds"]
    if seeds != list(range(300005, 300055)):
        raise ValueError("Expected the frozen 50-start final evaluation")
    trials = {}
    for method in METHODS:
        directory = root / "pusht" / method
        if sorted(int(path.name.removeprefix("seed_")) for path in directory.glob("seed_*")) != seeds:
            raise ValueError(f"Missing or unexpected final records: {method}")
        trials[method] = [load_trial(directory / f"seed_{seed}", method, seed, protocol) for seed in seeds]
    for first, second in zip(trials[METHODS[0]], trials[METHODS[1]]):
        np.testing.assert_array_equal(first.states[0], second.states[0])
        if first.geometry != second.geometry:
            raise ValueError(f"Paired task geometry differs: {first.seed}")
    return trials


def load_protocol(root):
    released = root / "protocols/pusht.json"
    if released.exists():
        protocol = read_json(released)
        protocol["release_protocol"] = True
        protocol["test_seeds"] = [int(start[0]) for start in protocol["test_starts"]]
        return protocol
    return read_json(root / "pusht/protocol.json")


def summarize(trials):
    times = np.arange(round(HORIZON / DT) + 1) * DT
    summaries = {}
    for method, group in trials.items():
        values = np.full((len(group), len(times)), np.nan)
        for row, trial in enumerate(group):
            values[row, :len(trial.time)] = trial.coverage
        summaries[method] = dict(
            count=np.isfinite(values).sum(axis=0), median=np.nanmedian(values, axis=0),
            q25=np.nanquantile(values, .25, axis=0, method="linear"),
            q75=np.nanquantile(values, .75, axis=0, method="linear"),
            mean=np.nanmean(values, axis=0), sample_sd=np.nanstd(values, axis=0, ddof=1))
    return times, summaries


def configure_fonts():
    for path in sorted((Path.home() / ".local/share/fonts/msttcorefonts").glob("*.TTF")):
        font_manager.fontManager.addfont(str(path))
    pin(Path(font_manager.findfont(FONT_FAMILY, fallback_to_default=False)))
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": 10, "axes.labelsize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def method_handles():
    return [Line2D([], [], color=COLORS[method], linewidth=2.6, label=LABELS[method]) for method in METHODS]


def save_figure(figure, output, name):
    for extension in ("pdf", "svg", "png"):
        figure.savefig(output / f"{name}.{extension}", dpi=240, facecolor="white")
    plt.close(figure)


def draw_coverage(trials, times, summaries, output, font_scale=COVERAGE_FONT_SCALE):
    figure, axis = plt.subplots(figsize=(8, 3.8))
    figure.subplots_adjust(left=.19, right=.96, bottom=.26, top=.79)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dddddd", linewidth=.6)
    axis.set_axisbelow(True)
    axis.axhline(100 * THRESHOLD, color="#999999", linestyle=":", linewidth=.9, zorder=1)
    for method in METHODS:
        color = COLORS[method]
        for trial in trials[method]:
            axis.plot(trial.time, 100 * trial.coverage, color=color, alpha=.17, linewidth=.65, zorder=2)
            axis.plot(trial.time[-1], 100 * trial.coverage[-1],
                      marker="o" if trial.result["success"] else "x", color=color,
                      markerfacecolor="none", markersize=3.5, markeredgewidth=.8, clip_on=False, zorder=4)
        summary = summaries[method]
        axis.fill_between(times, 100 * summary["q25"], 100 * summary["q75"],
                          color=color, alpha=.18, linewidth=0, zorder=1)
        complete = summary["count"] == len(trials[method])
        axis.plot(times, np.where(complete, 100 * summary["median"], np.nan),
                  color=color, linewidth=2.4, zorder=3)
        conditional = ~complete
        if conditional.any():
            conditional[max(0, np.flatnonzero(conditional)[0] - 1)] = True
            axis.plot(times, np.where(conditional, 100 * summary["median"], np.nan),
                      color=color, linewidth=2.4, linestyle="--", zorder=3)
    axis.set(xlabel="Time (s)", ylabel="Coverage (%)", xlim=(0, HORIZON), ylim=(-1, 101),
             xticks=np.arange(0, 31, 5), yticks=np.arange(0, 101, 20))
    axis.xaxis.label.set_fontsize(11 * font_scale)
    axis.yaxis.label.set_fontsize(11 * font_scale)
    axis.tick_params(axis="both", labelsize=10 * font_scale)
    figure.legend(handles=method_handles(), loc="upper center", ncol=2, frameon=False,
                  bbox_to_anchor=(.575, 1.02), fontsize=11 * font_scale)
    save_figure(figure, output, "coverage_comparison")



def world_outlines(trial):
    return np.array([transform(trial.outline, pose) for pose in trial.states[:, 2:]])


def shape_distance(first, second):
    # The repeated closing vertex is not a second material point.
    return float(np.sqrt(np.mean(np.sum((first[:-1] - second[:-1]) ** 2, axis=1))))


def verify_start_selection(candidates):
    """Reproduce diversity selection using initial block geometry and BD outcome."""
    outlines = {seed: transform(trial.outline, trial.states[0, 2:])
                for seed, trial in candidates.items()}
    additions = []
    for success, original, expected in ((True, BASE_LEFT_SEEDS, LEFT_SEEDS),
                                         (False, BASE_RIGHT_SEEDS, RIGHT_SEEDS)):
        selected = list(original)
        for _ in range(2):
            distances = {
                seed: min(shape_distance(outlines[seed], outlines[chosen]) for chosen in selected)
                for seed, trial in candidates.items()
                if trial.result["success"] == success and seed not in selected}
            seed = max(distances, key=lambda item: (distances[item], -item))
            state = candidates[seed].states[0]
            additions.append(dict(
                seed=seed, bd_success=success, already_selected=list(selected),
                minimum_block_pose_distance_pixel=distances[seed],
                minimum_pusher_distance_pixel=min(float(np.linalg.norm(state[:2] - candidates[other].states[0, :2]))
                                                  for other in selected),
                initial_state=state.tolist(),
                wrapped_block_angle_deg=float(np.rad2deg(np.arctan2(np.sin(state[4]), np.cos(state[4]))))))
            selected.append(seed)
        if tuple(selected) != expected:
            raise ValueError(f"Initial-pose diversity selection no longer matches frozen seeds: {selected}")
    return dict(
        rule="Greedy maximin separately within BD success/failure groups; lower seed breaks ties.",
        distance="RMS displacement of corresponding initial world-space T-outline vertices, excluding closing duplicate.",
        original_success_seeds=list(BASE_LEFT_SEEDS), original_failure_seeds=list(BASE_RIGHT_SEEDS),
        final_success_seeds=list(LEFT_SEEDS), final_failure_seeds=list(RIGHT_SEEDS), additions=additions,
        outcomes_used="Binary BD success only; no subsequent trajectory, cost, or coverage enters the diversity score.")


def pose_indices(outlines):
    increments = [shape_distance(first, second) for first, second in zip(outlines[:-1], outlines[1:])]
    progress = np.r_[0., np.cumsum(increments)]
    endpoint = len(outlines) - 1
    indices = [0]
    # Small/no-motion traces show only the real initial and final poses.
    if progress[-1] >= 36:
        for fraction in (1 / 3, 2 / 3):
            index = int(np.searchsorted(progress, fraction * progress[-1]))
            separated = all(shape_distance(outlines[index], outlines[other]) >= 12
                            for other in [*indices, endpoint])
            if 0 < index < endpoint and separated:
                indices.append(index)
    indices.append(endpoint)
    return indices, progress


def pair_bounds(pair, outlines):
    points = [shape.reshape(-1, 2) for shape in outlines.values()]
    first = pair[METHODS[0]]
    points.append(transform(first.outline, first.goal_pose))
    points = np.vstack(points)
    lower, upper = points.min(axis=0), points.max(axis=0)
    side = max(240., float(np.max(upper - lower)) + 32.)
    minimum = (lower + upper) / 2 - side / 2
    # Keep the crop in the workspace when this does not hide any block poses.
    if side <= 512 and np.all(lower >= 0) and np.all(upper <= 512):
        minimum = np.clip(minimum, 0., 512. - side)
    maximum = minimum + side
    if np.any(points < minimum - 1e-9) or np.any(points > maximum + 1e-9):
        raise ValueError("Pair crop would truncate a recorded block or the reference")
    return minimum, maximum


def draw_direction_arrows(axis, centers, color):
    progress = np.r_[0., np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))]
    arrows = []
    if progress[-1] < 20:
        return arrows
    for fraction in (.35, .75):
        position = fraction * progress[-1]
        half_span = min(7., .08 * progress[-1])
        first = int(np.searchsorted(progress, position - half_span))
        last = min(len(centers) - 1, int(np.searchsorted(progress, position + half_span)))
        if np.linalg.norm(centers[last] - centers[first]) < 4:
            continue
        axis.annotate("", xy=centers[last], xytext=centers[first],
                      arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.5,
                                      mutation_scale=9.5, shrinkA=0, shrinkB=0), zorder=8)
        arrows.append([first, last])
    return arrows


def draw_trial(axis, trial, outlines, minimum, maximum):
    color = COLORS[trial.method]
    goal = transform(trial.outline, trial.goal_pose)
    axis.add_patch(Polygon(goal, facecolor="#e8e8e8", edgecolor="#aaaaaa",
                           linestyle=":", linewidth=.8, zorder=1))
    indices, progress = pose_indices(outlines)
    pose_rows = []
    for order, index in enumerate(indices):
        initial, terminal = index == 0, index == len(outlines) - 1
        alpha = .7 if initial else .95 if terminal else .25 + .35 * progress[index] / progress[-1]
        edge = "#777777" if initial else color
        axis.add_patch(Polygon(outlines[index], facecolor="none" if not terminal else to_rgba(color, .035),
                               edgecolor=to_rgba(edge, alpha), linestyle="--" if initial else "-",
                               linewidth=1.35 if terminal else .8,
                               zorder=6 if terminal else 2 + order))
        pose_rows.append([trial.method, trial.seed, index, trial.time[index], initial, terminal,
                          progress[index], alpha, *trial.states[index, 2:]])
    centers = trial.centers
    axis.plot(centers[:, 0], centers[:, 1], color=color, linewidth=1.2, zorder=5)
    axis.plot(*centers[0], marker="o", color=color, markerfacecolor="white",
              markeredgewidth=.8, markersize=3.4, zorder=9)
    axis.plot(*centers[-1], marker="o", color=color, markersize=2.2, zorder=10)
    arrows = draw_direction_arrows(axis, centers, color)
    axis.set(xlim=(minimum[0], maximum[0]), ylim=(maximum[1], minimum[1]), aspect="equal")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10], integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10], integer=True))
    axis.tick_params(labelsize=7, width=.5, length=2, pad=2)
    for spine in axis.spines.values():
        spine.set_color("#bbbbbb")
        spine.set_linewidth(.5)
    return pose_rows, arrows, progress


def draw_trajectories(trials, output):
    figure = plt.figure(figsize=FIGURE_SIZE)
    grid = figure.add_gridspec(5, 5, width_ratios=[1, 1, .12, 1, 1],
                              left=.065, right=.99, bottom=.055, top=.95, hspace=.20, wspace=.14)
    pose_rows, arrow_rows, panel_rows = [], [], []
    for row, seeds in enumerate(zip(LEFT_SEEDS, RIGHT_SEEDS)):
        for group, seed in enumerate(seeds):
            pair = {method: trials[method, seed] for method in METHODS}
            if pair["bd_chi2"].result["success"] != (group == 0):
                raise ValueError(f"Unexpected BD outcome for the selected column: {seed}")
            outlines = {method: world_outlines(trial) for method, trial in pair.items()}
            minimum, maximum = pair_bounds(pair, outlines)
            for method_index, method in enumerate(METHODS):
                column = 3 * group + method_index
                axis = figure.add_subplot(grid[row, column])
                trial = pair[method]
                poses, arrows, progress = draw_trial(axis, trial, outlines[method], minimum, maximum)
                if method_index == 1:
                    axis.tick_params(labelleft=False)
                axis.apply_aspect()
                pose_rows.extend(poses)
                arrow_rows.extend([method, seed, first, last, trial.time[first], trial.time[last]]
                                  for first, last in arrows)
                panel_rows.append([row + 1, 2 * group + method_index + 1, method, seed,
                                   trial.result["success"], trial.time[-1], *minimum, *maximum, progress[-1]])
                if row == 0:
                    box = axis.get_position()
                    figure.text((box.x0 + box.x1) / 2, .99, LABELS[method],
                                ha="center", va="top", fontsize=10, color=COLORS[method])
    figure.supxlabel("x (pixel)", fontsize=10, y=.018)
    figure.supylabel("y (pixel)", fontsize=10, x=.008)
    save_figure(figure, output, "trajectory_comparison")
    return pose_rows, arrow_rows, panel_rows


def write_trajectory_records(output, pose_rows, arrow_rows, panel_rows, root, selection):
    (output / "start_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    write_csv(output / "displayed_poses.csv",
                     ["method", "seed", "frame", "time_s", "initial", "terminal", "cumulative_shape_motion_px",
                      "edge_alpha", "origin_x_px", "origin_y_px", "angle_rad"], pose_rows)
    write_csv(output / "direction_arrows.csv",
                     ["method", "seed", "first_frame", "last_frame", "first_time_s", "last_time_s"], arrow_rows)
    write_csv(output / "panels.csv",
                     ["row", "column", "method", "seed", "success", "duration_s", "x_min_px", "y_min_px",
                      "x_max_px", "y_max_px", "total_shape_motion_px"], panel_rows)
    notes = """# Push-T trajectories: separate methods

[PDF](trajectory_comparison.pdf) · [SVG](trajectory_comparison.svg) · [PNG](trajectory_comparison.png)

![Separate-method trajectories](trajectory_comparison.png)

The 5×4 grid fits within one paper page (native size 6.8×8.2 inches). Columns
are BD-χ², Select-DPC, BD-χ², Select-DPC. The left pairs use BD-success seeds
300013, 300021, 300052, 300044, 300022; the right pairs use BD-failure seeds
300015, 300023, 300053, 300028, 300051, top to bottom. All ten Select-DPC examples fail.

The last two starts in each outcome group were chosen for initial block-pose
diversity: repeatedly select the candidate whose minimum RMS T-vertex distance
from already selected starts is largest. This considers translation and
rotation jointly and breaks ties by lower seed. Pusher positions also differ,
but do not enter the selection score. Only the initial states and binary BD
outcomes enter selection, not later trajectory shape, cost, or final coverage.
`start_selection.json` lists the selection rule and initial states.
These are illustrative outcome-conditioned examples, not a representative sample.

## Short caption

Push-T trajectories for five BD-χ² successes (left pairs) and five failures
(right pairs), each compared with Select-DPC. Green and orange denote BD-χ² and
Select-DPC. Solid curves and arrows show block-center motion; open and filled
circles mark initial and final positions. Gray dashed outlines indicate initial
blocks; gray filled shapes with dotted borders indicate reference poses. Faint
colored outlines show intermediate poses, and bold outlines show final poses.

## Caption notes

Green denotes BD-χ² and orange Select-DPC. Solid paths show the block's center
of mass, with arrows in the direction of travel. Hollow and filled center
markers denote initial and final positions. The reference T has light-gray
fill and a dotted border; the initial block has a gray dashed outline.
Intermediate block outlines are faint; the final outline is emphasized.
Pusher paths are omitted. All geometry uses saved poses, without smoothing,
controller calls, or simulator steps. Figures use Times New Roman when available,
otherwise DejaVu Serif.

At most four block poses are shown: initial, one-third and two-thirds of
cumulative motion, and the actual last recorded state. Motion is the RMS
displacement of corresponding polygon vertices, which includes rotation as
well as translation. Intermediate poses are omitted below 36 pixels of total
shape motion, or when they lie within 12 pixels RMS of another displayed pose.
Opacity orders the intermediate poses by motion progress, not elapsed time.
The last recorded frame is always used, including when the block stopped moving
well before timeout. Exact pose times are in `displayed_poses.csv`.

Arrows are omitted below 20 pixels of cumulative center travel or for an arrow
span shorter than 4 pixels. Select-DPC seed 300021 is exactly stationary; seeds
300023 and 300053 also move the block very little. Overlapping initial/final
poses in these panels faithfully show that lack of motion.

Both methods in each seed pair use the same square crop and equal physical
aspect. Crops include every recorded block pose from both methods plus the
reference; different seed pairs can have different bounds. Axes retain pixel
coordinates, with y increasing downward as in the simulator. The block outline
is transformed about the saved Pymunk body origin; the local center of gravity
is transformed separately for the trajectory. `panels.csv` records all bounds.

The coverage figure remains in the parent folder. Both paper figures are
generated by the same script from the saved final records.

## Regenerate

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-pusht/bin/python \\
  scripts/plots/pusht.py
```

`scripts/plots/pusht.py` generates both coverage and trajectory figures.
`--root` and `--output` override the result and output paths.
"""
    (output / "README.md").write_text(notes)
    for name, expected in INPUT_HASHES.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Plot input changed: {name}")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_root=str(root),
                    script_path=str(Path(__file__).resolve()), script_sha256=digest(Path(__file__)),
                    input_sha256=INPUT_HASHES, rows=5, columns=4, size_inches=list(FIGURE_SIZE),
                    left_seeds=list(LEFT_SEEDS), right_seeds=list(RIGHT_SEEDS),
                    start_selection=selection,
                    methods=list(METHODS), colors=COLORS, font=FONT_FAMILY,
                    include_pusher=False, path="recorded block center of mass", path_smoothing=False,
                    path_linewidth_pt=1.2, arrow_linewidth_pt=1.5, arrow_mutation_scale=9.5,
                    seed_labels=False, axis_unit="pixel",
                    pose_selection="Initial, RMS polygon-motion thirds, actual endpoint; omit redundant intermediates",
                    common_crop="Per seed, includes both complete block trajectories and reference",
                    new_simulator_steps=0, controller_calls=0,
                    output_sha256={path.name: digest(path) for path in sorted(output.iterdir())
                                   if path.is_file() and path.name != "manifest.json"})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")



def write_csv(path, header, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)



def export_coverage_data(trials, times, summaries, output):
    write_csv(output / "coverage_summary.csv",
              ["method", "time_s", "n_recorded", "median", "q25", "q75", "mean", "sample_sd"],
              ([method, time, *(summary[name][index] for name in
                ("count", "median", "q25", "q75", "mean", "sample_sd"))]
               for method, summary in summaries.items() for index, time in enumerate(times)))
    write_csv(output / "coverage_trials.csv", ["method", "seed", "success", "time_s", "coverage"],
              ([method, trial.seed, trial.result["success"], time, value]
               for method, group in trials.items() for trial in group for time, value in zip(trial.time, trial.coverage)))


def write_coverage_documentation(root, trials, summaries, output, font_scale=COVERAGE_FONT_SCALE):
    notes = f"""# Push-T paper comparison figures

[Coverage PDF](coverage_comparison.pdf) · [Trajectory PDF](separated/trajectory_comparison.pdf)

![Coverage](coverage_comparison.png)

![Selected trajectories](separated/trajectory_comparison.png)

Both figures use green for BD-χ² and orange for Select-DPC, with Times New Roman
when available and DejaVu Serif otherwise. Coverage tick labels are {10 * font_scale:g} pt;
axis labels and legend are {11 * font_scale:g} pt. The canvas is 8×3.8 inches.
PDF, SVG, and 240-dpi PNG versions are provided. `scripts/plots/pusht.py` generates
both figures from the 50-start test results (seeds 300005–300054).

## Coverage caption

Instantaneous target coverage over 50 test trials for BD-χ² (green) and Select-DPC
(orange). Thin curves show individual trials; thick curves and shaded bands show
the pointwise median and interquartile range. Open circles mark successful
endpoints, and crosses mark timeouts. Success requires coverage above 95%,
indicated by the dotted horizontal line. Successful trials terminate immediately;
dashed medians indicate that fewer than 50 trials remain, with corresponding
bands computed from those remaining trials. No values are carried forward after
termination.

Coverage is intersection area divided by reference-T area, not best-so-far
coverage or the normalized score. The interquartile range uses empirical linear
25th–75th percentiles and is not a confidence interval. BD's first success ends
at 9.3 s. Its later dashed median/IQR are conditional on remaining recorded
trials, which can make the aggregate decline as successes leave. At 30 s the
summaries include 31 BD trials and all 50 Select-DPC trials. Final outcomes are
19/50 and 0/50 successes, respectively.

`coverage_summary.csv` retains sample counts, median, quartiles, mean, and sample
standard deviation at every time. `coverage_trials.csv` contains every recorded
individual coverage value. CSV values are fractions; the figure uses percent.

## Trajectories

The 5×4 figure, caption, selected starts, illustrated poses, direction
arrows and panel bounds are in [`separated/`](separated/README.md). The left pairs
show five BD successes and the right pairs five BD failures, each compared with
Select-DPC. Seed labels and pusher paths are omitted. The selected starts include
different initial block poses; the selection rule accompanies the trajectory figure.

## Regeneration

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-pusht/bin/python \\
  scripts/plots/pusht.py
```

`--root` selects a study and `--output` selects the comparison output root.
`--coverage-only` regenerates the coverage figure and its records without changing
the trajectory figures.
`--coverage-font-scale` scales the base 10/11 pt fonts
(default 2.5); use `--output` to keep a variant in a separate folder.
Both paper figures and their CSV data are generated in one pass.
"""
    trajectory_folder = output / "separated"
    if not trajectory_folder.exists():
        trajectory_folder = root / "figures/pusht/separated"
    trajectory_relative = Path(os.path.relpath(trajectory_folder, output)).as_posix()
    notes = notes.replace("separated/", f"{trajectory_relative}/")
    notes += (f"\nThis coverage variant can be regenerated with `--coverage-only "
              f"--coverage-font-scale {font_scale:g} --output {output}`.\n")
    (output / "README.md").write_text(notes)
    names = ["README.md", "coverage_summary.csv", "coverage_trials.csv",
             "coverage_comparison.pdf", "coverage_comparison.svg", "coverage_comparison.png"]
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), source_root=str(root),
        script_path=str(Path(__file__).resolve()), script_sha256=digest(Path(__file__)),
        input_sha256=INPUT_HASHES, font_family=FONT_FAMILY, colors=COLORS,
        coverage_font_scale=font_scale,
        coverage_font_sizes_pt={"ticks": 10 * font_scale,
                                "axis_labels": 11 * font_scale,
                                "legend": 11 * font_scale},
        left_seeds=list(LEFT_SEEDS), right_seeds=list(RIGHT_SEEDS),
        metric="instantaneous intersection area divided by reference T area",
        summary="Available-trial pointwise median and IQR; no endpoint padding",
        timestep_s=DT, horizon_s=HORIZON, success_threshold=THRESHOLD, success_comparison="strictly_greater",
        seeds=[trial.seed for trial in trials[METHODS[0]]],
        outcomes={method: {"successes": sum(trial.result["success"] for trial in group),
                           "trials": len(group), "count_at_30s": int(summaries[method]["count"][-1])}
                  for method, group in trials.items()},
        geometry_max_checked_coverage_error=max(trial.geometry_error for group in trials.values() for trial in group),
        trajectory_manifest=f"{trajectory_relative}/manifest.json", new_simulator_steps=0, controller_calls=0,
        software={"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        output_sha256={name: digest(output / name) for name in names})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=(Path(__file__).resolve().parents[2] / "results/paper"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--coverage-font-scale", type=float, default=COVERAGE_FONT_SCALE)
    args = parser.parse_args()
    if not np.isfinite(args.coverage_font_scale) or args.coverage_font_scale <= 0:
        parser.error("--coverage-font-scale must be finite and positive")
    root = args.root.resolve()
    output = (args.output or root / "figures/pusht").resolve()
    configure_fonts()
    trials = load_trials(root)
    times, summaries = summarize(trials)
    output.mkdir(parents=True, exist_ok=True)
    # Use path simplification for the coverage curves.
    plt.rcParams["path.simplify"] = True
    draw_coverage(trials, times, summaries, output, args.coverage_font_scale)
    export_coverage_data(trials, times, summaries, output)
    if args.coverage_only:
        write_coverage_documentation(root, trials, summaries, output, args.coverage_font_scale)
        print(json.dumps({"output": str(output), "paired_trials": 50,
                          "coverage_font_scale": args.coverage_font_scale,
                          "new_simulator_steps": 0}, indent=2))
        return

    trajectory_output = output / "separated"
    trajectory_output.mkdir(exist_ok=True)
    candidates = {trial.seed: trial for trial in trials["bd_chi2"]}
    selection = verify_start_selection(candidates)
    lookup = {(trial.method, trial.seed): trial for group in trials.values() for trial in group}
    plt.rcParams["path.simplify"] = False
    pose_rows, arrow_rows, panel_rows = draw_trajectories(lookup, trajectory_output)
    write_trajectory_records(trajectory_output, pose_rows, arrow_rows, panel_rows, root, selection)
    write_coverage_documentation(root, trials, summaries, output, args.coverage_font_scale)
    print(json.dumps({"output": str(output), "paired_trials": 50, "trajectory_panels": len(panel_rows),
                      "left_seeds": LEFT_SEEDS, "right_seeds": RIGHT_SEEDS, "new_simulator_steps": 0}, indent=2))


if __name__ == "__main__":
    main()
