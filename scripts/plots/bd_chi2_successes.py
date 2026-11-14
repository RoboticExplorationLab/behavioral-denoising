"""Draw four successful BD-chi2 trajectories from saved benchmark results.

Run with .venv-pusht/bin/python. This script never runs a controller or simulator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.plots.common import PAPER_RESULTS, REPOSITORY, paper_font, replay_input_path
FONT_FAMILY = paper_font()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from matplotlib.ticker import MaxNLocator
import numpy as np


METHOD = "bd_chi2"
COLOR = "#2ca02c"
FONT_SCALE = 2.
TASKS = ("cartpole", "rocket", "reacher", "pusht")
TITLES = ("Cartpole", "Rocket landing", "Reacher", "Push-T")
DEFAULT_ROOT = (Path(__file__).resolve().parents[2] / "results/paper")
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


def load_plotter(name, path):
    spec = importlib.util.spec_from_file_location(name, pin(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def select_seed(directory, excluded, require_clean=False):
    """Choose the lowest unused successful seed, with strict Reacher feasibility."""
    eligible = []
    for path in sorted(directory.glob("seed_*/result.json")):
        result = read_json(path)
        seed = int(result["seed"])
        successful = bool(result["success"])
        if require_clean:
            successful = successful and bool(result["clean_success_strict"])
        if successful and seed not in excluded:
            eligible.append(seed)
    if not eligible:
        raise ValueError(f"No unused successful trial in {directory}")
    return min(eligible)


def square_bounds(lower, upper):
    center = (np.asarray(lower) + np.asarray(upper)) / 2
    side = float(np.max(np.asarray(upper) - np.asarray(lower)))
    return center - side / 2, center + side / 2


def draw_cartpole(axis, trial, plotter):
    geometry = plotter.load_pole_geometry(trial)
    angle = trial.states[:, 1]
    tip_x = trial.states[:, 0] + geometry[-1, 0] * np.cos(angle) + geometry[-1, 2] * np.sin(angle)
    tip_y = -geometry[-1, 0] * np.sin(angle) + geometry[-1, 2] * np.cos(angle)
    axis.plot(tip_x, tip_y, color=COLOR, linewidth=1., linestyle="--", alpha=.65)
    indices = plotter.trajectory_pose_indices(trial)
    drawn_x = list(tip_x)
    for order, index in enumerate(indices):
        state = trial.states[index]
        drawn_x.extend(plotter.pose_coordinates(state, geometry)[0])
        drawn_x.extend([state[0] - .12, state[0] + .12])
        alpha = .10 + .80 * trial.time[index] / plotter.HORIZON
        plotter.draw_pose(axis, state, geometry, COLOR, alpha=alpha, zorder=2 + order)
    lower = [min(drawn_x) - .07, -.75]
    upper = [max(drawn_x) + .07, .75]
    lower, upper = square_bounds(lower, upper)
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
    axis.axhline(0, color="#aaaaaa", linewidth=.8, zorder=0)
    axis.axvline(0, color="#dddddd", linewidth=.6, linestyle=":", zorder=0)
    plotter.style_axis(axis)
    return {"pose_indices": indices, "pose_times_s": trial.time[indices].tolist(),
            "path": "Complete recorded pole-tip trajectory", "padded_geometry_bounds": [lower.tolist(), upper.tolist()]}


def draw_rocket(axis, trial, plotter):
    lower, upper = square_bounds(*plotter.scene_bounds({METHOD: trial}))
    barge = next(body for body in trial.rendering["rocket_bodies"] if body["name"] == "barge")
    barge_index = trial.bodies["body_names"].tolist().index("barge")
    for field in ("body_position", "body_angle"):
        values = trial.bodies[field][:, barge_index]
        if not np.allclose(values, values[0], atol=1e-5, rtol=0):
            raise ValueError("The landing platform moves during this record")
    for vertices in plotter.world_vertices(trial, barge, 0):
        axis.add_patch(Polygon(vertices, facecolor="#eeeeee", edgecolor="#777777", linewidth=.8))
    axis.plot(trial.states[:, 0], trial.states[:, 1], color=COLOR,
              linestyle="--", linewidth=1., alpha=.65, zorder=2)
    # Spread the four saved poses over the descent, including the actual endpoint.
    indices = np.unique(np.rint(np.linspace(0, len(trial.time) - 1, 4)).astype(int))
    for order, index in enumerate(indices):
        alpha = .25 + .65 * trial.time[index] / trial.time[-1]
        for body in trial.rendering["rocket_bodies"]:
            if body["name"] == "barge":
                continue
            for vertices in plotter.world_vertices(trial, body, index):
                axis.add_patch(Polygon(vertices, facecolor="none", edgecolor=COLOR,
                                       linewidth=1., alpha=alpha, zorder=3 + order))
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
    plotter.style_axis(axis)
    return {"pose_indices": indices.tolist(), "pose_times_s": trial.time[indices].tolist(),
            "path": "Complete recorded rocket-body origin trajectory"}


def draw_reacher(axis, trial, plotter):
    if len(trial.completion_steps) != 5 or np.any(trial.violation_mm > 0):
        raise ValueError("Selected Reacher trial is not a constraint-clean five-reference success")
    axis.axhspan(plotter.HEIGHT_LIMIT, .23, color="#eeeeee", zorder=0)
    axis.axhline(plotter.HEIGHT_LIMIT, color="#aaaaaa", linewidth=.7, zorder=1)
    axis.plot(trial.points[:, -1, 0], trial.points[:, -1, 1], color=COLOR,
              linestyle="--", linewidth=1.15, alpha=.8, zorder=2)
    indices = plotter.pose_indices(trial)
    for index in indices:
        alpha = .25 + .65 * trial.time[index] / (plotter.MAX_STEPS * plotter.DT)
        points = trial.points[index]
        axis.plot(points[:, 0], points[:, 1], color=COLOR, alpha=alpha,
                  linewidth=4., solid_capstyle="round", zorder=3)
        axis.scatter(points[:, 0], points[:, 1], s=[15, 15, 11], color=COLOR, alpha=alpha, zorder=4)
    for number, reference in enumerate(trial.targets, start=1):
        axis.add_patch(Circle(reference, plotter.TARGET_TOLERANCE, facecolor="white",
                              edgecolor="#555555", linewidth=.75, zorder=5))
        axis.plot(*reference, marker=".", markersize=2, color="#444444", zorder=6)
        offset = (-12, 5) if number == 1 else (5, 5)
        axis.annotate(str(number), reference, xytext=np.asarray(offset) * FONT_SCALE,
                      textcoords="offset points", fontsize=9 * FONT_SCALE, color="#333333", zorder=7)
    axis.scatter(*trial.points[0, 0], s=19, color="#555555", zorder=6)
    axis.set(xlim=(-.23, .23), ylim=(-.23, .23), aspect="equal",
             xticks=[-.2, 0, .2], yticks=[-.2, 0, .2])
    axis.spines[["top", "right"]].set_visible(False)
    return {"pose_indices": indices, "pose_times_s": trial.time[indices].tolist(),
            "reference_positions_m": trial.targets.tolist(),
            "reference_completion_times_s": trial.time[trial.completion_steps].tolist(),
            "strict_violation_count": int(trial.result["violations_strict"]),
            "path": "Complete recorded fingertip trajectory; numbered logical references"}


def draw_pusht(axis, trial, plotter, replay_folder=None):
    outlines = plotter.world_outlines(trial)
    lower, upper = plotter.pair_bounds({METHOD: trial}, {METHOD: outlines})
    poses, arrows, progress = plotter.draw_trial(axis, trial, outlines, lower, upper)
    recorded_centers = trial.centers
    centers = recorded_centers
    path_metadata = {"sampling_hz": 10, "smoothing": False, "interpolation": False}
    if replay_folder is not None:
        report = read_json(replay_folder / "manifest.json")
        archive = pin(replay_folder / "trajectory.npz")
        if digest(archive) != report["output_sha256"][archive.name]:
            raise ValueError("Physics replay archive differs from its manifest")
        if (report["seed"] != trial.seed or not report["control_boundary_states_bitwise_equal"]
                or report["smoothing"] or report["interpolation"]):
            raise ValueError("Physics path is not a verified replay of this trial")
        for path, expected in report["input_sha256"].items():
            if digest(pin(replay_input_path(path, plotter.RECORD_ROOT / METHOD / f"seed_{trial.seed}"))) != expected:
                raise ValueError(f"Physics replay input changed: {path}")
        with np.load(archive, allow_pickle=False) as saved:
            states, times = saved["states"], saved["time"]
        ratio = report["physics_hz"] // report["control_hz"]
        np.testing.assert_array_equal(states[::ratio], trial.states)
        np.testing.assert_array_equal(times[::ratio], trial.time)
        center = np.asarray(trial.geometry["block_center_of_gravity"])
        centers = np.asarray([plotter.transform(center, state[2:]) for state in states])
        if np.any(centers < lower) or np.any(centers > upper):
            raise ValueError("The verified physics path would be clipped by the plot bounds")
        path_metadata.update(sampling_hz=report["physics_hz"], samples=len(centers),
                             recorded_samples=report["recorded_samples"],
                             replay_manifest=str((replay_folder / "manifest.json").resolve()),
                             recorded_boundary_state_error=0.,
                             replayed_physics_steps=report["replayed_physics_steps"],
                             controller_calls=report["controller_calls"])
        # Replace only the center path and its arrows; retain all recorded T poses.
        for annotation in list(axis.texts):
            if not annotation.get_text() and annotation.arrow_patch is not None:
                annotation.remove()
        arrows = plotter.draw_direction_arrows(axis, centers, COLOR)
    path_line = next(line for line in axis.lines
                     if np.array_equal(line.get_xdata(), recorded_centers[:, 0])
                     and np.array_equal(line.get_ydata(), recorded_centers[:, 1]))
    path_line.set_data(centers[:, 0], centers[:, 1])
    path_line.set_solid_joinstyle("round")
    path_line.set_solid_capstyle("round")
    path_line.set_antialiased(True)
    path_line.get_path().should_simplify = False
    return {"pose_indices": [int(row[2]) for row in poses],
            "pose_times_s": [float(row[3]) for row in poses],
            "direction_arrow_indices": arrows, "final_coverage": float(trial.coverage[-1]),
            "path": "Actual block center-of-mass path; no smoothing or pusher path",
            "path_metadata": path_metadata}


def compact_figure():
    """Use square panels with explicit gaps sized for the doubled text."""
    panel_size = 3.8
    left_margin, bottom_margin = 1., .75
    horizontal_gap, vertical_gap = .95, 1.15
    width = left_margin + 2 * panel_size + horizontal_gap + .2
    height = bottom_margin + 2 * panel_size + vertical_gap + 1.05
    figure = plt.figure(figsize=(width, height))
    axes = []
    for row in range(2):
        for column in range(2):
            left = left_margin + column * (panel_size + horizontal_gap)
            bottom = bottom_margin + (1 - row) * (panel_size + vertical_gap)
            axes.append(figure.add_axes([left / width, bottom / height,
                                        panel_size / width, panel_size / height]))
    return figure, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pusht-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pusht-physics-path", type=Path,
                        help="Folder containing a verified 100 Hz saved-action replay.")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    if arguments.pusht_physics_path is None and root == PAPER_RESULTS:
        arguments.pusht_physics_path = REPOSITORY / "data/pusht/physics_replay"
    pusht_root = (arguments.pusht_root or root).resolve()
    output = (arguments.output or root / "figures/four_tasks_bd_chi2").resolve()
    output.mkdir(parents=True, exist_ok=True)
    modules = {task: load_plotter(f"success_figure_{task}", Path(__file__).parent / f"{task}.py")
               for task in TASKS[:3]}
    modules["pusht"] = load_plotter("success_figure_pusht", Path(__file__).with_name("pusht.py"))
    modules["pusht"].RECORD_ROOT = pusht_root / "pusht"
    modules["pusht"].configure_fonts()
    plt.rcParams.update({"font.size": 10 * FONT_SCALE, "axes.labelsize": 11 * FONT_SCALE,
                         "axes.titlesize": 12 * FONT_SCALE})
    trials, panels = {}, []
    for task in TASKS:
        if task == "pusht":
            excluded = sorted(modules[task].LEFT_SEEDS + modules[task].RIGHT_SEEDS)
            directory = pusht_root / "pusht" / METHOD
        else:
            excluded = [{"cartpole": 10006, "rocket": 3005, "reacher": 4000}[task]]
            directory = root / task / METHOD
        seed = select_seed(directory, excluded, require_clean=task == "reacher")
        if task == "pusht":
            protocol = modules[task].load_protocol(pusht_root)
            trial = modules[task].load_trial(directory / f"seed_{seed}", METHOD, seed, protocol)
        else:
            trial = modules[task].load_trial(directory / f"seed_{seed}", METHOD)
        trials[task] = trial
        panels.append({"task": task, "seed": seed, "excluded_comparison_seeds": excluded,
                       "success": bool(trial.result["success"]), "duration_s": float(trial.time[-1]),
                       "record_directory": str(directory / f"seed_{seed}")})
    figure, axes = compact_figure()
    drawers = (draw_cartpole, draw_rocket, draw_reacher, draw_pusht)
    for index, (axis, task, title, draw, panel) in enumerate(zip(axes, TASKS, TITLES, drawers, panels)):
        if task == "pusht":
            panel.update(draw(axis, trials[task], modules[task], arguments.pusht_physics_path))
        else:
            panel.update(draw(axis, trials[task], modules[task]))
        unit = "pixel" if task == "pusht" else "m"
        axis.set_xlabel(f"x ({unit})", fontsize=11 * FONT_SCALE)
        axis.set_ylabel(f"{'Height' if task in ('cartpole', 'rocket') else 'y'} ({unit})",
                        fontsize=11 * FONT_SCALE)
        axis.set_title(title, fontsize=12 * FONT_SCALE, pad=10)
        axis.tick_params(axis="both", labelsize=10 * FONT_SCALE)
        if task != "reacher":
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        panel.update(row=index // 2 + 1, column=index % 2 + 1,
                     xlim=axis.get_xlim(), ylim=axis.get_ylim(), aspect="equal")
    figure.legend(handles=[Line2D([], [], color=COLOR, linewidth=2.6, label="BD-χ²")],
                  loc="upper center", bbox_to_anchor=(.5, .995), frameon=False, fontsize=11 * FONT_SCALE)
    figure.canvas.draw()
    figure_size = figure.get_size_inches().tolist()
    for extension in ("pdf", "svg", "png"):
        figure.savefig(output / f"trajectory_comparison.{extension}", dpi=300, facecolor="white")
    plt.close(figure)
    for module in modules.values():
        INPUT_HASHES.update(module.INPUT_HASHES)
    for path, expected in INPUT_HASHES.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Input changed while generating the figure: {path}")
    rows = "\n".join(f"| {title} | {panel['seed']} | {panel['duration_s']:.2f} |"
                     for title, panel in zip(TITLES, panels))
    physics_note = ""
    replay_argument = ""
    replay_instructions = ""
    if arguments.pusht_physics_path is not None:
        path_metadata = panels[3]["path_metadata"]
        physics_note = ("Push-T uses 100 Hz physics samples obtained by replaying the saved actions. "
                        "The path shows physical direction changes without smoothing or interpolation.\n")
        replay_argument = f" --pusht-physics-path {arguments.pusht_physics_path.resolve()}"
        replay_instructions = (
            "\nRebuild the cached physics path, only if needed:\n\n```bash\n"
            "SDL_VIDEODRIVER=dummy .venv-pusht/bin/python "
            "scripts/plots/replay_pusht_path.py "
            f"--record {panels[3]['record_directory']} "
            f"--output {arguments.pusht_physics_path.resolve()}\n```\n")
    notes = f"""# Successful BD-χ² trajectories on four tasks

[PDF](trajectory_comparison.pdf) · [PNG](trajectory_comparison.png) · [SVG](trajectory_comparison.svg)

Layout: Cartpole / Rocket landing above Reacher / Push-T. Font sizes are
20 pt for ticks, 22 pt for axis labels/legend, and 24 pt for task titles.
Every panel uses equal physical scaling on both axes.
The canvas is {figure_size[0]:g}×{figure_size[1]:g} inches.

| Task | Seed | Duration (s) |
| --- | ---: | ---: |
{rows}

Selection is the lowest successful seed absent from that task's existing
BD-χ²/Select-DPC trajectory comparisons. Reacher also requires zero strict
constraint violations; this trial reaches all five logical references.
These are selected successful examples, not a summary of typical performance.

Dashed paths follow the full recorded pole tip, rocket origin, and fingertip.
Fainter poses are earlier; each task includes its initial and final poses.
Reacher's numbered circles are the five references, and its gray region marks
the fingertip height constraint. Push-T shows a solid block-center path with
direction arrows, gray dashed initial T, gray reference T, intermediate outlines at accumulated-motion thirds, and the true final pose.
The Push-T pusher path is omitted. Push-T ends at {panels[3]['final_coverage']:.2%} coverage.
{physics_note}

Regenerate from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-pusht/bin/python \\
  scripts/plots/bd_chi2_successes.py{replay_argument}
```
{replay_instructions}

Figures use Times New Roman when available, otherwise DejaVu Serif.
Set `PAPER_FONT` to choose another installed font.
"""
    (output / "README.md").write_text(notes)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "script_path": str(Path(__file__).resolve()), "script_sha256": digest(Path(__file__)),
                "method": METHOD, "panels": panels, "input_sha256": INPUT_HASHES,
                "font_family": FONT_FAMILY, "font_scale": FONT_SCALE,
                "font_sizes_pt": {"ticks": 20, "axis_labels": 22, "legend": 22, "titles": 24, "reference_numbers": 18},
                "size_inches": figure_size, "new_simulator_steps": 0, "controller_calls": 0,
                "software": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
                "output_sha256": {path.name: digest(path) for path in sorted(output.iterdir())
                                  if path.name in ("trajectory_comparison.pdf", "trajectory_comparison.png",
                                                   "trajectory_comparison.svg", "README.md")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "selected_seeds": {task: trial.seed for task, trial in trials.items()},
                      "new_simulator_steps": 0}, indent=2))


if __name__ == "__main__":
    main()
