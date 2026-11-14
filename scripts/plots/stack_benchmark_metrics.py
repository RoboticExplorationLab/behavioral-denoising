"""Stack existing Cartpole, Rocket and Reacher metrics using saved records only.

No controller or simulator is run. Individual task figures remain untouched.
"""
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
from matplotlib import font_manager
import numpy as np


DEFAULT_ROOT = (Path(__file__).resolve().parents[2] / "results/paper")
TASKS = ("cartpole", "rocket", "reacher")
TITLES = ("(a) Cartpole", "(b) Rocket Landing", "(c) Reacher")
FONT_SCALE = 1.5
INPUT_HASHES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    path = path.resolve()
    value = digest(path)
    if INPUT_HASHES.setdefault(str(path), value) != value:
        raise ValueError(f"Input changed during plotting: {path}")
    return path


def load_plotter(task, root):
    path = pin(Path(__file__).parent / f"{task}.py")
    name = f"stacked_metrics_{task}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = (arguments.output or root / "figures/stacked_metrics").resolve()
    output.mkdir(parents=True, exist_ok=True)
    modules = {task: load_plotter(task, root) for task in TASKS}
    font = pin(Path(font_manager.findfont(FONT_FAMILY, fallback_to_default=False)))
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": 10 * FONT_SCALE,
                         "axes.labelsize": 11 * FONT_SCALE, "axes.titlesize": 12 * FONT_SCALE,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

    figure = plt.figure(figsize=(7.2, 10.4))
    groups = figure.add_gridspec(3, 1, left=.18, right=.985, bottom=.065, top=.915, hspace=.38)
    axes = {}
    for row, task in enumerate(TASKS):
        pair = groups[row].subgridspec(2, 1, hspace=.20)
        top = figure.add_subplot(pair[0])
        bottom = figure.add_subplot(pair[1], sharex=top)
        axes[task] = [top, bottom]

    cartpole = modules["cartpole"]
    cartpole_trials = cartpole.load_trials(root)
    cartpole_times, cartpole_summary = cartpole.summarize(cartpole_trials)
    cartpole.draw_tracking_axes(axes["cartpole"], cartpole_trials, cartpole_times, cartpole_summary)

    rocket = modules["rocket"]
    rocket_trials, landing_reference = rocket.load_trials(root)
    rocket_times, rocket_summary = rocket.summarize(rocket_trials, landing_reference)
    rocket.draw_errors(axes["rocket"], rocket_trials, landing_reference, rocket_times, rocket_summary)

    reacher = modules["reacher"]
    reacher_trials = reacher.load_trials(root)
    reacher_times, reached_counts = reacher.progress(reacher_trials)
    reacher.draw_progress(axes["reacher"], reacher_times, reached_counts)

    labels = {"cartpole": ("Pole-Angle\nError (deg)", "Cart-Position\nError (m)"),
              "rocket": ("Tilt Error\n(deg)", "Distance To\nLanding Point\n(m)"),
              "reacher": ("References\nReached", "All References\nReached (%)")}
    # Keep tick locations fixed across font sizes.
    axes["cartpole"][1].set_yticks([0, 1, 2])
    axes["rocket"][0].set_yticks([0, 20, 40])
    axes["rocket"][1].set_yticks([0, 10, 20])
    for task, title in zip(TASKS, TITLES):
        top, bottom = axes[task]
        top.tick_params(labelbottom=False)
        top.set_title(title, loc="left", pad=4)
        for axis, label in zip((top, bottom), labels[task]):
            axis.set_ylabel(label)
            axis.tick_params(axis="both", labelsize=10 * FONT_SCALE)
        bottom.set_xlabel("Time (s)")
    figure.align_ylabels([axis for task in TASKS for axis in axes[task]])
    figure.legend(handles=cartpole.method_handles(), loc="upper center",
                  bbox_to_anchor=(.57, .995), ncol=2, frameon=False, fontsize=11 * FONT_SCALE)
    figure.canvas.draw()
    geometry = {task: [{"xlim": axis.get_xlim(), "ylim": axis.get_ylim(),
                        "position_fraction": list(axis.get_position().bounds)} for axis in axes[task]]
                for task in TASKS}
    for extension in ("pdf", "svg", "png"):
        figure.savefig(output / f"stacked_metrics.{extension}", dpi=300, facecolor="white",
                       bbox_inches="tight", pad_inches=.04)
    plt.close(figure)

    for task in TASKS:
        INPUT_HASHES.update(modules[task].INPUT_HASHES)
    for path, expected in INPUT_HASHES.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Input changed during plotting: {path}")

    (output / "README.md").write_text("""# Stacked benchmark errors and completion

[PDF](stacked_metrics.pdf) · [PNG](stacked_metrics.png) · [SVG](stacked_metrics.svg)

Three vertical groups:
Cartpole angle/position errors; Rocket tilt/landing-distance errors; Reacher
reference count and percentage completing all five. One shared legend uses
green for BD-χ² and orange for Select-DPC. Font sizes are 15 pt for ticks,
16.5 pt for labels/legend, and 18 pt for task headings.
Headings are (a) Cartpole, (b) Rocket Landing, and (c) Reacher. Displayed labels
use title case, retaining lowercase units. Outer whitespace is tightly cropped.

All ten paired starts per task are included. Cartpole and Rocket show individual
trials, pointwise medians
and interquartile ranges; dashed medians use only the remaining recorded trials.
Physical trajectories end at their actual endpoints, without padding. Rocket's
completed successes and failures both leave those summaries. Cartpole crosses
mark early endpoints; Rocket circles/crosses mark successful/failed endpoints.

Reacher shows the mean number of completed references and the percentage of all
ten trials completing five. Recorded completion events persist after stopping;
this does not extend physical trajectories. Completion does not additionally
require constraint-clean motion. Each task retains its own time range.

Regenerate from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \\
  scripts/plots/stack_benchmark_metrics.py
```

Figures use Times New Roman when available, otherwise DejaVu Serif.
Set `PAPER_FONT` to choose another installed font.
""")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "script_path": str(Path(__file__).resolve()), "script_sha256": digest(Path(__file__)),
                "input_sha256": INPUT_HASHES, "order": list(TASKS),
                "methods": list(cartpole.METHODS), "trials_per_method_per_task": 10,
                "axes": geometry, "legend_count": 1,
                "font_family": FONT_FAMILY, "font_scale": FONT_SCALE,
                "task_titles": list(TITLES),
                "font_sizes_pt": {"ticks": 10 * FONT_SCALE, "labels": 11 * FONT_SCALE,
                                  "legend": 11 * FONT_SCALE, "titles": 12 * FONT_SCALE},
                "source_canvas_inches": [7.2, 10.4], "tight_crop_padding_inches": .04,
                "reacher_final_completion_percent": {
                    method: float(100 * np.mean(counts[:, -1] == reacher.TARGET_COUNT))
                    for method, counts in reached_counts.items()},
                "controller_calls": 0, "simulator_steps": 0,
                "software": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
                "output_sha256": {path.name: digest(path) for path in sorted(output.iterdir())
                                  if path.name in ("stacked_metrics.pdf", "stacked_metrics.png", "stacked_metrics.svg", "README.md")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "panels": 6, "legend_count": 1, "simulator_steps": 0}, indent=2))


if __name__ == "__main__":
    main()
