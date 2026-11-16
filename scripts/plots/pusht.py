"""Plot coverage from saved Push-T runs without running a controller or simulator.

    python scripts/plots/pusht.py --root runs/benchmark

The median and interquartile range use the trials still recorded at each time;
no coverage values are invented after termination.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from scripts.plots.common import DEFAULT_RUNS, plot_font

LABELS = {"bd_chi2": "BD-χ²", "select_dpc": "Select-DPC"}
COLORS = {"bd_chi2": "#2ca02c", "select_dpc": "#ff7f0e"}
FONT_FAMILY = plot_font()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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

    @property
    def centers(self):
        center = np.asarray(self.geometry["block_center_of_gravity"])
        return np.array([transform(center, pose) for pose in self.states[:, 2:]])


def load_protocol(root):
    path = Path(root) / "protocols/pusht.json"
    return json.loads(path.read_text())


def load_trial(directory, method, seed, protocol):
    """Validate the recorded payload, physical geometry and coverage outcomes."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    hashes = metadata["payload_sha256"]
    for filename in ("trajectory.npz", "result.json"):
        if digest(directory / filename) != hashes[filename]:
            raise ValueError(f"Record hash mismatch: {directory / filename}")
    result = json.loads((directory / "result.json").read_text())
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
    expected_methods = {entry["id"]: entry for entry in protocol["methods"]}
    if (metadata["protocol_sha256"] != protocol["sha256"]
            or metadata["method"] != expected_methods[method]
            or result["method"] != method or result["seed"] != seed
            or geometry["control_hz"] != 10
            or not np.array_equal(arrays["goal_pose"], geometry["goal_pose"])):
        raise ValueError(f"Unexpected method or task protocol: {directory}")
    np.testing.assert_allclose(arrays["time"], np.arange(count + 1) / geometry["control_hz"], atol=1e-12, rtol=0)
    coverage = arrays["coverage"]
    threshold = geometry["success_threshold"]
    if np.any(coverage < 0) or np.any(coverage > 1 + 1e-12):
        raise ValueError(f"Coverage outside [0, 1]: {directory}")
    success = bool(np.any(coverage[1:] > threshold))
    maximum = float(coverage[1:].max()) if count else 0.0
    if (success != result["success"] or abs(result["max_coverage"] - maximum) > 1e-12
            or abs(result["final_coverage"] - coverage[-1]) > 1e-12
            or (success and (coverage[-1] <= threshold or np.any(coverage[1:-1] > threshold)))):
        raise ValueError(f"Coverage/outcome mismatch: {directory}")
    local_shape = unary_union([Polygon(vertices) for vertices in geometry["block_polygons"]])
    if local_shape.geom_type != "Polygon" or local_shape.interiors:
        raise ValueError("Expected a single T-shaped outline without holes")
    outline = np.asarray(local_shape.exterior.coords)
    goal = Polygon(transform(outline, arrays["goal_pose"]))
    for index in sorted({0, count // 2, count}):
        block = Polygon(transform(outline, arrays["states"][index, 2:]))
        if abs(goal.intersection(block).area / goal.area - coverage[index]) > 1e-12:
            raise ValueError(f"Saved block geometry disagrees with coverage: {directory}")
    return Trial(method, seed, arrays["time"], arrays["states"], coverage,
                 arrays["goal_pose"], geometry, outline, result)


def load_trials(root, methods=None):
    """Load available completed runs, including partial cohorts and single methods."""
    root = Path(root)
    protocol = load_protocol(root)
    methods = methods or [entry["id"] for entry in protocol["methods"]]
    trials = {}
    for method in methods:
        directories = sorted((root / "pusht" / method).glob("seed_*"))
        group = [load_trial(directory, method, int(directory.name.removeprefix("seed_")), protocol)
                 for directory in directories if (directory / "result.json").is_file()]
        if group:
            trials[method] = group
    if not trials:
        raise ValueError(f"No completed Push-T trials found under {root}")
    return trials


def plot_coverage(trials, output):
    """Draw per-run traces and the median/IQR of available observations."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": 11})
    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=.2)
    horizon = max(trial.time[-1] for group in trials.values() for trial in group)
    times = np.arange(round(horizon * 10) + 1) / 10
    columns = {"time_seconds": times}
    for method, group in trials.items():
        color = COLORS.get(method, "#1f77b4")
        values = np.full((len(group), len(times)), np.nan)
        for row, trial in enumerate(group):
            values[row, :len(trial.time)] = trial.coverage
            axis.plot(trial.time, 100 * trial.coverage, color=color, alpha=.17, linewidth=.7)
        available = np.isfinite(values).sum(axis=0)
        median, lower, upper = np.full(len(times), np.nan), np.full(len(times), np.nan), np.full(len(times), np.nan)
        valid = available > 0
        lower[valid], median[valid], upper[valid] = np.nanquantile(values[:, valid], [.25, .5, .75], axis=0)
        axis.fill_between(times, 100 * lower, 100 * upper, color=color, alpha=.18, linewidth=0)
        complete = available == len(group)
        axis.plot(times, np.where(complete, 100 * median, np.nan), color=color,
                  linewidth=2, label=LABELS.get(method, method))
        partial = valid & ~complete
        if partial.any():
            partial[max(0, np.flatnonzero(partial)[0] - 1)] = True
            axis.plot(times, np.where(partial, 100 * median, np.nan), color=color, linewidth=2, linestyle="--")
        columns.update({f"{method}_recorded_trials": available, f"{method}_median": median,
                        f"{method}_q25": lower, f"{method}_q75": upper})
    axis.set(xlabel="Time (s)", ylabel="Coverage (%)", ylim=(-1, 101), xlim=(0, max(horizon, .1)))
    axis.legend(frameon=False)
    for extension in ("pdf", "png", "svg"):
        figure.savefig(output / f"coverage.{extension}", dpi=200)
    plt.close(figure)
    np.savetxt(output / "coverage.csv", np.column_stack(list(columns.values())), delimiter=",",
               header=",".join(columns), comments="")
    (output / "coverage.txt").write_text(
        "Thin lines: individual trials. Shading: interquartile range. Solid line: median while all "
        "trials have observations. Dashed line: median of the trials still recorded. "
        "No values are extended after termination.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    trials = load_trials(args.root, args.methods)
    plot_coverage(trials, args.output or args.root / "figures/pusht")


if __name__ == "__main__":
    main()
