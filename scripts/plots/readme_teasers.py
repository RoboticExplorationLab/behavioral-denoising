"""Render synchronized README teasers using validated, saved paper records.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-pusht/bin/python \
        scripts/plots/readme_teasers.py

Outputs a GIF, MP4, final-frame PNG, and provenance JSON for each composite.
Requires matplotlib, numpy, shapely, Pillow, and ffmpeg, but no simulator.
At 1x playback, every frame uses the latest recorded state at the shared clock;
there is no interpolation, smoothing, controller call, or new physics step.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "diffusion-filtering-matplotlib"))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Polygon, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.plots import cartpole, rocket, reacher, pusht
from scripts.plots.common import PAPER_RESULTS, REPOSITORY, paper_font, replay_input_path

PLOTTERS = {"cartpole": cartpole, "rocket": rocket, "reacher": reacher, "pusht": pusht}
COLORS = {"bd_chi2": "#2ca02c", "select_dpc": "#ff7f0e"}
LABELS = {"bd_chi2": "BD-χ²", "select_dpc": "Select-DPC"}
TITLES = {"cartpole": "Cartpole", "rocket": "Rocket Landing", "reacher": "Reacher", "pusht": "Push-T"}
SELECTIONS = {
    "success": [("cartpole", "bd_chi2", 10003), ("rocket", "bd_chi2", 3000),
                ("reacher", "bd_chi2", 3001), ("pusht", "bd_chi2", 300009)],
    "comparison": [(task, method, seed) for method in COLORS
                   for task, seed in (("cartpole", 10006), ("rocket", 3005), ("reacher", 4000))],
}
STEMS = {"success": "bd_chi2_four_tasks", "comparison": "select_dpc_comparison"}
PATHS = {"cartpole": "pole tip", "rocket": "rocket center", "reacher": "fingertip",
         "pusht": "T-block center of mass"}
INPUT_HASHES: dict[str, str] = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    value = digest(path)
    if INPUT_HASHES.setdefault(str(path), value) != value:
        raise ValueError(f"Input changed during rendering: {path}")
    return path


def square_bounds(lower, upper):
    middle = (np.asarray(lower) + np.asarray(upper)) / 2
    radius = float(np.max(np.asarray(upper) - np.asarray(lower))) / 2
    return middle - radius, middle + radius


def sample_index(times, clock):
    # Tiny tolerance handles representational error at an exact sample boundary.
    return int(np.clip(np.searchsorted(times, clock + 1e-10, side="right") - 1, 0, len(times) - 1))


@dataclass
class Episode:
    task: str
    method: str
    trial: object
    directory: Path
    time: np.ndarray
    states: np.ndarray | None
    path: np.ndarray
    geometry: object = None

    @property
    def endstate(self):
        if self.trial.result["success"]:
            return "success"
        if self.task == "reacher" or self.trial.result.get("termination_reason") == "timeout":
            return "timeout"
        return "terminated"


def cached_pusht_samples(trial, directory, cache):
    """Read the existing paper cache, verifying provenance and every saved boundary."""
    report = json.loads(pin(cache / "manifest.json").read_text())
    archive = pin(cache / "trajectory.npz")
    if (digest(archive) != report["output_sha256"][archive.name]
            or report["seed"] != trial.seed
            or not report["control_boundary_states_bitwise_equal"]
            or not report["velocities_contacts_and_termination_equal"]
            or report["smoothing"] or report["interpolation"] or report["controller_calls"] != 0):
        raise ValueError("The cached Push-T path is not a validated replay of this record")
    for original, expected in report["input_sha256"].items():
        if digest(pin(replay_input_path(original, directory))) != expected:
            raise ValueError(f"Cached replay input changed: {original}")
    with np.load(archive, allow_pickle=False) as saved:
        states, times = saved["states"].copy(), saved["time"].copy()
    ratio = report["physics_hz"] // report["control_hz"]
    np.testing.assert_array_equal(states[::ratio], trial.states)
    np.testing.assert_array_equal(times[::ratio], trial.time)
    if states[::ratio].tobytes() != trial.states.tobytes():
        raise ValueError("Cached control-boundary state bytes differ")
    if (not np.isfinite(states).all() or not np.isfinite(times).all()
            or states.shape != (len(times), 5) or np.any(np.diff(times) <= 0)):
        raise ValueError("Malformed Push-T cache")
    return times, states


def load_episodes(selection, root, cache):
    episodes = []
    for task, method, seed in SELECTIONS[selection]:
        module = PLOTTERS[task]
        directory = root / task / method / f"seed_{seed}"
        trial = (module.load_trial(directory, method, seed, module.load_protocol(root))
                 if task == "pusht" else module.load_trial(directory, method))
        times, states, geometry = trial.time, getattr(trial, "states", None), None
        if task == "cartpole":
            geometry = module.load_pole_geometry(trial)
            path = np.asarray([np.asarray(module.pose_coordinates(state, geometry))[:, -1] for state in states])
        elif task == "rocket":
            path = states[:, :2]
        elif task == "reacher":
            path = trial.points[:, -1]
            if method == "bd_chi2" and (not trial.result["clean_success_strict"] or len(trial.completion_steps) != 5):
                raise ValueError("Selected BD-χ² Reacher example must be a strict-clean five-target success")
        else:
            if cache is not None:
                times, states = cached_pusht_samples(trial, directory, cache)
            geometry = np.asarray(trial.geometry["block_center_of_gravity"])
            path = np.asarray([pusht.transform(geometry, state[2:]) for state in states])
        if selection == "success" and not trial.result["success"]:
            raise ValueError("A success teaser record is not successful")
        episodes.append(Episode(task, method, trial, directory, times, states, path, geometry))
    if selection == "comparison":
        for first, second in zip(episodes[:3], episodes[3:]):
            a = first.trial.points if first.task == "reacher" else first.states
            b = second.trial.points if first.task == "reacher" else second.states
            np.testing.assert_array_equal(a[0], b[0])
            if first.task == "reacher":
                np.testing.assert_array_equal(first.trial.observations[0], second.trial.observations[0])
                np.testing.assert_array_equal(first.trial.targets, second.trial.targets)
    return episodes


def bounds_for(episodes):
    """Equal world scaling, shared bounds within each comparison column."""
    result = {}
    for task in {episode.task for episode in episodes}:
        group = [episode for episode in episodes if episode.task == task]
        if task == "cartpole":
            points = np.vstack([np.asarray(cartpole.pose_coordinates(state, episode.geometry)).T
                                for episode in group for state in episode.states])
            carts = np.concatenate([episode.states[:, 0] for episode in group])
            result[task] = square_bounds([min(points[:, 0].min(), carts.min() - .12) - .12, -.75],
                                         [max(points[:, 0].max(), carts.max() + .12) + .12, .75])
        elif task == "rocket":
            result[task] = square_bounds(*rocket.scene_bounds({episode.method: episode.trial for episode in group}))
        elif task == "reacher":
            result[task] = (np.array([-.23, -.23]), np.array([.23, .23]))
        else:
            episode = group[0]
            outlines = np.asarray([pusht.transform(episode.trial.outline, state[2:]) for state in episode.states])
            radius = episode.trial.geometry["agent_radius"]
            points = np.vstack([outlines.reshape(-1, 2), pusht.transform(episode.trial.outline, episode.trial.goal_pose),
                                episode.states[:, :2] - radius, episode.states[:, :2] + radius])
            result[task] = square_bounds(points.min(0) - 12, points.max(0) + 12)
    return result


class Panel:
    def __init__(self, axis, episode, bounds):
        self.axis, self.episode = axis, episode
        self.color = COLORS[episode.method]
        low, high = bounds
        axis.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal")
        if episode.task == "pusht":
            axis.invert_yaxis()
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#e0e0e0")
            spine.set_linewidth(.6)
        self.trail, = axis.plot([], [], color=self.color, alpha=.65, lw=1.15,
                               linestyle="-" if episode.task == "pusht" else "--", zorder=2)
        self.trail.get_path().should_simplify = False
        self.status = axis.text(.97, .97, "", transform=axis.transAxes, ha="right", va="top",
                                fontsize=11, color="#555555", zorder=20,
                                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92, "pad": 2})
        getattr(self, f"init_{episode.task}")()

    def init_cartpole(self):
        self.axis.axhline(0, color="#b9b9b9", lw=.8, zorder=0)
        self.axis.axvline(0, color="#dedede", lw=.6, linestyle=":", zorder=0)
        self.cart = Rectangle((0, -.06), .24, .12, facecolor=to_rgba(self.color, .12),
                              edgecolor=self.color, lw=1.5, zorder=4)
        self.axis.add_patch(self.cart)
        self.pole, = self.axis.plot([], [], color=self.color, lw=4, solid_capstyle="round", zorder=5)
        self.tip, = self.axis.plot([], [], "o", color=self.color, ms=5, zorder=6)

    def init_rocket(self):
        self.bodies = []
        for body in self.episode.trial.rendering["rocket_bodies"]:
            for fixture, vertices in enumerate(rocket.world_vertices(self.episode.trial, body, 0)):
                platform = body["name"] == "barge"
                patch = Polygon(vertices, facecolor="#ececec" if platform else to_rgba(self.color, .13),
                                edgecolor="#888888" if platform else self.color,
                                lw=.9 if platform else 1.3, zorder=1 if platform else 5)
                self.axis.add_patch(patch)
                self.bodies.append((body, fixture, patch))

    def init_reacher(self):
        self.axis.axhspan(reacher.HEIGHT_LIMIT, .23, color="#f1f1f1", zorder=0)
        self.axis.axhline(reacher.HEIGHT_LIMIT, color="#b5b5b5", lw=.7, zorder=1)
        self.targets = []
        for number, point in enumerate(self.episode.trial.targets, 1):
            target = Circle(point, reacher.TARGET_TOLERANCE, facecolor="white", edgecolor="#888888", lw=1, zorder=7)
            self.axis.add_patch(target)
            self.targets.append(target)
            offset = (-9, 5) if number == 1 else (5, 5)
            self.axis.annotate(str(number), point, xytext=offset, textcoords="offset points",
                               fontsize=10, color="#777777", zorder=8)
        self.arm, = self.axis.plot([], [], color=self.color, lw=5, solid_capstyle="round", zorder=5)
        self.joints, = self.axis.plot([], [], "o", color=self.color, ms=5, zorder=6)
        self.axis.plot(*self.episode.trial.points[0, 0], "o", ms=5, color="#555555", zorder=9)

    def init_pusht(self):
        trial = self.episode.trial
        goal = Polygon(pusht.transform(trial.outline, trial.goal_pose), facecolor="#e8e8e8",
                       edgecolor="#a2a2a2", lw=1, linestyle=":", zorder=1)
        self.axis.add_patch(goal)
        self.block = Polygon(pusht.transform(trial.outline, self.episode.states[0, 2:]),
                             facecolor=to_rgba(self.color, .10), edgecolor=self.color, lw=1.8, zorder=5)
        self.axis.add_patch(self.block)
        self.pusher = Circle(self.episode.states[0, :2], trial.geometry["agent_radius"],
                             facecolor="#d4d4d4", edgecolor="#a8a8a8", lw=.8, zorder=4)
        self.axis.add_patch(self.pusher)

    def update(self, clock):
        episode = self.episode
        index = sample_index(episode.time, clock)
        self.trail.set_data(episode.path[:index + 1, 0], episode.path[:index + 1, 1])
        self.trail.get_path().should_simplify = False
        end = float(episode.time[-1])
        self.status.set_text(f"{episode.endstate} · {end:.2f} s" if clock + 1e-10 >= end else "")
        if episode.task == "cartpole":
            state = episode.states[index]
            x, y = cartpole.pose_coordinates(state, episode.geometry)
            self.cart.set_xy((state[0] - .12, -.06))
            self.pole.set_data(x, y)
            self.tip.set_data([x[-1]], [y[-1]])
        elif episode.task == "rocket":
            for body, fixture, patch in self.bodies:
                patch.set_xy(list(rocket.world_vertices(episode.trial, body, index))[fixture])
        elif episode.task == "reacher":
            points = episode.trial.points[index]
            self.arm.set_data(points[:, 0], points[:, 1])
            self.joints.set_data(points[:, 0], points[:, 1])
            completed = int(np.count_nonzero(episode.trial.completion_steps <= index))
            for number, target in enumerate(self.targets):
                target.set_facecolor(to_rgba(self.color, .2) if number < completed else "white")
                target.set_edgecolor(self.color if number <= completed else "#888888")
                target.set_linewidth(1.4 if number == completed else .9)
        else:
            state = episode.states[index]
            self.block.set_xy(pusht.transform(episode.trial.outline, state[2:]))
            self.pusher.center = state[:2]


def make_figure(selection, episodes, width):
    columns = 2 if selection == "success" else 3
    # All task scenes occupy square panels; headers sit in narrow reserved strips.
    outer, gap = 20, 20
    panel = (width - 2 * outer - (columns - 1) * gap) / columns
    header, between, bottom = 75, 50 if selection == "success" else 70, 16
    height = int(np.ceil((2 * panel + header + between + bottom) / 2) * 2)
    # Agg truncates floating canvas dimensions; the subpixel epsilon prevents
    # e.g. 8.28 inches becoming 827 pixels and invalidating H.264's even size.
    figure = plt.figure(figsize=((width + .001) / 100, (height + .001) / 100), dpi=100, facecolor="white")
    if figure.canvas.get_width_height() != (width, height):
        raise ValueError("Unexpected raster dimensions")
    clock = figure.text((width - outer) / width, 1 - 23 / height, "", ha="right", va="center", fontsize=15, color="#444444")
    figure.text(outer / width, 1 - 23 / height, LABELS["bd_chi2"], color=COLORS["bd_chi2"], fontsize=17, va="center")
    if selection == "comparison":
        figure.text(outer / width, 1 - (header + panel + 27) / height, LABELS["select_dpc"],
                    color=COLORS["select_dpc"], fontsize=17, va="center")
    bounds = bounds_for(episodes)
    panels = []
    for index, episode in enumerate(episodes):
        row, column = divmod(index, columns)
        left = outer + column * (panel + gap)
        top = header + row * (panel + between)
        axis = figure.add_axes([left / width, 1 - (top + panel) / height, panel / width, panel / height])
        figure.text((left + panel / 2) / width, 1 - (top - 16) / height,
                    TITLES[episode.task], ha="center", va="center", fontsize=18)
        panels.append(Panel(axis, episode, bounds[episode.task]))
    return figure, clock, panels, (width, height), bounds


def ffmpeg(*arguments):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1",
                    "-filter_threads", "1", "-filter_complex_threads", "1", *map(str, arguments)], check=True)


def encode_frames(directory, output, fps, colors):
    source = directory / "%05d.png"
    palette = directory / "palette.png"
    ffmpeg("-framerate", fps, "-i", source, "-vf", f"palettegen=max_colors={colors}:stats_mode=diff",
           "-frames:v", "1", "-threads", "1", palette)
    ffmpeg("-framerate", fps, "-i", source, "-i", palette, "-lavfi",
           "paletteuse=dither=none:diff_mode=rectangle", "-loop", "0", "-threads", "1", output.with_suffix(".gif"))
    ffmpeg("-framerate", fps, "-i", source, "-c:v", "libx264", "-crf", "18", "-preset", "slow",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "1", output.with_suffix(".mp4"))


def relative(path):
    path = Path(path)
    return str(path.relative_to(REPOSITORY)) if path.is_relative_to(REPOSITORY) else str(path)


def render(selection, args):
    INPUT_HASHES.clear()
    for module in PLOTTERS.values():
        module.INPUT_HASHES.clear()
        pin(module.__file__)
    pin(Path(__file__))
    pin(Path(__file__).with_name("common.py"))
    episodes = load_episodes(selection, args.root.resolve(), None if args.recorded_pusht_only else args.pusht_cache.resolve())
    font = paper_font()
    plt.rcParams.update({"font.family": font, "font.size": 12, "path.simplify": False})
    width = args.width or (960 if selection == "success" else 1080)
    figure, clock, panels, dimensions, bounds = make_figure(selection, episodes, width)
    duration = max(float(episode.time[-1]) for episode in episodes)
    frame_count = int(np.ceil((duration + args.final_hold) * args.fps))
    frame_times = np.minimum(np.arange(frame_count) / args.fps, duration)
    if frame_times[-1] < duration:
        raise ValueError("Final hold must allow the final recorded frame to appear")
    output = args.output.resolve() / STEMS[selection]
    output.parent.mkdir(parents=True, exist_ok=True)
    preview_indices = {sample_index(frame_times, min(time, duration)) for time in (0., 1.25, 3.5, 7.5, duration)}
    with tempfile.TemporaryDirectory(prefix=f"{output.stem}-") as temporary:
        directory = Path(temporary)
        for index, time in enumerate(frame_times):
            clock.set_text(f"t = {time:.2f} s")
            for panel in panels:
                panel.update(float(time))
            figure.canvas.draw()
            frame = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
            frame.save(directory / f"{index:05d}.png", compress_level=1)
            if args.preview_dir is not None and index in preview_indices:
                args.preview_dir.mkdir(parents=True, exist_ok=True)
                frame.save(args.preview_dir / f"{output.stem}_{time:05.2f}s.png")
            if index == frame_count - 1:
                frame.save(output.with_suffix(".png"), optimize=True)
            if index % (args.fps * 5) == 0:
                print(f"{selection}: {time:.2f}/{duration:.2f} s", flush=True)
        plt.close(figure)
        encode_frames(directory, output, args.fps, args.colors)
    for module in PLOTTERS.values():
        INPUT_HASHES.update(module.INPUT_HASHES)
    for path, expected in INPUT_HASHES.items():
        if digest(path) != expected:
            raise ValueError(f"Input changed during rendering: {path}")
    with Image.open(output.with_suffix(".gif")) as gif:
        gif_duration_ms = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            gif_duration_ms += gif.info["duration"]
        if abs(gif_duration_ms / 1000 - frame_count / args.fps) > .011:
            raise ValueError("GIF duration differs from the common physical clock")
        gif_frames = gif.n_frames
    manifest = {
        "selection": selection, "font": font, "dimensions_px": dimensions,
        "fps": args.fps, "speed": 1, "physical_duration_s": duration,
        "final_hold_s": frame_count / args.fps - duration,
        "encoded_duration_s": gif_duration_ms / 1000, "rendered_frames": frame_count,
        "gif_frames": gif_frames, "gif_palette_colors": args.colors,
        "sampling": "Latest recorded state at or before shared physical clock; hold endpoints with outcome label",
        "smoothing": False, "interpolation": False, "new_simulator_steps": 0, "controller_calls": 0,
        "cached_pusht_physics_samples": selection == "success" and not args.recorded_pusht_only,
        "panels": [{"row": index // (2 if selection == "success" else 3) + 1,
                    "column": index % (2 if selection == "success" else 3) + 1,
                    "task": episode.task, "method": episode.method, "color": COLORS[episode.method],
                    "seed": episode.trial.seed, "endstate": episode.endstate,
                    "duration_s": float(episode.time[-1]), "source": relative(episode.directory),
                    "samples": len(episode.time), "path": PATHS[episode.task],
                    "bounds": [bound.tolist() for bound in bounds[episode.task]]}
                   for index, episode in enumerate(episodes)],
        "input_sha256": {relative(Path(path)): value for path, value in sorted(INPUT_HASHES.items())},
        "outputs": {suffix: {"path": relative(output.with_suffix(suffix)),
                              "bytes": output.with_suffix(suffix).stat().st_size,
                              "sha256": digest(output.with_suffix(suffix))}
                    for suffix in (".gif", ".mp4", ".png")},
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "dimensions": dimensions, "duration_s": gif_duration_ms / 1000,
                      "sizes_bytes": {key: value["bytes"] for key, value in manifest["outputs"].items()}}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("success", "comparison", "all"), default="all")
    parser.add_argument("--root", type=Path, default=PAPER_RESULTS)
    parser.add_argument("--output", type=Path, default=REPOSITORY / "media/teasers")
    parser.add_argument("--fps", type=int, choices=(5, 10, 20, 25, 50), default=20,
                        help="Frames per simulated second; GIF-compatible rates keep exactly 1x time")
    parser.add_argument("--width", type=int, help="Even canvas width; defaults to 960 / 1080 pixels")
    parser.add_argument("--colors", type=int, default=128, help="Global GIF palette size (16–256)")
    parser.add_argument("--final-hold", type=float, default=1., help="Seconds to hold the final composite (default: 1)")
    parser.add_argument("--pusht-cache", type=Path, default=REPOSITORY / "data/pusht/physics_replay")
    parser.add_argument("--recorded-pusht-only", action="store_true", help="Use the original 10 Hz Push-T samples instead of the validated cache")
    parser.add_argument("--preview-dir", type=Path, help="Optionally save representative PNG frames for inspection")
    args = parser.parse_args()
    if args.width is not None and (args.width < 600 or args.width % 2):
        parser.error("--width must be even and at least 600")
    if not 16 <= args.colors <= 256 or args.final_hold < 1 / args.fps:
        parser.error("Use 16–256 colors and at least one frame of final hold")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required for GIF and MP4 encoding")
    for selection in SELECTIONS if args.only == "all" else (args.only,):
        render(selection, args)


if __name__ == "__main__":
    main()
