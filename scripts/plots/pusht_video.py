"""Plot saved Push-T episodes without importing a simulator or rerunning a policy.

    python scripts/plots/pusht_video.py RUN/seed_300005 \
        --output RUN/seed_300005/visualizations --format gif

Pass two episode directories to compare methods from the same start. Animation
uses saved simulation times, freezing a completed episode at its final state.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.patches import Circle, Polygon
import numpy as np


METHODS = {"bd_chi2": ("BD-χ²", "#2ca02c"), "select_dpc": ("Select-DPC", "#ff7f0e")}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
BLOCK_VERTICES = (
    np.array([[-60, 30], [60, 30], [60, 0], [-60, 0]], dtype=float),
    np.array([[-15, 30], [-15, 120], [15, 120], [15, 30]], dtype=float),
)


def load_episode(directory):
    directory = Path(directory)
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    result = json.loads((directory / "result.json").read_text())
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    count = len(arrays["actions"])
    shapes = {"states": (count + 1, 5), "observations": (count + 1, 6),
              "actions": (count, 2), "proposed_actions": (count, 2),
              "coverage": (count + 1,), "rewards": (count,),
              "time": (count + 1,), "goal_pose": (3,)}
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{directory}: {key} must have shape {shape}")
    if not np.isfinite(arrays["time"]).all() or np.any(np.diff(arrays["time"]) <= 0):
        raise ValueError(f"{directory}: simulation times must be finite and strictly increasing")
    method = result.get("method", metadata.get("method", directory.parent.name))
    if isinstance(method, dict):
        method = method.get("id", method.get("name", directory.parent.name))
    label, color = METHODS.get(method, (str(method), "#1f77b4"))
    return {"directory": directory, "arrays": arrays, "result": result,
            "metadata": metadata, "label": label, "color": color}


def transformed_block(pose):
    """Transform local vertices using the saved Pymunk body origin, not its CoM."""
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return [vertices @ rotation.T + pose[:2] for vertices in BLOCK_VERTICES]


def save_figure(figure, directory, name):
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_traces(episodes, output):
    figure, axes = plt.subplots(3, 2, figsize=(9, 8), sharex=True)
    labels = ("Agent x (pixel)", "Agent y (pixel)", "Block x (pixel)",
              "Block y (pixel)", "Block angle (rad)")
    for episode in episodes:
        arrays, color = episode["arrays"], episode["color"]
        for component, axis in enumerate(axes.flat[:5]):
            values = arrays["states"][:, component]
            if component == 4:
                values = np.unwrap(values)
            axis.plot(arrays["time"], values, color=color, label=episode["label"])
        axes.flat[5].plot(arrays["time"], arrays["coverage"], color=color)
    for axis, label in zip(axes.flat, (*labels, "Target coverage")):
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    axes.flat[5].axhline(0.95, color="0.45", ls="--", lw=1)
    axes.flat[5].set_ylim(0, 1.02)
    axes.flat[0].legend(frameon=False)
    save_figure(figure, output, "states")

    figure, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for episode in episodes:
        arrays, color = episode["arrays"], episode["color"]
        if not len(arrays["actions"]):
            continue
        for component, axis in enumerate(axes):
            applied = np.r_[arrays["actions"][:, component], arrays["actions"][-1, component]]
            axis.step(arrays["time"], applied, where="post",
                      color=color, label=episode["label"])
            if not np.array_equal(arrays["actions"], arrays["proposed_actions"]):
                proposed = np.r_[arrays["proposed_actions"][:, component],
                                 arrays["proposed_actions"][-1, component]]
                axis.step(arrays["time"], proposed,
                          where="post", color=color, alpha=0.5, linestyle="--")
    for axis, label in zip(axes, ("Agent target x (pixel)", "Agent target y (pixel)")):
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Time (s)")
    save_figure(figure, output, "controls")

    figure, axis = plt.subplots(figsize=(6, 3.5))
    for episode in episodes:
        arrays = episode["arrays"]
        axis.plot(arrays["time"], arrays["coverage"], color=episode["color"], label=episode["label"])
    axis.axhline(0.95, color="0.45", ls="--", lw=1)
    axis.set(xlabel="Time (s)", ylabel="Target coverage", ylim=(0, 1.02))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    save_figure(figure, output, "coverage")


def animate(episodes, output, movie_format, fps):
    if movie_format == "mp4" and not writers.is_available("ffmpeg"):
        raise RuntimeError("MP4 output requires ffmpeg; use --format gif if it is unavailable")
    figure, axes = plt.subplots(1, len(episodes), figsize=(4 * len(episodes), 4), squeeze=False)
    artists = []
    for axis, episode in zip(axes[0], episodes):
        arrays = episode["arrays"]
        for polygon in transformed_block(arrays["goal_pose"]):
            axis.add_patch(Polygon(polygon, facecolor="#c4e5be", edgecolor="none", zorder=0))
        blocks = [Polygon(polygon, facecolor="0.65", edgecolor="0.3", zorder=2)
                  for polygon in transformed_block(arrays["states"][0, 2:5])]
        for block in blocks:
            axis.add_patch(block)
        agent = Circle(arrays["states"][0, :2], radius=15,
                       facecolor=episode["color"], edgecolor="white", zorder=3)
        axis.add_patch(agent)
        path, = axis.plot([], [], color=episode["color"], lw=0.8, alpha=0.6, zorder=1)
        axis.set(xlim=(0, 512), ylim=(512, 0), xlabel="x (pixel)", ylabel="y (pixel)",
                 title=episode["label"], aspect="equal")
        artists.append((episode, blocks, agent, path))
    duration = max(float(episode["arrays"]["time"][-1]) for episode in episodes)
    frame_times = np.arange(int(np.ceil(duration * fps)) + 1) / fps

    def draw(time):
        for episode, blocks, agent, path in artists:
            arrays = episode["arrays"]
            index = int(np.clip(np.searchsorted(arrays["time"], time, side="right") - 1,
                                0, len(arrays["states"]) - 1))
            state = arrays["states"][index]
            for block, vertices in zip(blocks, transformed_block(state[2:5])):
                block.set_xy(vertices)
            agent.center = state[:2]
            path.set_data(arrays["states"][:index + 1, 0], arrays["states"][:index + 1, 1])
        return []

    figure.tight_layout()
    movie = FuncAnimation(figure, draw, frames=frame_times, interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps) if movie_format == "gif" else FFMpegWriter(fps=fps, codec="libx264", extra_args=["-threads", "1", "-filter_threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    movie.save(output / f"closed_loop.{movie_format}", writer=writer, dpi=110)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path, help="Saved per-start episode directories")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=("none", "gif", "mp4"), default="none")
    parser.add_argument("--fps", type=int, default=10, help="Frames per simulated second")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    episodes = [load_episode(directory) for directory in args.episodes]
    if len(episodes) > 1:
        initial = episodes[0]["arrays"]["states"][0]
        goal = episodes[0]["arrays"]["goal_pose"]
        for episode in episodes[1:]:
            if (not np.allclose(episode["arrays"]["states"][0], initial, atol=1e-9, rtol=0)
                    or not np.array_equal(episode["arrays"]["goal_pose"], goal)):
                parser.error("Comparison episodes must share the same initial state and goal")
    args.output.mkdir(parents=True, exist_ok=True)
    plot_traces(episodes, args.output)
    if args.format != "none":
        animate(episodes, args.output, args.format, args.fps)
    manifest = {"episodes": [str(episode["directory"].resolve()) for episode in episodes],
                "animation": args.format, "fps": args.fps,
                "controls": "Solid: applied position targets. Dashed, when different: unclipped proposals.",
                "geometry": "Original Push-T T-block vertices; body-origin poses; agent radius 15 pixels.",
                "new_simulator_steps": 0}
    manifest["input_sha256"] = {
        str((episode["directory"] / filename).resolve()): hashlib.sha256(
            (episode["directory"] / filename).read_bytes()).hexdigest()
        for episode in episodes for filename in ("trajectory.npz", "result.json", "metadata.json")
        if (episode["directory"] / filename).is_file()
    }
    (args.output / "visualization.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
