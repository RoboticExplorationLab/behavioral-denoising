"""Render saved BD-χ² Push-T trials for a standalone video gallery.

    .venv-pusht/bin/python scripts/plots/pusht_gallery.py \
        --root runs/benchmark --output website/build --workers 6

Only saved poses are drawn: no controller, simulator, interpolation, or temporal
smoothing is used. Every recorded sample, including t=0 and the terminal pose,
occupies one frame at the native 10 Hz. The last pose therefore remains visible
for one frame beyond the recorded trajectory duration. ``duration_s`` in the
gallery data is the recorded terminal time, not this extra display interval.

Saved-record validation checks input hashes, protocol, coverage, and outcomes. Per-trial render records allow interrupted builds to resume after
checking inputs, output hashes, decoded frame counts, and encoding properties.
Requires numpy, matplotlib, shapely, pillow, and FFmpeg/ffprobe on PATH.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


# Set these before importing numerical libraries, including in spawned workers.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pusht-gallery-mpl"))
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.plots import common, pusht


METHOD = "bd_chi2"
LABEL = "BD-χ²"
SIZE = 512
FPS = 10
GREEN = "#2ca02c"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_trials(root):
    trials = pusht.load_trials(root, [METHOD])[METHOD]
    expected = {key: value for key, value in trials[0].geometry.items() if key != "block_polygons"}
    for trial in trials:
        geometry = {key: value for key, value in trial.geometry.items() if key != "block_polygons"}
        if geometry != expected or not pusht.Polygon(trial.outline).equals(pusht.Polygon(trials[0].outline)):
            raise ValueError("Expected the same task geometry in all gallery trials")
    return trials


def view_bounds(trials):
    """Use one full-workspace view, expanding only for saved geometry outside it."""
    xmin, xmax, ymin, ymax = trials[0].geometry["workspace"]
    lower, upper = np.array([xmin, ymin]), np.array([xmax, ymax])
    for trial in trials:
        outlines = np.concatenate([pusht.transform(trial.outline, state[2:])
                                   for state in trial.states])
        goal = pusht.transform(trial.outline, trial.goal_pose)
        radius = float(trial.geometry["agent_radius"])
        lower = np.minimum(lower, np.minimum(outlines.min(axis=0), goal.min(axis=0)))
        lower = np.minimum(lower, trial.states[:, :2].min(axis=0) - radius)
        upper = np.maximum(upper, np.maximum(outlines.max(axis=0), goal.max(axis=0)))
        upper = np.maximum(upper, trial.states[:, :2].max(axis=0) + radius)
    if np.any(lower < [xmin, ymin]) or np.any(upper > [xmax, ymax]):
        # Eight world units provide room for edge strokes on the expanded view.
        lower, upper = lower - 8, upper + 8
    center = (lower + upper) / 2
    half_width = float(np.max(upper - lower)) / 2
    return [float(center[0] - half_width), float(center[0] + half_width),
            float(center[1] - half_width), float(center[1] + half_width)]


def scene(trial, bounds):
    plt.rcParams.update({"font.family": pusht.FONT_FAMILY, "font.size": 11})
    figure = plt.figure(figsize=(SIZE / 100, SIZE / 100), dpi=100, facecolor="white")
    axis = figure.add_axes([.035, .085, .93, .90])
    axis.set(xlim=bounds[:2], ylim=bounds[2:][::-1], aspect="equal")
    axis.set_axis_off()
    xmin, xmax, ymin, ymax = trial.geometry["workspace"]
    axis.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             facecolor="none", edgecolor="#e1e5df", linewidth=.7))
    axis.add_patch(Polygon(pusht.transform(trial.outline, trial.goal_pose),
                           facecolor="#e4eddc", edgecolor="#bacbae", linewidth=.8,
                           zorder=1))
    trail, = axis.plot([], [], color="#437d3e", linewidth=.75, alpha=.38,
                       linestyle=(0, (3, 3)), zorder=2)
    block = Polygon(pusht.transform(trial.outline, trial.states[0, 2:]),
                    facecolor=GREEN, edgecolor="#287c28", linewidth=.75, zorder=3)
    axis.add_patch(block)
    pusher = Circle(trial.states[0, :2], radius=trial.geometry["agent_radius"],
                    facecolor="#87908b", edgecolor="white", linewidth=.8, zorder=4)
    axis.add_patch(pusher)
    elapsed = figure.text(.06, .035, "", ha="left", va="center", color="#596259")
    coverage = figure.text(.94, .035, "", ha="right", va="center", color="#596259")
    centers = trial.centers

    def draw(index):
        block.set_xy(pusht.transform(trial.outline, trial.states[index, 2:]))
        pusher.center = trial.states[index, :2]
        trail.set_data(centers[:index + 1, 0], centers[:index + 1, 1])
        elapsed.set_text(f"{trial.time[index]:.1f} s")
        coverage.set_text(f"Coverage {100 * trial.coverage[index]:.1f}%")
        figure.canvas.draw()
        return figure.canvas.buffer_rgba()

    return figure, draw


def has_faststart(path):
    """Check the MP4 atom ordering without loading the video payload into memory."""
    atoms = []
    with path.open("rb") as source:
        total = path.stat().st_size
        while source.tell() + 8 <= total:
            start = source.tell()
            size, name = struct.unpack(">I4s", source.read(8))
            if size == 1:
                size = struct.unpack(">Q", source.read(8))[0]
            elif size == 0:
                size = total - start
            if size < source.tell() - start or start + size > total:
                return False
            atoms.append(name)
            source.seek(start + size)
    return b"moov" in atoms and b"mdat" in atoms and atoms.index(b"moov") < atoms.index(b"mdat")


def validate_media(video, poster, expected_frames):
    completed = subprocess.run([
        "ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-of", "json", str(video)], check=True, capture_output=True, text=True)
    stream, = json.loads(completed.stdout)["streams"]
    if (completed.stderr or stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p"
            or (stream["width"], stream["height"]) != (SIZE, SIZE)
            or Fraction(stream["r_frame_rate"]) != FPS
            or Fraction(stream["avg_frame_rate"]) != FPS
            or int(stream["nb_frames"]) != expected_frames
            or int(stream["nb_read_frames"]) != expected_frames
            or abs(float(stream["duration"]) - expected_frames / FPS) > 1e-5
            or not has_faststart(video)):
        raise ValueError(f"Unexpected video encoding or frame timing: {video}: {stream}")
    with Image.open(poster) as still:
        if still.size != (SIZE, SIZE) or still.format != "JPEG":
            raise ValueError(f"Unexpected poster format: {poster}")
        still.verify()
    return {"frames": expected_frames, "fps": FPS, "width": SIZE, "height": SIZE,
            "codec": "h264", "pixel_format": "yuv420p", "faststart": True,
            "encoded_duration_s": float(stream["duration"])}


def render_trial(task):
    trial, bounds, output, signature, inputs = task
    prefix = output / "media/pusht" / f"seed_{trial.seed}"
    video, poster = prefix.with_suffix(".mp4"), prefix.with_suffix(".jpg")
    record_path = prefix.with_suffix(".render.json")
    if record_path.exists() and video.exists() and poster.exists():
        try:
            record = json.loads(record_path.read_text())
            if (record["render_sha256"] == signature
                    and record["output_sha256"] == {"video": sha256(video), "poster": sha256(poster)}):
                validate_media(video, poster, len(trial.time))
                return trial.seed, "reused", record
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
            pass

    temporary_video = prefix.with_suffix(".partial.mp4")
    temporary_poster = prefix.with_suffix(".partial.jpg")
    figure, draw = scene(trial, bounds)
    process = None
    try:
        initial = np.asarray(draw(0))[:, :, :3].copy()
        Image.fromarray(initial).save(temporary_poster, quality=92, subsampling=0, optimize=True)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgba", "-s", f"{SIZE}x{SIZE}", "-r", str(FPS),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-threads", "1", "-filter_threads", "1", "-movflags", "+faststart",
            str(temporary_video)]
        # A file, rather than an unread pipe, prevents stderr from blocking FFmpeg.
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            try:
                for index in range(len(trial.time)):
                    process.stdin.write(draw(index))
                process.stdin.close()
                returncode = process.wait()
            except BaseException:
                process.kill()
                process.wait()
                raise
            if returncode:
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace"))
        checked = validate_media(temporary_video, temporary_poster, len(trial.time))
        temporary_video.replace(video)
        temporary_poster.replace(poster)
        record = {"seed": trial.seed, "render_sha256": signature, "input_sha256": inputs,
                  "output_sha256": {"video": sha256(video), "poster": sha256(poster)},
                  "workspace_view": bounds, "trajectory_duration_s": float(trial.time[-1]),
                  "native_saved_samples_only": True, "new_simulator_steps": 0, **checked}
        write_json(record_path, record)
        return trial.seed, "rendered", record
    finally:
        plt.close(figure)
        temporary_video.unlink(missing_ok=True)
        temporary_poster.unlink(missing_ok=True)


def contact_sheet(output, font_path, trials, destination):
    # Evenly spaced seeds show a range of starting poses without outcome selection.
    indices = np.linspace(0, len(trials) - 1, min(8, len(trials)), dtype=int)
    preview_seeds = [trials[index].seed for index in indices]
    tile, gap, margin, heading = 224, 10, 18, 50
    sheet = Image.new("RGB", (4 * tile + 3 * gap + 2 * margin,
                             2 * tile + gap + heading + margin), "white")
    painter = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(font_path, 27)
    painter.text((sheet.width / 2, 24), f"All {len(trials)} trials", font=font,
                 fill="#354135", anchor="mm")
    for index, seed in enumerate(preview_seeds):
        with Image.open(output / "media/pusht" / f"seed_{seed}.jpg") as poster:
            # Omit the small video status footer from this compact linked preview.
            image = poster.crop((0, 0, SIZE, 480)).resize((tile, round(tile * 480 / SIZE)),
                                                       Image.Resampling.LANCZOS)
            sheet.paste(image, (margin + (index % 4) * (tile + gap),
                               heading + (index // 4) * (tile + gap)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    sheet.save(temporary, optimize=True)
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=common.DEFAULT_RUNS,
                        help="Saved experiment output root")
    parser.add_argument("--output", type=Path, default=common.REPOSITORY / "website/build")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--preview", type=Path, help="Optional contact-sheet PNG")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            parser.error(f"{command} is required on PATH")
    root, output = args.root.resolve(), args.output.resolve()
    trials = load_trials(root)
    bounds = view_bounds(trials)
    print(f"Validated {len(trials)} saved trials; mean peak coverage "
          f"{100 * np.mean([trial.result['max_coverage'] for trial in trials]):.2f}%.", flush=True)
    print(f"Shared workspace view: {bounds}; {SIZE} × {SIZE}, {FPS} fps, native poses.", flush=True)
    (output / "media/pusht").mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    font_path = font_manager.findfont(pusht.FONT_FAMILY, fallback_to_default=False)
    sources = {"renderer": sha256(Path(__file__)), "validators": sha256(Path(pusht.__file__)),
               "common": sha256(Path(common.__file__)), "font": sha256(font_path)}
    protocol_path = root / "protocols/pusht.json"
    tasks = []
    for trial in trials:
        directory = root / "pusht" / METHOD / f"seed_{trial.seed}"
        paths = [protocol_path, *(directory / name for name in
                  ("trajectory.npz", "result.json", "metadata.json")
                  if (directory / name).exists())]
        inputs = {str(path.relative_to(root)): sha256(path) for path in paths}
        settings = {"sources": sources, "inputs": inputs, "bounds": bounds,
                    "size": SIZE, "fps": FPS}
        signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        tasks.append((trial, bounds, output, signature, inputs))
    records = {}
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(render_trial, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            seed, status, record = future.result()
            records[seed] = record
            print(f"[{done:02d}/{len(trials)}] {seed}: {status}; {record['frames']} frames checked", flush=True)

    if sorted(records) != sorted(trial.seed for trial in trials):
        raise ValueError("Not all media pairs were validated")
    # Recheck the pinned source files before publishing a complete gallery index.
    for task in tasks:
        for name, expected in task[-1].items():
            if sha256(root / name) != expected:
                raise ValueError(f"Input changed during rendering: {root / name}")
    rows = [{"seed": trial.seed,
             "peak_coverage": float(trial.result["max_coverage"]),
             "final_coverage": float(trial.result["final_coverage"]),
             "duration_s": float(trial.time[-1]),
             "video": f"media/pusht/seed_{trial.seed}.mp4",
             "poster": f"media/pusht/seed_{trial.seed}.jpg"} for trial in trials]
    write_json(output / "data/trials.json", {"method": LABEL, "total": len(rows), "trials": rows})
    if args.preview:
        contact_sheet(output, font_path, trials, args.preview)
    total_bytes = sum((output / row[key]).stat().st_size for row in rows for key in ("video", "poster"))
    print(f"Gallery complete: {len(trials)} verified videos and posters, {total_bytes / 1024**2:.1f} MiB.", flush=True)
    print(output / "data/trials.json", flush=True)


if __name__ == "__main__":
    main()
