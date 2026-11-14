"""Render the 50 frozen BD-chi-squared PushT trials for the public gallery.

    .venv-pusht/bin/python scripts/plots/pusht_gallery.py \
        --root results/paper --output website/build --workers 6

Only saved poses are drawn: no controller, simulator, interpolation, or temporal
smoothing is used. Every recorded sample, including t=0 and the terminal pose,
occupies one frame at the native 10 Hz. The last pose therefore remains visible
for one frame beyond the recorded trajectory duration. ``duration_s`` in the
gallery data is the recorded terminal time, not this extra display interval.

The existing paper validators check input hashes, protocol, coverage, and
outcomes. Per-trial render records allow interrupted builds to resume after
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
SEEDS = list(range(300005, 300055))
SIZE = 512
FPS = 10
GREEN = "#2ca02c"
PREVIEW_SEEDS = (300013, 300021, 300044, 300052,
                 300015, 300023, 300028, 300053)


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
    protocol = pusht.load_protocol(root)
    if protocol["test_seeds"] != SEEDS:
        raise ValueError("Expected the frozen 50 seeds 300005 through 300054")
    directory = root / "pusht" / METHOD
    recorded = sorted(int(path.name.removeprefix("seed_"))
                      for path in directory.glob("seed_*") if path.is_dir())
    if recorded != SEEDS:
        raise ValueError("Missing or unexpected BD-chi-squared final records")
    trials = [pusht.load_trial(directory / f"seed_{seed}", METHOD, seed, protocol)
              for seed in SEEDS]
    if sum(bool(trial.result["success"]) for trial in trials) != 19:
        raise ValueError("Expected 19 successes and 31 failures in the frozen evaluation")
    for trial in trials:
        if trial.geometry != trials[0].geometry:
            raise ValueError("Expected the same task geometry in all 50 trials")
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


def contact_sheet(output, font_path):
    # Four successful and four unsuccessful starts, all shown at their initial pose.
    tile, gap, margin, heading = 224, 10, 18, 50
    sheet = Image.new("RGB", (4 * tile + 3 * gap + 2 * margin,
                             2 * tile + gap + heading + margin), "white")
    painter = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(font_path, 27)
    painter.text((sheet.width / 2, 24), "All 50 trials", font=font,
                 fill="#354135", anchor="mm")
    for index, seed in enumerate(PREVIEW_SEEDS):
        with Image.open(output / "media/pusht" / f"seed_{seed}.jpg") as poster:
            # Omit the small video status footer from this compact linked preview.
            image = poster.crop((0, 0, SIZE, 480)).resize((tile, round(tile * 480 / SIZE)),
                                                       Image.Resampling.LANCZOS)
            sheet.paste(image, (margin + (index % 4) * (tile + gap),
                               heading + (index // 4) * (tile + gap)))
    destination = common.REPOSITORY / "media/teasers/pusht_gallery.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    sheet.save(temporary, optimize=True)
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=common.PAPER_RESULTS,
                        help="Extracted paper result root")
    parser.add_argument("--output", type=Path, default=common.REPOSITORY / "website/build")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            parser.error(f"{command} is required on PATH")
    root, output = args.root.resolve(), args.output.resolve()
    trials = load_trials(root)
    bounds = view_bounds(trials)
    print(f"Validated {len(trials)} saved trials: 19 successes, 31 failures.", flush=True)
    print(f"Shared workspace view: {bounds}; {SIZE} × {SIZE}, {FPS} fps, native poses.", flush=True)
    (output / "media/pusht").mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    font_path = font_manager.findfont(pusht.FONT_FAMILY, fallback_to_default=False)
    sources = {"renderer": sha256(Path(__file__)), "validators": sha256(Path(pusht.__file__)),
               "common": sha256(Path(common.__file__)), "font": sha256(font_path)}
    protocol_path = root / "protocols/pusht.json"
    if not protocol_path.exists():
        protocol_path = root / "pusht/protocol.json"
    tasks = []
    for trial in trials:
        directory = root / "pusht" / METHOD / f"seed_{trial.seed}"
        paths = [protocol_path, *(directory / name for name in
                  ("trajectory.npz", "result.json", "metadata.json", "record.json")
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
            print(f"[{done:02d}/50] {seed}: {status}; {record['frames']} frames checked", flush=True)

    if sorted(records) != SEEDS:
        raise ValueError("Not all 50 media pairs were validated")
    # Recheck the pinned source files before publishing a complete gallery index.
    for task in tasks:
        for name, expected in task[-1].items():
            if sha256(root / name) != expected:
                raise ValueError(f"Input changed during rendering: {root / name}")
    rows = [{"seed": trial.seed, "success": bool(trial.result["success"]),
             "peak_coverage": float(trial.result["max_coverage"]),
             "final_coverage": float(trial.result["final_coverage"]),
             "duration_s": float(trial.time[-1]),
             "video": f"media/pusht/seed_{trial.seed}.mp4",
             "poster": f"media/pusht/seed_{trial.seed}.jpg"} for trial in trials]
    write_json(output / "data/trials.json", {"method": LABEL, "total": 50,
                                             "successes": 19, "failures": 31, "trials": rows})
    contact_sheet(output, font_path)
    total_bytes = sum((output / row[key]).stat().st_size for row in rows for key in ("video", "poster"))
    print(f"Gallery complete: 50 verified videos and posters, {total_bytes / 1024**2:.1f} MiB.", flush=True)
    print(output / "data/trials.json", flush=True)


if __name__ == "__main__":
    main()
