"""Plot saved experiment runs; optionally render their closed-loop motion.

Uses recorded states and actions. No experiments are rerun.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/benchmark")
    parser.add_argument("--group", choices=("control", "pusht", "all"), default="all")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--videos", action="store_true", help="Also render MP4 motion; requires FFmpeg")
    args = parser.parse_args()
    tasks = ("cartpole", "rocket", "reacher") if args.group == "control" else ("pusht",) if args.group == "pusht" else ("cartpole", "rocket", "reacher", "pusht")
    records = []
    for task in tasks:
        for directory in sorted((args.root / task).glob("*/seed_*")):
            if not (directory / "result.json").is_file():
                continue
            if args.methods and directory.parent.name not in args.methods:
                continue
            if args.seeds and int(directory.name.removeprefix("seed_")) not in args.seeds:
                continue
            records.append((task, directory))
    if not records:
        parser.error(f"No completed runs match under {args.root}")
    for task, directory in records:
        output = args.root / "figures" / task / directory.parent.name / directory.name
        if task == "pusht":
            command = ["pusht_video.py", str(directory), "--output", str(output),
                       "--format", "mp4" if args.videos else "none"]
        else:
            command = ["visualize.py", "--record", str(directory), "--output", str(output)]
            if not args.videos:
                command.append("--plots-only")
        subprocess.run([sys.executable, str(ROOT / "scripts/plots" / command[0]), *command[1:]], check=True)
    if any(task == "pusht" for task, _ in records):
        sys.path.insert(0, str(ROOT))
        from scripts.plots import pusht
        selected = {}
        protocol = pusht.load_protocol(args.root)
        for task, directory in records:
            if task == "pusht":
                method = directory.parent.name
                trial = pusht.load_trial(directory, method, int(directory.name.removeprefix("seed_")), protocol)
                selected.setdefault(method, []).append(trial)
        pusht.plot_coverage(selected, args.root / "figures/pusht")


if __name__ == "__main__":
    main()
