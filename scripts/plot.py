"""Regenerate the paper figures from saved records; no controllers are run.

Use the benchmark environment for --group control and the Push-T environment
with plotting dependencies for --group pusht or --group all.
"""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/paper")
    parser.add_argument("--group", choices=("control", "pusht", "all"), default="all")
    args = parser.parse_args()
    records = args.root.resolve()

    def run(name, *arguments):
        subprocess.run([sys.executable, str(ROOT / "scripts/plots" / f"{name}.py"),
                        "--root", str(records), *arguments], check=True)

    if args.group in ("control", "all"):
        run("cartpole", "--skip-video")
        run("rocket")
        run("reacher")
        run("combine_trajectories")
        run("stack_benchmark_metrics")
    if args.group in ("pusht", "all"):
        run("pusht")
        run("pusht", "--coverage-only", "--coverage-font-scale", "2",
            "--output", str(records / "figures/pusht/font_2x"))
        run("bd_chi2_successes")


if __name__ == "__main__":
    main()
