"""Summarize benchmark runs from saved records; no controller or simulator calls."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarks.reporting import aggregate_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/benchmark")
    parser.add_argument("--tasks", nargs="+", choices=["cartpole", "rocket", "reacher", "pusht"])
    parser.add_argument("--expert-root", type=Path, help="Optional directory containing paired MPC runs")
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    report = aggregate_results(arguments.root, tasks=arguments.tasks, expert_root=arguments.expert_root,
                               require_complete=arguments.require_complete)
    print(json.dumps({"root": str(arguments.root), "complete": report["complete"],
                      "controller_records": report["controller_records"], "rows": len(report["rows"])}))


if __name__ == "__main__":
    main()
