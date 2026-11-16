"""Summarize saved episodes without running policies or simulators.

Costs use successful runs and an expert from the same evaluation conditions.
Reacher reference completion and constraint-respecting success are kept separate.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tempfile

import numpy as np


TASKS = ("cartpole", "rocket", "reacher", "pusht")
METHODS = {
    "bd_chi2": "BD-χ²", "select_dpc": "Select-DPC", "bd_mean": "BD-mean",
    "bd_sample": "BD-sample", "bd_kl": "BD-KL", "bd_mean_chi2": "BD-mean-χ²",
    "windowed_deepc": "Windowed DeePC", "deepc": "DeePC",
}
STATE_WEIGHTS = {
    "cartpole": [5, 0, 10000, .1, .1],
    "reacher": [0, 0, 0, 0, 40000, 40000, 10, 10],
    "rocket": [40, 20, 20, 1, 3000, 300],
}
ACTION_WEIGHTS = {"cartpole": [1], "reacher": [10, 10], "rocket": [10, 10, 10]}


def statistics_summary(values):
    values = list(values)
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    temporary.replace(path)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Inputs:
    """Track only the local files actually read, with relocatable manifest paths."""

    def __init__(self, root):
        self.root = root
        self.paths = {}

    def read(self, path):
        content = path.read_bytes()
        self.paths[path] = hashlib.sha256(content).hexdigest()
        return content

    def json(self, path):
        return json.loads(self.read(path))

    def arrays(self, path):
        with np.load(io.BytesIO(self.read(path)), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def manifest(self):
        entries = {}
        for path, expected in self.paths.items():
            _require(_digest(path) == expected, f"Input changed while reporting: {path}")
            key = str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
            entries[key] = expected
        return entries


def verify_imported_records(root):
    """Verify the included result files."""
    path = root / "import_manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    for entry in manifest["files"]:
        file = root / entry["path"]
        _require(file.is_file() and _digest(file) == entry["sha256"], f"Changed imported file: {file}")
    return {"files": len(manifest["files"]), "manifest_sha256": _digest(path)}


def _controller_failed(result):
    return bool(result.get("controller_failure") or result.get("job_failure")
                or result.get("controller_error"))


def episode_metrics(task, result, arrays, verify=True, configuration=None):
    """Recompute outcome and cost using the saved pre-action observations."""
    configuration = configuration or {}
    steps = int(result["steps"])
    failure = _controller_failed(result)
    _require(len(arrays["actions"]) == steps, "Action count differs from recorded steps")
    _require(len(arrays["observations"]) == steps + 1, "Observation/action alignment differs")
    metrics = {"steps": steps, "controller_failure": failure,
               "terminal_reason": result.get("terminal_reason", result.get("termination_reason"))}
    if task == "pusht":
        coverage = arrays["coverage"]
        _require(len(coverage) == steps + 1, "Coverage/action alignment differs")
        # Upstream reward and the saved study evaluate coverage after actions.
        maximum = float(np.max(coverage[1:])) if steps else 0.0
        score = min(maximum / .95, 1.0)
        success = bool(steps and coverage[-1] > .95 and not failure)
        if verify:
            _require(math.isclose(score, result["score"], rel_tol=0, abs_tol=1e-12), "Push-T score mismatch")
            _require(success == bool(result["success"]), "Push-T success mismatch")
            if steps and np.any(coverage[1:-1] > .95):
                raise ValueError("Push-T continued after crossing its success threshold")
        return {**metrics, "success": success, "completion_success": success,
                "score": score, "max_coverage": maximum, "final_coverage": float(coverage[-1])}

    observations = arrays["observations"]
    _require(len(arrays["references"]) == steps, "Reference/action alignment differs")
    errors = observations[:-1] - arrays["references"]
    cost = float(np.sum(errors ** 2 * configuration.get("state_weights", STATE_WEIGHTS[task]))
                 + np.sum(arrays["actions"] ** 2 * configuration.get("action_weights", ACTION_WEIGHTS[task])))
    if verify:
        for name in ("cost", "trace_cost"):
            recorded = result.get(name)
            if recorded is not None:
                _require(np.isclose(cost, recorded, rtol=1e-11, atol=1e-7), f"{name} mismatch")
        _require(result.get("cost") is not None or failure, "Missing cost without controller failure")
    metrics["cost"] = result.get("cost")
    metrics["recomputed_trace_cost"] = cost

    if task == "cartpole":
        states = arrays["states"][1:]
        angles = np.arctan2(np.sin(states[:, 1]), np.cos(states[:, 1]))
        stable = ((np.abs(angles) <= np.deg2rad(15)) & (np.abs(states[:, 3]) <= 1)
                  & (np.abs(states[:, 0]) <= 1))
        horizon = int(result.get("evaluation_steps", 300))
        success = steps == horizon and len(stable) >= 50 and bool(stable[-50:].all()) and not failure
        completion = success
    elif task == "reacher":
        target_count = len(arrays["targets"])
        tolerance = configuration.get("task", {}).get("target_tolerance", .005)
        height_limit = configuration.get("height_limit", .1)
        completed = 0
        counts = []
        for observation in observations:
            if completed < target_count and np.linalg.norm(observation[4:6] - arrays["targets"][completed]) < tolerance:
                completed += 1
            counts.append(completed)
        completion = completed == target_count and not failure and steps <= int(result.get("evaluation_steps", 500))
        post_action_violations = int(np.sum(observations[1:, 5] > height_limit))
        initial_violation = bool(observations[0, 5] > height_limit)
        success = completion and post_action_violations == 0 and not initial_violation
        if verify:
            _require(np.array_equal(counts, arrays["target_completion_counts"]), "Reacher reference counts mismatch")
            _require(post_action_violations == int(result["violations_strict"]), "Reacher height counts mismatch")
            _require(bool(result["clean_success_strict"]) == success, "Reacher strict success mismatch")
            if completion:
                _require(counts[-2] < target_count and metrics["terminal_reason"] == "targets_completed",
                         "Reacher did not stop on its last reference")
        metrics.update(references_completed=completed, height_violations=post_action_violations,
                       initial_height_violation=initial_violation,
                       max_height_excess_m=float(max(0.0, np.max(observations[:, 5]) - height_limit)))
    else:
        success = bool(result["settled"] and result["contact_success"] and not failure)
        completion = success
        if verify and success:
            _require(metrics["terminal_reason"] == "settled", "Rocket landing termination mismatch")
            if "terminated" in arrays:
                _require(steps and bool(arrays["terminated"][-1]) and arrays["rewards"][-1] > 0,
                         "Rocket has no positive terminal landing")
    if verify:
        # Reacher's recorder calls reference completion `success`.
        _require(bool(result["success"]) == completion, f"{task} recorded success mismatch")
    return {**metrics, "success": bool(success), "completion_success": bool(completion)}


def _read_episode(directory, task, seed, inputs, verify):
    result = inputs.json(directory / "result.json")
    metadata = inputs.json(directory / "metadata.json")
    _require(int(result["seed"]) == seed and result.get("task", task) == task, f"Wrong episode identity: {directory}")
    _require(result.get("record_status", "completed") == "completed", f"Unfinished episode: {directory}")
    arrays = inputs.arrays(directory / "trajectory.npz")
    if verify and metadata.get("trajectory_sha256"):
        _require(_digest(directory / "trajectory.npz") == metadata["trajectory_sha256"], f"Trajectory hash mismatch: {directory}")
    marker = directory / "record.json"
    if verify and marker.is_file():
        saved = inputs.json(marker)
        for name in ("result.json", "metadata.json", "trajectory.npz"):
            _require(_digest(directory / name) == saved["sha256"][name], f"Completed file changed: {directory / name}")
    metrics = episode_metrics(task, result, arrays, verify, metadata.get("config"))
    metrics.update(seed=seed, expert_comparable=bool(metadata.get("expert_comparable", False)))
    return result, metadata, arrays, metrics


def _expert(task, seed, root, expert_root, inputs, metadata):
    for base in (root, expert_root):
        if base is None:
            continue
        path = base / task / "mpc_expert" / f"seed_{seed}" / "result.json"
        if path.is_file():
            result = inputs.json(path)
            arrays_path = path.parent / "trajectory.npz"
            arrays = inputs.arrays(arrays_path) if arrays_path.is_file() else None
            return result, arrays
    if metadata.get("paired_expert") is not None:
        return metadata["paired_expert"], None
    path = Path(__file__).resolve().parents[2] / "data" / task / "expert_evaluations.json"
    if path.is_file():
        saved = inputs.json(path)
        matching = [record for record in saved["runs"] if int(record["seed"]) == seed]
        _require(len(matching) <= 1, "Duplicate saved expert start")
        if matching:
            return {**matching[0]["result"], "seed": seed}, None
    return None, None


def _row(task, method, records, expected_seeds):
    successful = [record for record in records if record["success"]]
    row = {"task": task, "method": method, "label": METHODS[method],
           "expected_trials": len(expected_seeds), "completed_trials": len(records),
           "successful_trials": len(successful), "successful_seeds": [r["seed"] for r in successful],
           "completion_only_successes": sum(record["completion_success"] for record in records),
           "zero_action_failures": sum(not record["success"] and not record["steps"] for record in records),
           "records": records}
    if task == "pusht":
        row.update(score=statistics_summary(r["score"] for r in records),
                   median_score=statistics.median(r["score"] for r in records),
                   peak_coverage=statistics_summary(r["max_coverage"] for r in records),
                   final_coverage=statistics_summary(r["final_coverage"] for r in records))
    else:
        row["cost_ratio"] = statistics_summary(r["cost_ratio"] for r in successful if r["cost_ratio"] is not None)
    return row


def timing_rows(root, inputs):
    """Aggregate complete, action-validated timing replays by task and method."""
    groups = {}
    for path in sorted((root / "timing").glob("*/*/seed_*/result.json")):
        trial = inputs.json(path)
        if trial.get("state") == "no_actions":
            continue
        _require(trial.get("validated_replay") is True and trial.get("complete_sequence") is True
                 and not trial.get("pilot", False),
                 f"Unvalidated or incomplete timing replay: {path}")
        _require(trial.get("decisions", 0) > 0, "Timing replay has no decisions")
        value = trial.get("cpu_ms_per_action")
        _require(value is not None and math.isfinite(value) and value > 0, "Invalid timing value")
        groups.setdefault((trial["task"], trial["method"]), []).append(trial)
    rows = []
    for (task, method), trials in sorted(groups.items()):
        seeds = [trial["seed"] for trial in trials]
        _require(len(seeds) == len(set(seeds)), "Duplicate timing start")
        rows.append({"task": task, "method": method, "seeds": seeds,
                     "cpu_ms_per_action": statistics_summary(trial["cpu_ms_per_action"] for trial in trials)})
    return rows


def _stat_text(value, digits=3):
    if not value["n"]:
        return "—"
    mean = f'{value["mean"]:.{digits}f}'
    return mean if value["sample_sd"] is None else f'{mean} ± {value["sample_sd"]:.{digits}f}'


def _markdown(report):
    rows = report["rows"]
    lines = ["# Benchmark results", "",
             "Cost / expert is the mean ± sample standard deviation of paired cost ratios over successful runs. "
             "Ratios require matching evaluation conditions; successful subsets can differ between methods. "
             "One successful run has no sample standard deviation; no eligible runs gives —.", ""]
    control_rows = [row for row in rows if row["task"] != "pusht"]
    if control_rows:
        lines += ["| Task | Method | Successes | Cost / expert |", "|---|---|---:|---:|"]
        for row in control_rows:
            lines.append(f'| {row["task"].title()} | {row["label"]} | {row["successful_trials"]}/{row["completed_trials"]} | '
                         f'{_stat_text(row["cost_ratio"])} |')
        if any(row["task"] == "reacher" for row in control_rows):
            lines += ["", "Reacher success requires completing the ordered references within the configured budget "
                      "without a recorded height-limit violation. Completion alone is saved separately. "
                      "Paired expert costs are not filtered by expert height violations."]
    push_rows = [row for row in rows if row["task"] == "pusht"]
    if push_rows:
        lines += ["", "Push-T coverage includes every saved trial; success requires coverage strictly above 95%. "
                  "Final coverage is measured at termination.", "",
                  "| Method | Trials | Peak coverage (0–1) | Final coverage (0–1) |", "|---|---:|---:|---:|"]
        for row in push_rows:
            lines.append(f'| {row["label"]} | {row["completed_trials"]} | '
                         f'{_stat_text(row["peak_coverage"], 4)} | {_stat_text(row["final_coverage"], 4)} |')
    if report["timing"]:
        lines += ["", "Timing averages equally weighted per-start process-CPU means (ms/action), with one numerical thread. "
                  "Initialization, simulation and task overrides are excluded. CPU affinity and background load are recorded; "
                  "CPU reservation is not enforced.", "",
                  "| Task | Method | Starts | CPU ms/action |", "|---|---|---:|---:|"]
        for row in report["timing"]:
            lines.append(f'| {row["task"].title()} | {METHODS[row["method"]]} | {len(row["seeds"])} | '
                         f'{_stat_text(row["cpu_ms_per_action"])} |')
    return "\n".join(lines) + "\n"


def aggregate_results(root, tasks=None, verify=True, require_complete=False, expert_root=None):
    """Write summaries for complete saved episodes under ``root/TASK/METHOD``.

    A task runner may call this after each batch. Missing tasks, methods or starts
    make the report partial; ``require_complete`` turns these into an error.
    """
    root = Path(root).resolve()
    expert_root = Path(expert_root).resolve() if expert_root is not None else None
    tasks = tuple(tasks) if tasks is not None else tuple(task for task in TASKS if (root / task).is_dir())
    _require(tasks and set(tasks) <= set(TASKS), "Select at least one known task")
    imported = verify_imported_records(root) if verify else None
    inputs = Inputs(root)
    rows = []
    complete = True
    for task in tasks:
        protocol_path = root / "protocols" / f"{task}.json"
        protocol = inputs.json(protocol_path) if protocol_path.exists() else None
        if protocol is not None:
            method_names = [method["id"].removesuffix("_upstream") for method in protocol["methods"]]
            expected_seeds = sorted(start[0] for start in protocol["test_starts"])
        else:
            method_names = [method for method in METHODS if (root / task / method).is_dir()]
            expected_seeds = []
            complete = False
        for method in method_names:
            directories = sorted((root / task / method).glob("seed_*/result.json"))
            if not directories:
                complete = False
                continue
            records = []
            for result_path in directories:
                directory = result_path.parent
                seed = int(directory.name.removeprefix("seed_"))
                result, metadata, arrays, metrics = _read_episode(directory, task, seed, inputs, verify)
                if task != "pusht":
                    expert, expert_arrays = _expert(task, seed, root, expert_root, inputs, metadata)
                    expert_cost = expert.get("cost") if expert else result.get("expert_cost")
                    if expert:
                        _require(int(expert["seed"]) == seed and bool(expert["success"]), "Invalid paired expert")
                    if verify and expert_arrays is not None:
                        episode_metrics(task, expert, expert_arrays, verify=True, configuration=metadata.get("config"))
                        _require(np.array_equal(arrays["states"][0], expert_arrays["states"][0]), "Expert initial state differs")
                        if task == "reacher":
                            _require(np.array_equal(arrays["targets"], expert_arrays["targets"]), "Expert references differ")
                    ratio = None
                    if metrics["success"] and metrics["expert_comparable"]:
                        _require(expert_cost is not None and math.isfinite(expert_cost) and expert_cost > 0,
                                 f"Missing valid expert cost for {task}/{method}/{seed}")
                        ratio = metrics["cost"] / expert_cost
                    metrics.update(expert_cost=expert_cost, cost_ratio=ratio)
                records.append(metrics)
            row = _row(task, method, records, expected_seeds)
            complete = complete and sorted(r["seed"] for r in records) == expected_seeds
            rows.append(row)
    if require_complete:
        _require(complete, "Incomplete benchmark: every configured method and start is required")
    report = {"schema_version": 1, "tasks": list(tasks), "complete": bool(complete),
              "controller_records": sum(row["completed_trials"] for row in rows),
              "success_definition": {"reacher": "Five ordered references and no strict height violation",
                                     "other_tasks": "Configured task-specific success criterion"},
              "cost_population": "Successful controller trials with matching expert evaluation conditions, paired by start",
              "cost_std_ddof": 1, "rows": rows, "timing": timing_rows(root, inputs),
              "import_verification": imported}
    outputs = {"summary.json": json.dumps(report, indent=2, allow_nan=False) + "\n", "README.md": _markdown(report)}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["task", "method", "completed", "expected", "successes", "completion_only_successes",
                                               "cost_ratio_n", "cost_ratio_mean", "cost_ratio_sample_sd", "score_mean", "score_sample_sd"])
    writer.writeheader()
    for row in rows:
        costs = row.get("cost_ratio", {})
        scores = row.get("score", {})
        writer.writerow({"task": row["task"], "method": row["method"], "completed": row["completed_trials"],
                         "expected": row["expected_trials"], "successes": row["successful_trials"],
                         "completion_only_successes": row["completion_only_successes"], "cost_ratio_n": costs.get("n"),
                         "cost_ratio_mean": costs.get("mean"), "cost_ratio_sample_sd": costs.get("sample_sd"),
                         "score_mean": scores.get("mean"), "score_sample_sd": scores.get("sample_sd")})
    outputs["summary.csv"] = buffer.getvalue()
    hashes = inputs.manifest()
    for name, content in outputs.items():
        _write(root / name, content)
    manifest = {"input_sha256": hashes, "inputs_unchanged": True, "new_controller_calls": 0,
                "new_simulator_steps": 0, "generator_sha256": _digest(Path(__file__)),
                "output_sha256": {name: _digest(root / name) for name in outputs}}
    _write(root / "report_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return report
