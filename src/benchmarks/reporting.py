"""Build paper metrics from saved episodes, without running policies or simulators.

Reacher paper success means five ordered references and no height-limit violation.
Its completion-only outcome is retained separately. Expert costs remain paired by
start; the paper does not discard denominators for tiny expert height violations.
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
SEEDS = {
    "cartpole": list(range(10003, 10013)),
    "rocket": list(range(3000, 3010)),
    "reacher": list(range(3000, 3005)) + list(range(4000, 4005)),
    "pusht": list(range(300005, 300055)),
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


def episode_metrics(task, result, arrays, verify=True):
    """Recompute outcome and cost using the saved pre-action observations."""
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
    cost = float(np.sum(errors ** 2 * STATE_WEIGHTS[task])
                 + np.sum(arrays["actions"] ** 2 * ACTION_WEIGHTS[task]))
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
        completed = 0
        counts = []
        for observation in observations:
            if completed < 5 and np.linalg.norm(observation[4:6] - arrays["targets"][completed]) < .005:
                completed += 1
            counts.append(completed)
        completion = completed == 5 and not failure and steps <= int(result.get("evaluation_steps", 500))
        post_action_violations = int(np.sum(observations[1:, 5] > .1))
        initial_violation = bool(observations[0, 5] > .1)
        success = completion and post_action_violations == 0 and not initial_violation
        if verify:
            _require(np.array_equal(counts, arrays["target_completion_counts"]), "Reacher reference counts mismatch")
            _require(post_action_violations == int(result["violations_strict"]), "Reacher height counts mismatch")
            _require(bool(result["clean_success_strict"]) == success, "Reacher strict success mismatch")
            if completion:
                _require(counts[-2] < 5 and metrics["terminal_reason"] == "targets_completed",
                         "Reacher did not stop on its fifth reference")
        metrics.update(references_completed=completed, height_violations=post_action_violations,
                       initial_height_violation=initial_violation,
                       max_height_excess_m=float(max(0.0, np.max(observations[:, 5]) - .1)))
    else:
        success = bool(result["settled"] and result["contact_success"] and not failure)
        completion = success
        if verify and success:
            _require(metrics["terminal_reason"] == "settled", "Rocket landing termination mismatch")
            if "terminated" in arrays:
                _require(steps and bool(arrays["terminated"][-1]) and arrays["rewards"][-1] > 0,
                         "Rocket has no positive terminal landing")
    if verify:
        # The research records call reference completion `success` for Reacher.
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
    metrics = episode_metrics(task, result, arrays, verify)
    metrics.update(seed=seed, paper_protocol=bool(result.get("paper_protocol", metadata.get("paper_protocol", True))))
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
    return metadata.get("paired_expert"), None


def _row(task, method, records):
    successful = [record for record in records if record["success"]]
    row = {"task": task, "method": method, "label": METHODS[method],
           "planned_trials": len(SEEDS[task]), "completed_trials": len(records),
           "successful_trials": len(successful), "successful_seeds": [r["seed"] for r in successful],
           "completion_only_successes": sum(record["completion_success"] for record in records),
           "paper_protocol": all(record["paper_protocol"] for record in records),
           "zero_action_failures": sum(not record["success"] and not record["steps"] for record in records),
           "records": records}
    if task == "pusht":
        row.update(score=statistics_summary(r["score"] for r in records),
                   median_score=statistics.median(r["score"] for r in records))
    else:
        row["cost_ratio"] = statistics_summary(r["cost_ratio"] for r in successful if r["cost_ratio"] is not None)
    return row


def timing_rows(root, inputs):
    """Aggregate per-start process-CPU timing means."""
    rows = []
    for name in ("required", "deepc", "reacher_deepc"):
        path = root / "timing" / f"{name}.json"
        if not path.exists():
            continue
        cohort = inputs.json(path)
        _require(cohort.get("publication_eligible") is True, f"Unvalidated timing cohort: {name}")
        trials = cohort["trials"]
        _require(len(trials) == cohort["planned_trials"] == cohort["eligible_trials"], "Incomplete timing cohort")
        groups = {}
        for trial in trials:
            _require(trial["eligible"], "Ineligible timing trial")
            if trial["method"] != "bd_kl":
                _require(trial["state"] == "completed" and trial["resource_guard"]["passed"], "Failed timing replay/resource guard")
                _require(trial["decisions"] > 0, "Timing trial has no decisions")
            else:
                _require(trial["no_actions"] and trial["decisions"] == 0, "BD-KL must have undefined timing")
            groups.setdefault((trial["task"], trial["method"]), []).append(trial)
        for (task, method), group in groups.items():
            _require(len(group) == 3 and len({trial["seed"] for trial in group}) == 3, "Timing requires three distinct starts")
            values = [] if method == "bd_kl" else [trial["cpu_ms_per_action"] for trial in group]
            measured = statistics_summary(values)
            original = next(row for row in cohort["rows"] if row["task"] == task and row["method"] == method)
            if values:
                for key in ("mean", "sample_sd"):
                    _require(math.isclose(measured[key], original["cpu_ms_per_action"][key], rel_tol=1e-12), "Timing aggregation mismatch")
            rows.append({"task": task, "method": method, "cpu_ms_per_action": measured,
                         "seeds": [trial["seed"] for trial in group], "source_cohort": name})
    _require(len({(r["task"], r["method"]) for r in rows}) == len(rows), "Duplicate timing rows")
    return sorted(rows, key=lambda row: (TASKS.index(row["task"]), tuple(METHODS).index(row["method"])))


def _stat_text(value, digits=3):
    if not value["n"]:
        return "—"
    mean = f'{value["mean"]:.{digits}f}'
    return mean if value["sample_sd"] is None else f'{mean} ± {value["sample_sd"]:.{digits}f}'


def _markdown(report):
    rows = report["rows"]
    lines = ["# Paper benchmark results", ""]
    if not report["complete"]:
        lines += ["Partial results: only available trials are shown. Cost ratios are omitted for runs with a changed evaluation horizon.", ""]
    lines += ["Cost / expert is the mean ± sample standard deviation of method/expert cost ratios, paired by start "
             "and computed over successful trials only. Successful subsets differ between methods. "
             "A single successful trial has no standard deviation; no successes gives —.", "",
             "Reacher success requires all five references within 10 s and no recorded height above 0.1 m. "
             "The paired expert cost is not filtered by expert height violations.", ""]
    control_rows = [row for row in rows if row["task"] != "pusht"]
    if control_rows:
        lines += ["| Task | Method | Successes | Cost / expert |",
                  "|---|---|---:|---:|"]
        for row in control_rows:
            lines.append(f'| {row["task"].title()} | {row["label"]} | {row["successful_trials"]}/{row["completed_trials"]} | '
                         f'{_stat_text(row["cost_ratio"])} |')
    push_rows = [row for row in rows if row["task"] == "pusht"]
    if push_rows:
        lines += ["", "Push-T score is maximum post-action coverage / 0.95, capped at one. "
                  "Success requires coverage strictly above 95% within 300 actions.", "",
                  "| Method | Successes | Mean score ± sample SD | Median score |",
                  "|---|---:|---:|---:|"]
        for row in push_rows:
            lines.append(f'| {row["label"]} | {row["successful_trials"]}/{row["completed_trials"]} | '
                         f'{_stat_text(row["score"], 4)} | {row["median_score"]:.3f} |')
    if report["timing"]:
        lines += ["", "Action timing: mean ± sample SD of three equally weighted per-start process-CPU means "
                  "(ms/action), one numerical thread. Setup, simulation and Rocket touchdown overrides are excluded. "
                  "These were affinity-controlled measurements under monitored background load, not an unloaded-machine "
                  "comparison.", "",
                  "| Task | Method | CPU ms/action |", "|---|---|---:|"]
        for row in report["timing"]:
            lines.append(f'| {row["task"].title()} | {METHODS[row["method"]]} | {_stat_text(row["cpu_ms_per_action"])} |')
    lines += ["", "BD-KL produces no actions in the paper trials, so its action timing is undefined.", ""]
    return "\n".join(lines)


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
        method_names = ("bd_chi2", "select_dpc") if task == "pusht" else tuple(METHODS)
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
                        episode_metrics(task, expert, expert_arrays, verify=True)
                        _require(np.array_equal(arrays["states"][0], expert_arrays["states"][0]), "Expert initial state differs")
                        if task == "reacher":
                            _require(np.array_equal(arrays["targets"], expert_arrays["targets"]), "Expert references differ")
                    ratio = None
                    if metrics["success"] and metrics["paper_protocol"]:
                        _require(expert_cost is not None and math.isfinite(expert_cost) and expert_cost > 0,
                                 f"Missing valid expert cost for {task}/{method}/{seed}")
                        ratio = metrics["cost"] / expert_cost
                    metrics.update(expert_cost=expert_cost, cost_ratio=ratio)
                records.append(metrics)
            row = _row(task, method, records)
            complete = complete and row["paper_protocol"] and sorted(r["seed"] for r in records) == SEEDS[task]
            rows.append(row)
    if require_complete:
        _require(complete, "Incomplete paper cohort: every task method and prescribed start is required")
    report = {"schema_version": 1, "tasks": list(tasks), "complete": bool(complete),
              "controller_records": sum(row["completed_trials"] for row in rows),
              "success_definition": {"reacher": "Five ordered references and no strict height violation",
                                     "other_tasks": "Finalized task-specific paper criterion"},
              "cost_population": "Successful controller trials on the full paper protocol, paired by start",
              "cost_std_ddof": 1, "rows": rows, "timing": timing_rows(root, inputs),
              "import_verification": imported}
    outputs = {"summary.json": json.dumps(report, indent=2, allow_nan=False) + "\n", "README.md": _markdown(report)}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["task", "method", "completed", "planned", "successes", "completion_only_successes",
                                               "cost_ratio_n", "cost_ratio_mean", "cost_ratio_sample_sd", "score_mean", "score_sample_sd"])
    writer.writeheader()
    for row in rows:
        costs = row.get("cost_ratio", {})
        scores = row.get("score", {})
        writer.writerow({"task": row["task"], "method": row["method"], "completed": row["completed_trials"],
                         "planned": row["planned_trials"], "successes": row["successful_trials"],
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
