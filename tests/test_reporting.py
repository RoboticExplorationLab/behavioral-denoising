"""Metrics distinguish completion, constraints, paired costs and timing validity."""

import json

import numpy as np
import pytest

from src.benchmarks.reporting import aggregate_results, episode_metrics, statistics_summary, timing_rows, Inputs


def cartpole_record(steps=50, success=True):
    observations = np.tile([0., 0., 1., 0., 0.], (steps + 1, 1))
    arrays = {"observations": observations, "states": np.zeros((steps + 1, 4)),
              "actions": np.ones((steps, 1)), "references": observations[:-1].copy()}
    result = {"steps": steps, "evaluation_steps": 50, "cost": float(steps), "success": success}
    return result, arrays


def save_episode(root, seed, result, arrays, *, comparable=True):
    directory = root / "cartpole/bd_chi2" / f"seed_{seed}"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps({**result, "seed": seed}))
    (directory / "metadata.json").write_text(json.dumps({"expert_comparable": comparable,
        "paired_expert": {"seed": seed, "success": True, "cost": 25.}}))
    np.savez_compressed(directory / "trajectory.npz", **arrays)


def save_protocol(root, seeds, methods=("bd_chi2",)):
    (root / "protocols").mkdir(exist_ok=True)
    (root / "protocols/cartpole.json").write_text(json.dumps({
        "methods": [{"id": method} for method in methods],
        "test_starts": [[seed, None, 50] for seed in seeds]}))


def test_reacher_completion_with_height_violation_is_not_success():
    targets = np.array([[.02, 0.], [.04, 0.], [.06, .11], [.08, 0.], [.10, 0.]])
    observations = np.zeros((6, 8))
    observations[1:, 4:6] = targets
    arrays = {"observations": observations, "actions": np.zeros((5, 2)),
              "references": np.zeros((5, 8)), "targets": targets,
              "target_completion_counts": np.arange(6)}
    result = {"steps": 5, "cost": float(np.sum(observations[:-1, 4:6] ** 2) * 40000),
              "success": True, "violations_strict": 1, "clean_success_strict": False,
              "terminal_reason": "targets_completed", "evaluation_steps": 500}
    outcome = episode_metrics("reacher", result, arrays)
    assert outcome["completion_success"]
    assert not outcome["success"]
    assert outcome["height_violations"] == 1
    assert outcome["max_height_excess_m"] == pytest.approx(.01)


def test_report_uses_current_protocol_and_only_successful_paired_costs(tmp_path):
    save_episode(tmp_path, 17, *cartpole_record())
    save_episode(tmp_path, 23, *cartpole_record(steps=10, success=False))
    save_protocol(tmp_path, [17, 23])
    report = aggregate_results(tmp_path, require_complete=True)
    row = report["rows"][0]
    assert report["complete"] and report["controller_records"] == 2
    assert row["successful_seeds"] == [17]
    assert row["cost_ratio"] == {"n": 1, "mean": 2., "sample_sd": None}
    assert "paper" not in (tmp_path / "README.md").read_text().lower()
    assert "BD-KL produces no actions" not in (tmp_path / "README.md").read_text()


def test_incomplete_configured_cohort_is_detected(tmp_path):
    save_episode(tmp_path, 17, *cartpole_record())
    save_protocol(tmp_path, [17, 23])
    report = aggregate_results(tmp_path)
    assert not report["complete"]
    with pytest.raises(ValueError, match="Incomplete benchmark"):
        aggregate_results(tmp_path, require_complete=True)


def test_changed_evaluation_conditions_exclude_expert_ratio(tmp_path):
    save_episode(tmp_path, 17, *cartpole_record(), comparable=False)
    save_protocol(tmp_path, [17])
    report = aggregate_results(tmp_path, require_complete=True)
    assert report["rows"][0]["successful_trials"] == 1
    assert report["rows"][0]["cost_ratio"]["n"] == 0


def test_cost_change_is_detected():
    result, arrays = cartpole_record()
    arrays["actions"][0, 0] += 100
    with pytest.raises(ValueError, match="cost mismatch"):
        episode_metrics("cartpole", result, arrays)


def test_zero_action_failure_has_no_successful_cost():
    result, arrays = cartpole_record(steps=0, success=False)
    result.update(cost=None, controller_failure=True)
    outcome = episode_metrics("cartpole", result, arrays)
    assert not outcome["success"]
    assert outcome["cost"] is None
    assert outcome["recomputed_trace_cost"] == 0


def test_cartpole_short_run_still_needs_fifty_stable_steps():
    result, arrays = cartpole_record(steps=10, success=False)
    result["evaluation_steps"] = 10
    assert not episode_metrics("cartpole", result, arrays)["success"]


def test_configured_weights_are_used_when_verifying_cost():
    result, arrays = cartpole_record()
    result["cost"] *= 2
    assert episode_metrics("cartpole", result, arrays, configuration={"action_weights": [2]})["cost"] == 100
    with pytest.raises(ValueError, match="cost mismatch"):
        episode_metrics("cartpole", result, arrays)


def test_sample_standard_deviation_and_singleton_are_explicit():
    assert statistics_summary([1, 3]) == {"n": 2, "mean": 2, "sample_sd": pytest.approx(2 ** .5)}
    assert statistics_summary([2])["sample_sd"] is None
    assert statistics_summary([])["mean"] is None


def test_timing_rejects_unvalidated_replays_and_accepts_any_method(tmp_path):
    directory = tmp_path / "timing/cartpole/bd_kl/seed_17"
    directory.mkdir(parents=True)
    trial = {"task": "cartpole", "method": "bd_kl", "seed": 17, "validated_replay": False,
             "complete_sequence": True, "decisions": 50, "cpu_ms_per_action": 2.}
    path = directory / "result.json"
    path.write_text(json.dumps(trial))
    with pytest.raises(ValueError, match="Unvalidated"):
        timing_rows(tmp_path, Inputs(tmp_path))
    trial["validated_replay"] = True
    path.write_text(json.dumps(trial))
    row, = timing_rows(tmp_path, Inputs(tmp_path))
    assert row["cpu_ms_per_action"] == {"n": 1, "mean": 2., "sample_sd": None}
    assert row["seeds"] == [17]


def test_push_t_initial_overlap_does_not_count_as_post_action_score():
    arrays = {"actions": np.empty((0, 2)), "observations": np.zeros((1, 6)), "coverage": np.array([.99])}
    result = {"steps": 0, "score": 0., "success": False, "controller_error": "infeasible"}
    outcome = episode_metrics("pusht", result, arrays)
    assert outcome["score"] == 0
    assert not outcome["success"]


def test_push_t_does_not_continue_after_crossing_success_threshold():
    arrays = {"actions": np.zeros((2, 2)), "observations": np.zeros((3, 6)), "coverage": np.array([0., .96, .97])}
    result = {"steps": 2, "score": 1., "success": True}
    with pytest.raises(ValueError, match="continued after"):
        episode_metrics("pusht", result, arrays)
