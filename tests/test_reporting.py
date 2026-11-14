"""Paper metrics must distinguish completion, constraint success and timing scope."""

import json
import shutil

import numpy as np
import pytest

from src.benchmarks.reporting import aggregate_results, episode_metrics, statistics_summary, timing_rows, Inputs


@pytest.mark.reference_results
def test_reacher_completion_with_height_violation_is_not_paper_success(paper_results):
    directory = paper_results / "reacher/bd_chi2/seed_3000"
    result = json.loads((directory / "result.json").read_text())
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    outcome = episode_metrics("reacher", result, arrays)
    assert outcome["completion_success"]
    assert not outcome["success"]
    assert outcome["height_violations"] == 4


@pytest.mark.reference_results
def test_paper_table_reproduces_strict_subsets_and_all_push_t_scores(tmp_path, paper_results):
    # Symlinks keep the test small; only newly generated reports are written here.
    for task in ("cartpole", "rocket", "reacher", "pusht", "timing"):
        (tmp_path / task).symlink_to(paper_results / task, target_is_directory=True)
    report = aggregate_results(tmp_path, require_complete=True)
    rows = {(r["task"], r["method"]): r for r in report["rows"]}
    assert report["controller_records"] == 340
    assert report["complete"]
    assert rows["reacher", "bd_chi2"]["successful_trials"] == 7
    assert rows["reacher", "bd_chi2"]["completion_only_successes"] == 10
    assert rows["reacher", "bd_chi2"]["cost_ratio"]["n"] == 7
    assert rows["reacher", "bd_chi2"]["cost_ratio"]["mean"] == pytest.approx(1.104, abs=.0005)
    assert rows["reacher", "bd_mean_chi2"]["successful_trials"] == 4
    assert rows["reacher", "bd_mean_chi2"]["cost_ratio"]["mean"] == pytest.approx(1.047, abs=.0005)
    assert rows["cartpole", "select_dpc"]["cost_ratio"]["n"] == 4
    assert rows["pusht", "bd_chi2"]["successful_trials"] == 19
    assert rows["pusht", "bd_chi2"]["score"]["n"] == 50
    assert rows["pusht", "bd_chi2"]["score"]["mean"] == pytest.approx(.6606, abs=.00005)
    assert rows["pusht", "select_dpc"]["successful_trials"] == 0
    assert rows["pusht", "select_dpc"]["score"]["n"] == 50
    assert len(report["timing"]) == 24


@pytest.mark.reference_results
def test_cost_change_is_detected(paper_results):
    directory = paper_results / "cartpole/bd_chi2/seed_10003"
    result = json.loads((directory / "result.json").read_text())
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["actions"][0, 0] += 100
    with pytest.raises(ValueError, match="cost mismatch"):
        episode_metrics("cartpole", result, arrays)


@pytest.mark.reference_results
def test_zero_action_failure_has_no_successful_cost(paper_results):
    directory = paper_results / "cartpole/bd_kl/seed_10003"
    result = json.loads((directory / "result.json").read_text())
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    outcome = episode_metrics("cartpole", result, arrays)
    assert not outcome["success"]
    assert outcome["cost"] is None
    assert outcome["recomputed_trace_cost"] == 0


def test_cartpole_short_diagnostic_still_needs_fifty_stable_steps():
    observations = np.tile([0., 0., 1., 0., 0.], (11, 1))
    arrays = {"observations": observations, "states": np.zeros((11, 4)),
              "actions": np.zeros((10, 1)), "references": observations[:-1]}
    result = {"steps": 10, "evaluation_steps": 10, "cost": 0., "success": False}
    assert not episode_metrics("cartpole", result, arrays)["success"]


@pytest.mark.reference_results
def test_changed_horizon_is_diagnostic_and_has_no_ratio(tmp_path, paper_results):
    source = paper_results / "cartpole/bd_chi2/seed_10003"
    destination = tmp_path / "cartpole/bd_chi2/seed_10003"
    shutil.copytree(source, destination)
    result_path = destination / "result.json"
    result = json.loads(result_path.read_text())
    result["paper_protocol"] = False
    result_path.write_text(json.dumps(result))
    report = aggregate_results(tmp_path, tasks=["cartpole"], expert_root=paper_results)
    assert not report["complete"]
    assert report["rows"][0]["cost_ratio"]["n"] == 0
    with pytest.raises(ValueError, match="Incomplete paper cohort"):
        aggregate_results(tmp_path, tasks=["cartpole"], expert_root=paper_results, require_complete=True)


def test_sample_standard_deviation_and_singleton_are_explicit():
    assert statistics_summary([1, 3]) == {"n": 2, "mean": 2, "sample_sd": pytest.approx(2 ** .5)}
    assert statistics_summary([2])["sample_sd"] is None
    assert statistics_summary([])["mean"] is None


@pytest.mark.reference_results
def test_timing_rejects_an_ineligible_cohort(tmp_path, paper_results):
    folder = tmp_path / "timing"
    folder.mkdir()
    cohort = json.loads((paper_results / "timing/required.json").read_text())
    cohort["trials"][0]["eligible"] = False
    (folder / "required.json").write_text(json.dumps(cohort))
    with pytest.raises(ValueError, match="Ineligible timing trial"):
        timing_rows(tmp_path, Inputs(tmp_path))


def test_push_t_initial_overlap_does_not_count_as_post_action_score():
    arrays = {"actions": np.empty((0, 2)), "observations": np.zeros((1, 6)), "coverage": np.array([.99])}
    result = {"steps": 0, "score": 0., "success": False, "controller_error": "infeasible"}
    outcome = episode_metrics("pusht", result, arrays)
    assert outcome["score"] == 0
    assert not outcome["success"]
