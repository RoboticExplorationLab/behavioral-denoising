"""Full selected BD-chi2 episodes; opt in with pytest -m simulator.

These compare newly executed trajectories with the released paper records.
They do not tune parameters or run the expensive full DeePC benchmark.
"""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CASES = (("cartpole", 10003), ("rocket", 3000), ("reacher", 3001), ("pusht", 300009))


@pytest.mark.simulator
@pytest.mark.reference_results
@pytest.mark.parametrize("task,seed", CASES)
def test_full_bd_chi2_episode_matches_paper(task, seed, tmp_path, paper_results):
    expected_directory = paper_results / task / "bd_chi2" / f"seed_{seed}"
    expected = json.loads((expected_directory / "result.json").read_text())
    with np.load(expected_directory / "trajectory.npz", allow_pickle=False) as archive:
        paper = dict(archive)
    environment = ROOT / (".venv-pusht" if task == "pusht" else ".venv")
    python = environment / "bin/python"
    if not python.exists():
        pytest.skip(f"Create the documented {environment.name} environment first")
    output = tmp_path / "evaluation"
    command = [str(python), "-m", f"experiments.{task}", "--methods", "bd_chi2",
               "--seeds", str(seed), "--output", str(output)]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=180)
    actual_directory = output / task / "bd_chi2" / f"seed_{seed}"
    actual = json.loads((actual_directory / "result.json").read_text())
    assert actual["steps"] == expected["steps"]
    assert actual["success"] == expected["success"]
    with np.load(actual_directory / "trajectory.npz", allow_pickle=False) as new:
        keys = ["states", "observations", "actions"]
        keys += ["coverage"] if task == "pusht" else ["references"]
        if task == "reacher":
            keys += ["targets", "target_completion_counts"]
        for key in keys:
            np.testing.assert_allclose(new[key], paper[key], rtol=1e-11, atol=1e-11, err_msg=f"{task}: {key}")
    if task == "pusht":
        assert actual["score"] == pytest.approx(expected["score"], abs=1e-12)
    else:
        assert actual["cost"] == pytest.approx(expected["cost"], rel=1e-11, abs=1e-7)
    # A second invocation verifies hashes and skips, rather than rerunning, the episode.
    before = {name: (actual_directory / name).read_bytes() for name in ("result.json", "metadata.json", "trajectory.npz")}
    resumed = subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=60)
    assert "verified existing record" in resumed.stdout
    assert before == {name: (actual_directory / name).read_bytes() for name in before}
