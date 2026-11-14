"""Keep the paper inputs immutable and bind task starts to archived episodes."""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from scripts.prepare_data import extract_archive, verify_inputs


ROOT = Path(__file__).resolve().parents[1]


def test_all_included_inputs_match_checksums():
    assert verify_inputs()["verified_files"] >= 24


@pytest.mark.reference_results
def test_frozen_starts_match_saved_observations_and_references(paper_results):
    for task in ("cartpole", "reacher", "rocket", "pusht"):
        setup = json.loads((ROOT / "data" / task / "test_starts.json").read_text())
        assert len(setup["starts"]) == (50 if task == "pusht" else 10)
        for start in setup["starts"]:
            path = paper_results / task / "bd_chi2" / f'seed_{start["seed"]}' / "trajectory.npz"
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(start["initial_observation"], saved["observations"][0])
                if task == "reacher":
                    np.testing.assert_array_equal(start["targets"], saved["targets"])
                if task == "rocket":
                    assert start["initial_condition"] == ([.2, .9, 0.] if start["seed"] == 3000 else None)


def test_zip_paths_cannot_escape_destination(tmp_path):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as saved:
        saved.writestr("../outside", "not allowed")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        extract_archive(archive, tmp_path / "raw")
    assert not (tmp_path / "outside").exists()


def test_zip_extraction_can_keep_only_state_data(tmp_path):
    archive = tmp_path / "data.zip"
    with ZipFile(archive, "w") as saved:
        saved.writestr("data/state/0", b"state")
        saved.writestr("data/img/0", b"image")
    extract_archive(archive, tmp_path / "raw", members={"data/state/0"})
    assert (tmp_path / "raw/data/state/0").read_bytes() == b"state"
    assert not (tmp_path / "raw/data/img/0").exists()
