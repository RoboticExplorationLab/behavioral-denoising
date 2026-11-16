"""Validate demonstration inputs and safe dataset extraction."""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from scripts.prepare_data import extract_archive, verify_inputs


ROOT = Path(__file__).resolve().parents[1]


def test_all_included_inputs_match_checksums():
    assert verify_inputs()["verified_files"] > 0


def test_configured_starts_have_unique_seeds_and_finite_observations():
    for task in ("cartpole", "reacher", "rocket", "pusht"):
        setup = json.loads((ROOT / "data" / task / "test_starts.json").read_text())
        starts = setup["starts"]
        assert starts and len({start["seed"] for start in starts}) == len(starts)
        for start in starts:
            assert np.isfinite(start["initial_observation"]).all()
            if task == "reacher":
                assert np.asarray(start["targets"]).shape == (5, 2)
                assert np.isfinite(start["targets"]).all()


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
