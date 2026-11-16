"""Saved-run rendering accepts new subsets without archived benchmark outcomes."""
import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

pytest.importorskip("shapely", reason="Gallery geometry uses the Push-T environment")

from scripts.plots import pusht
from scripts.plots import pusht_gallery

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gallery_build", ROOT / "website/build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_record(root, seed=42):
    method = {"id": "bd_chi2", "parameters": {}}
    protocol = {"sha256": "fixture", "methods": [method]}
    write_json(root / "protocols/pusht.json", protocol)
    directory = root / "pusht/bd_chi2" / f"seed_{seed}"
    directory.mkdir(parents=True)
    geometry = {"block_polygons": [[[-60, 0], [60, 0], [60, 30], [-60, 30]],
                                    [[-15, 30], [15, 30], [15, 120], [-15, 120]]],
                "block_center_of_gravity": [0, 45], "goal_pose": [256, 256, 0],
                "control_hz": 10, "success_threshold": .95,
                "workspace": [0, 512, 0, 512], "agent_radius": 15}
    state = [25, 25, 80, 80, 0]
    np.savez(directory / "trajectory.npz", states=[state, state], time=[0, .1],
             coverage=[0., 0.], goal_pose=geometry["goal_pose"], actions=[[25, 25]],
             observations=[[25, 25, 80, 80, 1, 0]] * 2, rewards=[0.])
    result = {"method": "bd_chi2", "seed": seed, "steps": 1, "success": False,
              "max_coverage": 0., "final_coverage": 0., "termination_reason": "timeout"}
    write_json(directory / "result.json", result)
    metadata = {"method": method, "protocol_sha256": "fixture", "geometry": geometry,
                "payload_sha256": {name: pusht.digest(directory / name)
                                   for name in ("trajectory.npz", "result.json")}}
    write_json(directory / "metadata.json", metadata)
    return directory


def test_gallery_accepts_one_arbitrary_seed_and_short_horizon(tmp_path):
    make_record(tmp_path)
    trials = pusht_gallery.load_trials(tmp_path)
    assert [trial.seed for trial in trials] == [42]
    assert trials[0].time[-1] == .1
    pusht.plot_coverage({"bd_chi2": trials}, tmp_path / "figures")
    assert (tmp_path / "figures/coverage.csv").is_file()


def test_changed_result_is_rejected_before_plotting(tmp_path):
    directory = make_record(tmp_path)
    (directory / "result.json").write_text('{}')
    with pytest.raises(ValueError, match="hash mismatch"):
        pusht_gallery.load_trials(tmp_path)


def make_asset(path, extra=None):
    data = {"total": 1, "trials": [{"seed": 42, "peak_coverage": .94,
             "final_coverage": .8, "video": "media/pusht/seed_42.mp4",
             "poster": "media/pusht/seed_42.jpg"}]}
    with ZipFile(path, "w") as archive:
        archive.writestr("data/trials.json", json.dumps(data))
        archive.writestr("media/pusht/seed_42.mp4", b"video")
        archive.writestr("media/pusht/seed_42.jpg", b"poster")
        if extra:
            archive.writestr(extra, b"unexpected")
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_gallery_asset_is_verified_and_assembled(tmp_path):
    archive, output = tmp_path / "gallery.zip", tmp_path / "build"
    specification = make_asset(archive)
    result = builder.assemble(archive, output, specification)
    assert result["trials"] == 1
    for name in ("index.html", "gallery.js", "styles.css", ".nojekyll", "data/trials.json"):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        builder.assemble(archive, output, specification)


def test_asset_wrong_checksum_and_unsafe_members_fail_without_output(tmp_path):
    archive, output = tmp_path / "gallery.zip", tmp_path / "build"
    specification = make_asset(archive)
    with pytest.raises(ValueError, match="pinned release"):
        builder.assemble(archive, output, {**specification, "sha256": "wrong"})
    assert not output.exists()
    specification = make_asset(archive, "../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        builder.assemble(archive, output, specification)
    assert not output.exists()
