"""Verify the included datasets or import the official Push-T dataset.

Without arguments this only reads and checks the included inputs. A Push-T
download/import requires a new output directory; it never replaces paper data.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_inputs(root=ROOT):
    manifest = json.loads((root / "data/manifest.json").read_text())
    for name, entry in manifest.items():
        path = root / name
        if not path.is_file() or digest(path) != entry["sha256"]:
            raise ValueError(f"Paper input differs from its manifest: {name}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Paper input size differs from its manifest: {name}")
    return {"verified_files": len(manifest), "manifest_sha256": digest(root / "data/manifest.json")}


def extract_archive(archive, destination, members=None):
    """Extract regular ZIP members only, retaining the upstream directory layout."""
    with ZipFile(archive) as saved:
        for member in saved.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"ZIP symlinks are not supported: {member.filename}")
            if members is not None and not member.is_dir() and member.filename not in members:
                continue
            path = destination.joinpath(*relative.parts)
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with saved.open(member) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target)


def prepare_pusht(output, archive=None):
    """Download the checksum-pinned source and convert it in a fresh directory."""
    from src.tasks.pusht import import_demonstrations

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    provenance = json.loads((ROOT / "data/pusht/upstream_manifest.json").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pusht-import-", dir=output.parent) as temporary:
        staging = Path(temporary)
        raw_archive = staging / "pusht.zip"
        if archive is None:
            with urlopen(provenance["source_url"], timeout=60) as source, raw_archive.open("xb") as target:
                shutil.copyfileobj(source, target)
        else:
            shutil.copyfile(Path(archive), raw_archive)
        expected = provenance["source_archive"]
        if raw_archive.stat().st_size != expected["bytes"] or digest(raw_archive) != expected["sha256"]:
            raise ValueError("Push-T archive differs from the paper's pinned upstream download")
        # State-based control uses the state and action arrays only.
        prefix = "pusht/pusht_cchi_v7_replay.zarr/"
        members = {prefix + name for name in provenance["source_files_sha256"]}
        extract_archive(raw_archive, staging / "raw", members=members)
        sources = list((staging / "raw").rglob("pusht_cchi_v7_replay.zarr"))
        if len(sources) != 1:
            raise ValueError("Expected exactly one released Push-T Zarr dataset")
        actual_hashes = {str(path.relative_to(sources[0])): digest(path)
                         for path in sorted(sources[0].rglob("*")) if path.is_file()}
        if actual_hashes != provenance["source_files_sha256"]:
            raise ValueError("Extracted Push-T source differs from the archived source manifest")
        manifest = import_demonstrations(sources[0], staging / "demonstrations.npz",
                                         archive_path=raw_archive, manifest_path=staging / "manifest.json")
        if manifest["dataset_sha256"] != provenance["dataset_sha256"]:
            raise ValueError("Converted Push-T bank differs; use the pinned Push-T environment")
        if output.exists():
            raise FileExistsError(f"Output appeared during import: {output}")
        Path(temporary).rename(output)
    return {"output": str(output), "episodes": manifest["episodes"],
            "recorded_pairs": manifest["recorded_pairs"], "aligned_pairs": manifest["aligned_pairs"],
            "dataset_sha256": manifest["dataset_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify the included inputs (also the default)")
    parser.add_argument("--pusht-output", type=Path, help="New directory for the official archive and converted bank")
    parser.add_argument("--archive", type=Path, help="Use a local official pusht.zip instead of downloading")
    arguments = parser.parse_args()
    if arguments.archive is not None and arguments.pusht_output is None:
        parser.error("--archive requires --pusht-output")
    report = verify_inputs()
    if arguments.pusht_output is not None:
        report["pusht_import"] = prepare_pusht(arguments.pusht_output, arguments.archive)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
