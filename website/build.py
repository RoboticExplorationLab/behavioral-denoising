"""Assemble the current gallery and the frozen L4DC submission gallery."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def assemble(archive_path, output, specification, *, frozen_site=False):
    """Verify before writing; refuse to replace an existing build directory."""
    archive_path, output = Path(archive_path), Path(output)
    if (archive_path.stat().st_size != specification["bytes"]
            or digest(archive_path) != specification["sha256"]):
        raise ValueError("Gallery archive differs from the pinned release asset")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    with ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive member")
        for member in members:
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                    or stat.S_ISLNK(member.external_attr >> 16) or member.is_dir()):
                raise ValueError(f"Unsafe gallery member: {member.filename}")
        saved = json.loads(archive.read("data/trials.json"))
        trials = saved["trials"]
        if not trials or saved["total"] != len(trials):
            raise ValueError("Invalid gallery trial count")
        seeds, expected = set(), {"data/trials.json"}
        if frozen_site:
            expected.update(("index.html", "styles.css", "gallery.js", ".nojekyll"))
        for trial in trials:
            seed = trial["seed"]
            if type(seed) is not int or seed < 0 or seed in seeds:
                raise ValueError("Invalid or repeated gallery seed")
            seeds.add(seed)
            for key, extension in (("video", "mp4"), ("poster", "jpg")):
                relative = f"media/pusht/seed_{seed}.{extension}"
                if trial[key] != relative:
                    raise ValueError("Unexpected gallery asset path")
                expected.add(relative)
            if any(not isinstance(trial[key], (int, float)) or not 0 <= trial[key] <= 1
                   for key in ("peak_coverage", "final_coverage")):
                raise ValueError("Invalid coverage value")
        if set(names) != expected:
            raise ValueError("Missing or unexpected gallery assets")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".gallery-build-", dir=output.parent) as temporary:
            staging = Path(temporary) / "site"
            staging.mkdir()
            for member in members:
                destination = staging / member.filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
            if not frozen_site:
                for name in ("index.html", "styles.css", "gallery.js"):
                    shutil.copyfile(HERE / name, staging / name)
                (staging / ".nojekyll").touch()
            staging.rename(output)
    return {"trials": len(trials), "output": str(output)}


def assemble_site(current_archive, submission_archive, output, current_specification, submission_specification):
    """Publish both galleries together, preserving the archived HTML and assets."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gallery-site-", dir=output.parent) as temporary:
        staging = Path(temporary) / "site"
        current = assemble(current_archive, staging, current_specification)
        submission = assemble(submission_archive, staging / "l4dc", submission_specification, frozen_site=True)
        staging.rename(output)
    return {"current_trials": current["trials"], "l4dc_trials": submission["trials"], "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local copy of the pinned release asset")
    parser.add_argument("--l4dc-archive", type=Path, help="Use a local copy of the frozen submission gallery")
    parser.add_argument("--output", type=Path, default=HERE / "build")
    args = parser.parse_args()
    specifications = [json.loads((HERE / name).read_text()) for name in
                      ("gallery_asset.json", "l4dc_gallery_asset.json")]
    with tempfile.TemporaryDirectory(prefix="pusht-gallery-") as temporary:
        archives = []
        for index, (local, specification) in enumerate(zip((args.archive, args.l4dc_archive), specifications)):
            archive = local or Path(temporary) / f"gallery-{index}.zip"
            if local is None:
                with urlopen(specification["url"], timeout=60) as source, archive.open("xb") as target:
                    shutil.copyfileobj(source, target)
            archives.append(archive)
        report = assemble_site(*archives, args.output, *specifications)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
