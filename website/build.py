"""Assemble the gallery from a checksummed media archive and the website sources."""
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


def assemble(archive_path, output, specification):
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
            for name in ("index.html", "styles.css", "gallery.js"):
                shutil.copyfile(HERE / name, staging / name)
            (staging / ".nojekyll").touch()
            staging.rename(output)
    return {"trials": len(trials), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local copy of the pinned release asset")
    parser.add_argument("--output", type=Path, default=HERE / "build")
    args = parser.parse_args()
    specification = json.loads((HERE / "gallery_asset.json").read_text())
    if args.archive:
        report = assemble(args.archive, args.output, specification)
    else:
        with tempfile.TemporaryDirectory(prefix="pusht-gallery-") as temporary:
            archive = Path(temporary) / "gallery.zip"
            with urlopen(specification["url"], timeout=60) as source, archive.open("xb") as target:
                shutil.copyfileobj(source, target)
            report = assemble(archive, args.output, specification)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
