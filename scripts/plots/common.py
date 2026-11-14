"""Portable paths and font selection for the offline paper figures."""
import hashlib
import os
from pathlib import Path
import warnings

from matplotlib import font_manager


REPOSITORY = Path(__file__).resolve().parents[2]
PAPER_RESULTS = REPOSITORY / "results/paper"


def verify_file_hash(path, *expected_hashes):
    """Require a recorded checksum and check every supplied metadata version."""
    expected = [value for value in expected_hashes if value is not None]
    if not expected:
        raise ValueError(f"Missing recorded checksum: {path}")
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if any(value != actual for value in expected):
        raise ValueError(f"Recorded checksum mismatch: {path}")


def verify_rendering_hash(directory, metadata, *, required=False):
    """Check new payload hashes or the archived Rocket geometry checksum."""
    recording = metadata.get("recording", {})
    expected = (recording.get("rendering_sha256"),
                recording.get("assets", {}).get("rocket_geometry.json"),
                metadata.get("payload_sha256", {}).get("rendering.json"))
    if required or any(value is not None for value in expected):
        verify_file_hash(Path(directory) / "rendering.json", *expected)


def paper_font():
    """Use Times New Roman when installed, otherwise a serif fallback."""
    for path in sorted((Path.home() / ".local/share/fonts/msttcorefonts").glob("*.TTF")):
        font_manager.fontManager.addfont(str(path))
    requested = os.environ.get("PAPER_FONT", "Times New Roman")
    try:
        font_manager.findfont(requested, fallback_to_default=False)
        return requested
    except ValueError:
        warnings.warn(f"{requested} is unavailable; using DejaVu Serif. "
                      "Install the paper font or set PAPER_FONT to choose another font.", stacklevel=2)
        return "DejaVu Serif"


def replay_input_path(original, record_directory):
    """Resolve a replay input relative to the dataset or simulator directory."""
    original = Path(original)
    if "third_party" in original.parts:
        return REPOSITORY.joinpath(*original.parts[original.parts.index("third_party"):])
    return Path(record_directory) / original.name
