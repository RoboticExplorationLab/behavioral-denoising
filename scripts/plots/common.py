"""Portable paths and font selection for saved-run plots."""
import os
from pathlib import Path
from matplotlib import font_manager

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = REPOSITORY / "runs/benchmark"


def plot_font():
    """Use the requested font when installed, otherwise a portable serif."""
    for path in sorted((Path.home() / ".local/share/fonts/msttcorefonts").glob("*.TTF")):
        font_manager.fontManager.addfont(str(path))
    requested = os.environ.get("PLOT_FONT", "Times New Roman")
    try:
        font_manager.findfont(requested, fallback_to_default=False)
        return requested
    except ValueError:
        return "DejaVu Serif"
