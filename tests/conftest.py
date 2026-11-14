"""Optional reference records shared by paper-result checks."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def paper_results():
    path = Path(__file__).resolve().parents[1] / "results/paper"
    if not path.exists():
        pytest.skip(
            "Paper reference results are not extracted; run from the repository root: "
            "python -m zipfile -e results/paper.zip ."
        )
    return path
