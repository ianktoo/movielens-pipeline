"""Project paths and shared constants.

Everything path-related lives here so the rest of the library never has to
guess where data lives. Paths are derived relative to the project root, so the
library works the same whether it's imported from a notebook, a test, or a
script.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
# config.py lives at: <root>/src/movielens/config.py  ->  parents[2] == <root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"          # the unzipped MovieLens csv files
SAMPLES_DIR: Path = DATA_DIR / "samples"  # our generated, swappable sample sets

# The MovieLens 32M archive unzips into a folder called "ml-32m".
RAW_MOVIELENS_DIR: Path = RAW_DIR / "ml-32m"

DATASET_URL: str = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"

# ---------------------------------------------------------------------------
# Rating scale (MovieLens uses half-stars from 0.5 to 5.0)
# ---------------------------------------------------------------------------
RATING_MIN: float = 0.5
RATING_MAX: float = 5.0

# A fixed seed keeps every demo reproducible — the class sees the same numbers.
DEFAULT_SEED: int = 42


def ensure_dirs() -> None:
    """Create the data directories if they don't already exist."""
    for d in (DATA_DIR, RAW_DIR, SAMPLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
