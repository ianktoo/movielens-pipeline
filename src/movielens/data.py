"""Loading MovieLens data — real or synthetic.

Two ways to get a ``Dataset``:

* :func:`load_raw`        -- read the real csv files you downloaded.
* :func:`make_synthetic`  -- generate a realistic fake dataset of any size,
                             used as a fallback when the real files are missing
                             and inside the test-suite (fast + deterministic).

Both return the same lightweight :class:`Dataset` container, so the rest of the
pipeline doesn't care where the data came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Column schemas — declared once so loaders and tests agree on them.
RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]
MOVIE_COLUMNS = ["movieId", "title", "genres"]


@dataclass
class Dataset:
    """A bundle of ratings + movies that travels through the pipeline.

    Attributes
    ----------
    ratings:
        One row per (user, movie) rating with columns
        ``userId, movieId, rating, timestamp``.
    movies:
        One row per movie with columns ``movieId, title, genres``.
    name:
        Human-readable label (e.g. ``"raw"``, ``"sample:active_users"``) used in
        plots and logging so we always know which slice we're looking at.
    """

    ratings: pd.DataFrame
    movies: pd.DataFrame
    name: str = "dataset"

    # -- convenience numbers, handy for printing summaries in the notebook ----
    @property
    def n_ratings(self) -> int:
        return len(self.ratings)

    @property
    def n_users(self) -> int:
        return int(self.ratings["userId"].nunique())

    @property
    def n_movies(self) -> int:
        return int(self.ratings["movieId"].nunique())

    @property
    def sparsity(self) -> float:
        """Fraction of the user x movie matrix that is *empty*.

        Real recommender data is ~99.9% empty; this single number is the best
        one-line description of "how hard is this problem".
        """
        denom = self.n_users * self.n_movies
        if denom == 0:
            return 0.0
        return 1.0 - (self.n_ratings / denom)

    def summary(self) -> dict[str, float | int | str]:
        """A small dict of headline stats, ready to display as a table."""
        return {
            "name": self.name,
            "n_ratings": self.n_ratings,
            "n_users": self.n_users,
            "n_movies": self.n_movies,
            "sparsity": round(self.sparsity, 6),
            "avg_rating": round(float(self.ratings["rating"].mean()), 3),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset(name={self.name!r}, n_ratings={self.n_ratings:,}, "
            f"n_users={self.n_users:,}, n_movies={self.n_movies:,})"
        )


def raw_available(raw_dir: Path | None = None) -> bool:
    """True if the real MovieLens csv files are present on disk."""
    raw_dir = raw_dir or config.RAW_MOVIELENS_DIR
    return (raw_dir / "ratings.csv").exists() and (raw_dir / "movies.csv").exists()


def load_raw(
    raw_dir: Path | None = None,
    nrows: int | None = None,
) -> Dataset:
    """Read the real MovieLens csv files into a :class:`Dataset`.

    Parameters
    ----------
    raw_dir:
        Folder containing ``ratings.csv`` and ``movies.csv``. Defaults to the
        unzipped ``data/raw/ml-32m`` directory.
    nrows:
        Optionally read only the first ``nrows`` rows of ratings. Useful for a
        quick peek without loading all 32 million rows into memory.
    """
    raw_dir = raw_dir or config.RAW_MOVIELENS_DIR
    if not raw_available(raw_dir):
        raise FileNotFoundError(
            f"MovieLens csv files not found in {raw_dir}. "
            "Download + unzip ml-32m.zip, or use make_synthetic() instead."
        )

    # Explicit dtypes keep memory down and parsing fast on the big file.
    ratings = pd.read_csv(
        raw_dir / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32",
               "timestamp": "int64"},
        nrows=nrows,
    )
    movies = pd.read_csv(
        raw_dir / "movies.csv",
        dtype={"movieId": "int32", "title": "string", "genres": "string"},
    )
    return Dataset(ratings=ratings, movies=movies, name="raw")


# ---------------------------------------------------------------------------
# Synthetic data — the fallback / test fixture
# ---------------------------------------------------------------------------
# Genre vocabulary mirrors the real MovieLens genres so downstream feature code
# (which splits the pipe-delimited "genres" column) behaves identically.
_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def make_synthetic(
    n_users: int = 500,
    n_movies: int = 200,
    n_ratings: int = 20_000,
    seed: int = config.DEFAULT_SEED,
) -> Dataset:
    """Generate a realistic, fully-deterministic synthetic dataset.

    The generation isn't purely random — it bakes in the structure that makes a
    recommender *work*, so models trained on it behave like models trained on
    real data:

    * each user has a latent "taste" vector and a personal bias (some users
      rate everything highly, some are harsh critics),
    * each movie has a latent "appeal" vector and a quality bias,
    * a rating ~= global mean + user bias + movie bias + taste·appeal + noise,
    * popularity is long-tailed: a few movies get most of the ratings.

    Returns the same :class:`Dataset` shape as :func:`load_raw`.
    """
    rng = np.random.default_rng(seed)
    n_factors = 4

    # Latent vectors and biases.
    user_factors = rng.normal(0, 1, size=(n_users, n_factors))
    movie_factors = rng.normal(0, 1, size=(n_movies, n_factors))
    user_bias = rng.normal(0, 0.5, size=n_users)
    movie_bias = rng.normal(0, 0.5, size=n_movies)
    global_mean = 3.5

    # Long-tailed popularity: movie i chosen with prob ~ 1/(rank+1).
    pop = 1.0 / (np.arange(1, n_movies + 1))
    pop = pop / pop.sum()
    rng.shuffle(pop)  # so it isn't always the low-id movies that are popular

    user_ids = rng.integers(0, n_users, size=n_ratings)
    movie_ids = rng.choice(n_movies, size=n_ratings, p=pop)

    raw = (
        global_mean
        + user_bias[user_ids]
        + movie_bias[movie_ids]
        + np.sum(user_factors[user_ids] * movie_factors[movie_ids], axis=1) * 0.5
        + rng.normal(0, 0.4, size=n_ratings)
    )
    # Snap to the MovieLens half-star grid and clip to the legal range.
    rating = np.clip(np.round(raw * 2) / 2, config.RATING_MIN, config.RATING_MAX)

    # Spread timestamps over ~3 years of unix time for temporal-split demos.
    base_ts = 1_500_000_000
    timestamp = base_ts + rng.integers(0, 3 * 365 * 24 * 3600, size=n_ratings)

    ratings = pd.DataFrame(
        {
            "userId": (user_ids + 1).astype("int32"),       # 1-based, like real data
            "movieId": (movie_ids + 1).astype("int32"),
            "rating": rating.astype("float32"),
            "timestamp": timestamp.astype("int64"),
        }
    )
    # Collapse accidental duplicate (user, movie) pairs, keeping the last rating.
    ratings = (
        ratings.sort_values("timestamp")
        .drop_duplicates(["userId", "movieId"], keep="last")
        .reset_index(drop=True)
    )

    # Build a movies table with 1-3 random genres each.
    titles, genres = [], []
    for m in range(n_movies):
        k = rng.integers(1, 4)
        g = rng.choice(_GENRES, size=k, replace=False)
        titles.append(f"Synthetic Movie {m + 1} ({2000 + (m % 20)})")
        genres.append("|".join(g))
    movies = pd.DataFrame(
        {
            "movieId": np.arange(1, n_movies + 1, dtype="int32"),
            "title": pd.array(titles, dtype="string"),
            "genres": pd.array(genres, dtype="string"),
        }
    )

    return Dataset(ratings=ratings, movies=movies, name="synthetic")


def load_or_synthesize(prefer_raw: bool = True, **synthetic_kwargs) -> Dataset:
    """Best-effort loader: real data if available, otherwise synthetic.

    This is what the notebook calls so the demo never crashes — if the 32M
    files are missing we transparently fall back to a synthetic dataset.
    """
    if prefer_raw and raw_available():
        return load_raw()
    return make_synthetic(**synthetic_kwargs)
