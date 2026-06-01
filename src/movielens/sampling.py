"""Smart sampling: turn 32 million ratings into manageable, swappable slices.

The full dataset is too big to model live in a classroom, so we cut it down.
But *how* you cut matters: a naive random sample of rows shatters every user's
history into a few disconnected ratings, which is useless for a recommender.

This module offers several **named strategies** that each keep the data
coherent, and a small registry so the notebook can swap one sample set for
another with a single string:

    sample = sampling.build_sample(full, strategy="active_users", size=2000)
    sample = sampling.build_sample(full, strategy="dense", size=2000)

Generated samples can be saved to / loaded from ``data/samples`` so the class
demo starts instantly instead of re-sampling 32M rows every time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import config
from .data import Dataset

# A strategy takes (dataset, size, rng, **kwargs) and returns the userIds to keep.
StrategyFn = Callable[..., np.ndarray]

_STRATEGIES: dict[str, StrategyFn] = {}


def register(name: str) -> Callable[[StrategyFn], StrategyFn]:
    """Decorator to add a sampling strategy to the registry."""

    def deco(fn: StrategyFn) -> StrategyFn:
        _STRATEGIES[name] = fn
        return fn

    return deco


def available_strategies() -> list[str]:
    """Names of all registered strategies (so the notebook can list them)."""
    return sorted(_STRATEGIES)


# ---------------------------------------------------------------------------
# The strategies. Each one returns an array of userIds to keep.
# We always sample by *user* so each kept user brings their whole history with
# them — that's what keeps the rating matrix usable for recommendation.
# ---------------------------------------------------------------------------
def _user_activity(ds: Dataset) -> pd.Series:
    """Ratings-per-user, sorted descending (a user's 'activity')."""
    return ds.ratings.groupby("userId").size().sort_values(ascending=False)


@register("random_users")
def _random_users(ds: Dataset, size: int, rng: np.random.Generator, **_) -> np.ndarray:
    """Pick ``size`` users uniformly at random. The simple baseline."""
    users = ds.ratings["userId"].unique()
    size = min(size, len(users))
    return rng.choice(users, size=size, replace=False)


@register("active_users")
def _active_users(ds: Dataset, size: int, rng: np.random.Generator, **_) -> np.ndarray:
    """Pick the ``size`` most active users.

    These users have long rating histories, so models have lots of signal to
    learn from — the friendliest slice for demoing a recommender.
    """
    activity = _user_activity(ds)
    return activity.head(size).index.to_numpy()


@register("dense")
def _dense(
    ds: Dataset,
    size: int,
    rng: np.random.Generator,
    min_user_ratings: int = 20,
    top_movies: int = 500,
    **_,
) -> np.ndarray:
    """Build a *dense* slice: active users restricted to popular movies.

    Density is the enemy of cold-start. By keeping only the most-rated movies
    and reasonably active users, the user x movie matrix has far fewer holes,
    which makes collaborative filtering visibly work in a short demo.

    (Note: this strategy also signals, via :data:`DENSE_TOP_MOVIES`, that the
    caller should restrict movies too; :func:`build_sample` handles that.)
    """
    activity = _user_activity(ds)
    eligible = activity[activity >= min_user_ratings].index.to_numpy()
    # permutation returns a fresh, writable array (to_numpy() can be read-only).
    eligible = rng.permutation(eligible)
    return eligible[:size]


# Strategies that also restrict the movie set record their movie cap here.
DENSE_TOP_MOVIES = "dense"


def build_sample(
    ds: Dataset,
    strategy: str = "active_users",
    size: int = 2000,
    seed: int = config.DEFAULT_SEED,
    top_movies: int = 500,
    min_user_ratings: int = 20,
    name: str | None = None,
) -> Dataset:
    """Produce a smaller :class:`Dataset` using a named strategy.

    Parameters
    ----------
    ds:
        The full (or already-sampled) dataset to draw from.
    strategy:
        One of :func:`available_strategies`. Controls *which* users are kept.
    size:
        Target number of users to keep.
    top_movies:
        For the ``dense`` strategy, also restrict to this many most-rated
        movies (ignored by other strategies).
    min_user_ratings:
        Minimum ratings a user must have to be eligible (``dense`` only).
    name:
        Optional label for the resulting dataset; defaults to
        ``"sample:<strategy>"``.
    """
    if strategy not in _STRATEGIES:
        raise KeyError(
            f"Unknown strategy {strategy!r}. "
            f"Available: {available_strategies()}"
        )

    rng = np.random.default_rng(seed)
    keep_users = _STRATEGIES[strategy](
        ds, size=size, rng=rng,
        min_user_ratings=min_user_ratings, top_movies=top_movies,
    )

    sub = ds.ratings[ds.ratings["userId"].isin(keep_users)]

    # The dense strategy also trims to the most popular movies.
    if strategy == DENSE_TOP_MOVIES:
        popular = (
            sub.groupby("movieId").size().sort_values(ascending=False)
            .head(top_movies).index
        )
        sub = sub[sub["movieId"].isin(popular)]

    sub = sub.reset_index(drop=True)
    movies = ds.movies[ds.movies["movieId"].isin(sub["movieId"].unique())].reset_index(drop=True)

    return Dataset(
        ratings=sub,
        movies=movies,
        name=name or f"sample:{strategy}",
    )


# ---------------------------------------------------------------------------
# Persistence — save / load sample sets so demos start instantly
# ---------------------------------------------------------------------------
def save_sample(ds: Dataset, key: str, samples_dir: Path | None = None) -> Path:
    """Write a sample to ``data/samples/<key>/`` as two parquet files."""
    samples_dir = samples_dir or config.SAMPLES_DIR
    out = samples_dir / key
    out.mkdir(parents=True, exist_ok=True)
    ds.ratings.to_parquet(out / "ratings.parquet", index=False)
    ds.movies.to_parquet(out / "movies.parquet", index=False)
    return out


def load_sample(key: str, samples_dir: Path | None = None) -> Dataset:
    """Read back a sample previously written by :func:`save_sample`."""
    samples_dir = samples_dir or config.SAMPLES_DIR
    src = samples_dir / key
    ratings = pd.read_parquet(src / "ratings.parquet")
    movies = pd.read_parquet(src / "movies.parquet")
    return Dataset(ratings=ratings, movies=movies, name=f"sample:{key}")


def list_saved_samples(samples_dir: Path | None = None) -> list[str]:
    """Keys of all sample sets currently saved on disk."""
    samples_dir = samples_dir or config.SAMPLES_DIR
    if not samples_dir.exists():
        return []
    return sorted(
        p.name for p in samples_dir.iterdir()
        if p.is_dir() and (p / "ratings.parquet").exists()
    )
