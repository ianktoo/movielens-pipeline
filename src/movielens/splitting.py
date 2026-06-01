"""Splitting ratings into train / test sets.

Splitting a recommender dataset is subtler than ``train_test_split`` on a flat
table, because we must never test on a user we never trained on. Two honest
strategies are provided:

* :func:`random_split`   -- shuffle all ratings, hold out a fraction. Simple,
                            but can leak "future" ratings into the past.
* :func:`temporal_split` -- for each user, hold out their *most recent* ratings.
                            This mimics reality: predict the future from the
                            past. It's the fairer evaluation for a recommender.

Both guarantee every test user also appears in train.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class Split:
    """A train/test pair of rating tables (movies are shared, so not copied)."""

    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Split(n_train={self.n_train:,}, n_test={self.n_test:,})"


def random_split(
    ratings: pd.DataFrame,
    test_frac: float = 0.2,
    seed: int = config.DEFAULT_SEED,
) -> Split:
    """Randomly hold out ``test_frac`` of ratings, per user.

    We sample within each user so that every user keeps most of their history
    in train and contributes a few ratings to test — no user is left untrained.
    """
    rng = np.random.default_rng(seed)

    def _take_test(group: pd.DataFrame) -> pd.Index:
        n_test = max(1, int(round(len(group) * test_frac)))
        n_test = min(n_test, len(group) - 1) if len(group) > 1 else 0
        if n_test == 0:
            return pd.Index([])
        chosen = rng.choice(group.index.to_numpy(), size=n_test, replace=False)
        return pd.Index(chosen)

    test_idx = (
        ratings.groupby("userId", group_keys=False)
        .apply(_take_test, include_groups=False)
    )
    test_index = np.concatenate([idx.to_numpy() for idx in test_idx]) if len(test_idx) else np.array([], dtype=int)

    test_mask = ratings.index.isin(test_index)
    return Split(
        train=ratings[~test_mask].reset_index(drop=True),
        test=ratings[test_mask].reset_index(drop=True),
    )


def temporal_split(
    ratings: pd.DataFrame,
    test_frac: float = 0.2,
) -> Split:
    """Per-user, hold out the most *recent* ``test_frac`` of ratings.

    This is the realistic setup: train on what a user did before, test whether
    we'd have predicted what they did next. Requires a ``timestamp`` column.
    """
    if "timestamp" not in ratings.columns:
        raise ValueError("temporal_split requires a 'timestamp' column.")

    ordered = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)

    # Rank each rating within its user, newest = highest rank.
    grp = ordered.groupby("userId")
    counts = grp["userId"].transform("size")
    rank = grp.cumcount()  # 0 = oldest

    n_test = np.maximum(1, np.round(counts * test_frac).astype(int))
    # Keep at least one training rating per user.
    n_test = np.minimum(n_test, counts - 1).clip(lower=0)
    threshold = counts - n_test  # ranks >= threshold go to test

    is_test = rank >= threshold
    return Split(
        train=ordered[~is_test].reset_index(drop=True),
        test=ordered[is_test].reset_index(drop=True),
    )


def split(
    ratings: pd.DataFrame,
    strategy: str = "temporal",
    test_frac: float = 0.2,
    seed: int = config.DEFAULT_SEED,
) -> Split:
    """Dispatch helper so the notebook can switch strategy with a string."""
    if strategy == "random":
        return random_split(ratings, test_frac=test_frac, seed=seed)
    if strategy == "temporal":
        return temporal_split(ratings, test_frac=test_frac)
    raise KeyError(f"Unknown split strategy {strategy!r}. Use 'random' or 'temporal'.")
