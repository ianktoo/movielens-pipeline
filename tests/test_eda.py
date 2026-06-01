"""Smoke tests for EDA tables and plots.

We don't assert on pixels — just that each table has the expected columns and
each plot builds a matplotlib Figure without error (so a broken chart can't
sneak into the class demo).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: build figures, never open a window
from matplotlib.figure import Figure

from movielens import eda


def test_rating_distribution_table(small_ds):
    df = eda.rating_distribution(small_ds)
    assert {"rating", "count"} <= set(df.columns)
    assert df["count"].sum() == small_ds.n_ratings


def test_activity_table(small_ds):
    df = eda.activity_table(small_ds)
    assert {"ratings_per_user", "ratings_per_movie"} <= set(df.columns)


def test_top_movies(small_ds):
    df = eda.top_movies(small_ds, n=5, min_ratings=1)
    assert len(df) <= 5
    assert {"movieId", "title", "avg_rating", "n_ratings"} <= set(df.columns)
    # Sorted best-first.
    assert df["avg_rating"].is_monotonic_decreasing


def test_genre_counts(small_ds):
    df = eda.genre_counts(small_ds)
    assert {"genre", "n_movies"} <= set(df.columns)
    assert (df["n_movies"] > 0).all()


def test_plots_return_figures(small_ds):
    assert isinstance(eda.plot_rating_distribution(small_ds), Figure)
    assert isinstance(eda.plot_activity_long_tail(small_ds), Figure)
    assert isinstance(eda.plot_genre_counts(small_ds), Figure)
