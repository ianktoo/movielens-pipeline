"""Tests for data loading and synthetic generation."""

from __future__ import annotations

import numpy as np

from movielens import config, data


def test_synthetic_shapes(small_ds):
    assert small_ds.n_ratings > 0
    assert small_ds.n_users <= 120
    assert small_ds.n_movies <= 60
    assert set(small_ds.ratings.columns) == {"userId", "movieId", "rating", "timestamp"}
    assert set(small_ds.movies.columns) == {"movieId", "title", "genres"}


def test_synthetic_is_deterministic():
    a = data.make_synthetic(n_users=50, n_movies=30, n_ratings=1000, seed=123)
    b = data.make_synthetic(n_users=50, n_movies=30, n_ratings=1000, seed=123)
    assert a.ratings.equals(b.ratings)
    assert a.movies.equals(b.movies)


def test_ratings_on_legal_scale(small_ds):
    r = small_ds.ratings["rating"]
    assert r.min() >= config.RATING_MIN
    assert r.max() <= config.RATING_MAX
    # All on the half-star grid.
    assert np.allclose((r * 2) % 1, 0)


def test_no_duplicate_user_movie_pairs(small_ds):
    dup = small_ds.ratings.duplicated(["userId", "movieId"]).sum()
    assert dup == 0


def test_sparsity_between_0_and_1(small_ds):
    assert 0.0 <= small_ds.sparsity <= 1.0


def test_summary_keys(small_ds):
    s = small_ds.summary()
    assert {"n_ratings", "n_users", "n_movies", "sparsity", "avg_rating"} <= set(s)


def test_load_or_synthesize_falls_back(monkeypatch):
    # Force the "raw not available" path and confirm we still get data.
    monkeypatch.setattr(data, "raw_available", lambda *a, **k: False)
    ds = data.load_or_synthesize(n_users=10, n_movies=8, n_ratings=100)
    assert ds.name == "synthetic"
    assert ds.n_ratings > 0
