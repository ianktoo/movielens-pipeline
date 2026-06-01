"""Tests for sampling strategies and persistence."""

from __future__ import annotations

import pytest

from movielens import sampling


def test_strategies_registered():
    strategies = sampling.available_strategies()
    assert {"random_users", "active_users", "dense"} <= set(strategies)


@pytest.mark.parametrize("strategy", ["random_users", "active_users", "dense"])
def test_build_sample_is_subset(small_ds, strategy):
    sample = sampling.build_sample(small_ds, strategy=strategy, size=40)
    # Sample users/movies must be a subset of the source.
    assert set(sample.ratings["userId"]).issubset(set(small_ds.ratings["userId"]))
    assert set(sample.ratings["movieId"]).issubset(set(small_ds.ratings["movieId"]))
    assert sample.n_ratings <= small_ds.n_ratings
    # Movies table only contains movies that appear in the sample.
    assert set(sample.movies["movieId"]) == set(sample.ratings["movieId"])


def test_active_users_picks_most_active(small_ds):
    sample = sampling.build_sample(small_ds, strategy="active_users", size=10)
    assert sample.n_users <= 10
    # The most active user overall should be present.
    most_active = small_ds.ratings.groupby("userId").size().idxmax()
    assert most_active in set(sample.ratings["userId"])


def test_dense_respects_top_movies(small_ds):
    sample = sampling.build_sample(small_ds, strategy="dense", size=50, top_movies=15)
    assert sample.n_movies <= 15


def test_build_sample_is_deterministic(small_ds):
    a = sampling.build_sample(small_ds, strategy="random_users", size=30, seed=5)
    b = sampling.build_sample(small_ds, strategy="random_users", size=30, seed=5)
    assert a.ratings.equals(b.ratings)


def test_unknown_strategy_raises(small_ds):
    with pytest.raises(KeyError):
        sampling.build_sample(small_ds, strategy="does_not_exist")


def test_save_and_load_roundtrip(tiny_ds, tmp_path):
    sample = sampling.build_sample(tiny_ds, strategy="random_users", size=15)
    sampling.save_sample(sample, key="unit_test", samples_dir=tmp_path)
    assert "unit_test" in sampling.list_saved_samples(samples_dir=tmp_path)

    loaded = sampling.load_sample("unit_test", samples_dir=tmp_path)
    assert loaded.n_ratings == sample.n_ratings
    assert set(loaded.ratings["movieId"]) == set(sample.ratings["movieId"])
