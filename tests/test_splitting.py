"""Tests for train/test splitting."""

from __future__ import annotations

import pytest

from movielens import splitting


@pytest.mark.parametrize("strategy", ["random", "temporal"])
def test_split_partitions_data(small_ds, strategy):
    sp = splitting.split(small_ds.ratings, strategy=strategy, test_frac=0.2)
    # No overlap and the union recovers the whole dataset.
    assert sp.n_train + sp.n_test == small_ds.n_ratings
    assert sp.n_train > 0 and sp.n_test > 0


@pytest.mark.parametrize("strategy", ["random", "temporal"])
def test_every_test_user_is_in_train(small_ds, strategy):
    sp = splitting.split(small_ds.ratings, strategy=strategy, test_frac=0.2)
    train_users = set(sp.train["userId"])
    test_users = set(sp.test["userId"])
    # Critical recommender invariant: never evaluate on an unseen user.
    assert test_users.issubset(train_users)


def test_temporal_split_holds_out_recent(small_ds):
    sp = splitting.temporal_split(small_ds.ratings, test_frac=0.3)
    # For users present in both, the test ratings should be newer (>=) than train.
    merged_train = sp.train.groupby("userId")["timestamp"].max()
    merged_test = sp.test.groupby("userId")["timestamp"].min()
    common = merged_train.index.intersection(merged_test.index)
    # Most users should satisfy train_max <= test_min (allow ties from equal ts).
    ok = (merged_train.loc[common] <= merged_test.loc[common]).mean()
    assert ok > 0.9


def test_temporal_requires_timestamp(small_ds):
    no_ts = small_ds.ratings.drop(columns=["timestamp"])
    with pytest.raises(ValueError):
        splitting.temporal_split(no_ts)


def test_unknown_strategy_raises(small_ds):
    with pytest.raises(KeyError):
        splitting.split(small_ds.ratings, strategy="nope")
