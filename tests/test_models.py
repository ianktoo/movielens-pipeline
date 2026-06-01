"""Tests for the models — interface, sanity, and that learning beats baselines."""

from __future__ import annotations

import numpy as np
import pytest

from movielens import splitting
from movielens.evaluate import rmse
from movielens.models import (
    MODELS,
    BiasModel,
    GlobalMeanModel,
    ItemItemCF,
    MatrixFactorization,
)


@pytest.fixture(scope="module")
def split(small_ds):
    return splitting.temporal_split(small_ds.ratings, test_frac=0.2)


@pytest.mark.parametrize("name", list(MODELS))
def test_predictions_on_legal_scale(small_ds, split, name):
    model = MODELS[name]().fit(split.train)
    preds = model.predict(split.test["userId"].to_numpy(), split.test["movieId"].to_numpy())
    assert len(preds) == len(split.test)
    assert preds.min() >= 0.5 - 1e-9
    assert preds.max() <= 5.0 + 1e-9
    assert not np.isnan(preds).any()


def test_global_mean_predicts_constant(split):
    model = GlobalMeanModel().fit(split.train)
    preds = model.predict([1, 2, 3], [1, 2, 3])
    assert np.allclose(preds, preds[0])


def test_bias_beats_global_mean(split):
    gm = GlobalMeanModel().fit(split.train)
    bias = BiasModel().fit(split.train)
    gm_rmse = rmse(split.test["rating"], gm.predict(split.test["userId"], split.test["movieId"]))
    bias_rmse = rmse(split.test["rating"], bias.predict(split.test["userId"], split.test["movieId"]))
    assert bias_rmse <= gm_rmse


def test_mf_beats_or_matches_global_mean(split):
    gm = GlobalMeanModel().fit(split.train)
    mf = MatrixFactorization(n_factors=10).fit(split.train)
    gm_rmse = rmse(split.test["rating"], gm.predict(split.test["userId"], split.test["movieId"]))
    mf_rmse = rmse(split.test["rating"], mf.predict(split.test["userId"], split.test["movieId"]))
    assert mf_rmse <= gm_rmse + 1e-6


def test_cold_start_user_falls_back_gracefully(split):
    model = BiasModel().fit(split.train)
    # An unknown user id should not crash; prediction stays on-scale.
    pred = model.predict([10_000_000], [int(split.train["movieId"].iloc[0])])
    assert 0.5 <= pred[0] <= 5.0


def test_predict_all_for_user_covers_all_movies(small_ds, split):
    model = MatrixFactorization(n_factors=8).fit(split.train)
    a_user = int(split.train["userId"].iloc[0])
    scores = model.predict_all_for_user(a_user)
    assert len(scores) == split.train["movieId"].nunique()
    assert scores.notna().all()


def test_item_item_cf_runs(split):
    model = ItemItemCF(k_neighbors=10).fit(split.train)
    preds = model.predict(split.test["userId"].head(20), split.test["movieId"].head(20))
    assert len(preds) == 20
