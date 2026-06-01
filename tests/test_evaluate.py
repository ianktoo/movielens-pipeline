"""Tests for metrics and end-to-end evaluation."""

from __future__ import annotations

from movielens import splitting
from movielens.evaluate import (
    evaluate_model,
    mae,
    precision_at_k,
    recall_at_k,
    rmse,
)
from movielens.models import BiasModel


def test_rmse_zero_on_perfect():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_mae_known_value():
    assert mae([1.0, 2.0], [2.0, 2.0]) == 0.5


def test_rmse_ge_mae():
    y = [1, 2, 3, 4]
    p = [1.5, 1.0, 4.0, 2.0]
    assert rmse(y, p) >= mae(y, p)


def test_precision_at_k():
    rec = [1, 2, 3, 4]
    relevant = {2, 4, 9}
    # top-2 = [1,2], one hit -> 0.5
    assert precision_at_k(rec, relevant, k=2) == 0.5


def test_recall_at_k():
    rec = [1, 2, 3, 4]
    relevant = {2, 4, 9}
    # top-4 surfaces 2 of the 3 relevant -> 2/3
    assert abs(recall_at_k(rec, relevant, k=4) - (2 / 3)) < 1e-9


def test_precision_recall_edge_cases():
    assert precision_at_k([], {1}, k=5) == 0.0
    assert recall_at_k([1, 2], set(), k=5) == 0.0


def test_evaluate_model_returns_all_metrics(small_ds):
    sp = splitting.temporal_split(small_ds.ratings, test_frac=0.2)
    model = BiasModel().fit(sp.train)
    scores = evaluate_model(model, sp.test, k=5)
    for key in ("rmse", "mae", "precision@5", "recall@5", "n_ranking_users"):
        assert key in scores
    assert scores["rmse"] >= 0
    assert 0.0 <= scores["precision@5"] <= 1.0
