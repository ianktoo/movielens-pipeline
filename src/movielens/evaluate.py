"""Metrics for judging a recommender.

Two families of metric, because a recommender has two jobs:

* **Accuracy** -- when we predict a rating, how close is it?
    :func:`rmse`, :func:`mae`.
* **Ranking quality** -- of the items we put at the top, how many are good?
    :func:`precision_at_k`, :func:`recall_at_k`.

:func:`evaluate_model` ties them together: given a fitted model and a test set,
it returns a tidy dict of every metric so the notebook can build a comparison
table across models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import BaseModel


# ---------------------------------------------------------------------------
# Rating-accuracy metrics
# ---------------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    """Root mean squared error — penalizes big misses heavily."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error — average miss in 'stars'. Easy to explain."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Ranking metrics (top-K)
# ---------------------------------------------------------------------------
def precision_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int = 10,
) -> float:
    """Of the top-k we recommended, what fraction were actually relevant?"""
    if k == 0:
        return 0.0
    topk = recommended[:k]
    if not topk:
        return 0.0
    hits = sum(1 for item in topk if item in relevant)
    return hits / len(topk)


def recall_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int = 10,
) -> float:
    """Of all the relevant items, what fraction did we surface in the top-k?"""
    if not relevant:
        return 0.0
    topk = recommended[:k]
    hits = sum(1 for item in topk if item in relevant)
    return hits / len(relevant)


def evaluate_model(
    model: BaseModel,
    test: pd.DataFrame,
    k: int = 10,
    relevant_threshold: float = 4.0,
    max_ranking_users: int = 200,
) -> dict[str, float]:
    """Compute accuracy + ranking metrics for a fitted model on a test set.

    Parameters
    ----------
    model:
        An already-``fit`` model.
    test:
        Held-out ratings with ``userId, movieId, rating``.
    k:
        Cutoff for precision/recall.
    relevant_threshold:
        A test rating at or above this counts as a "good" (relevant) movie.
    max_ranking_users:
        Ranking metrics loop per-user and are the slow part; cap how many users
        we score so the notebook stays snappy. Accuracy uses the full test set.
    """
    # --- accuracy over every test rating --------------------------------
    preds = model.predict(test["userId"].to_numpy(), test["movieId"].to_numpy())
    out = {
        "rmse": rmse(test["rating"], preds),
        "mae": mae(test["rating"], preds),
    }

    # --- ranking over a sample of users ---------------------------------
    precisions: list[float] = []
    recalls: list[float] = []

    test_users = test["userId"].unique()
    if len(test_users) > max_ranking_users:
        # Deterministic subsample for stable, fast demos.
        rng = np.random.default_rng(0)
        test_users = rng.choice(test_users, size=max_ranking_users, replace=False)

    test_by_user = {u: g for u, g in test.groupby("userId")}

    for u in test_users:
        g = test_by_user[u]
        relevant = set(g.loc[g["rating"] >= relevant_threshold, "movieId"].astype(int))
        if not relevant:
            continue
        if not model.knows_user(int(u)):
            continue
        scored = model.predict_all_for_user(int(u))
        # Recommend the highest-scored movies that appear in this user's test set
        # (standard "ranking among the candidate test items" protocol).
        candidates = [int(m) for m in g["movieId"].astype(int)]
        scored = scored.reindex(candidates).dropna().sort_values(ascending=False)
        ranked = [int(m) for m in scored.index]
        precisions.append(precision_at_k(ranked, relevant, k))
        recalls.append(recall_at_k(ranked, relevant, k))

    out[f"precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
    out[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
    out["n_ranking_users"] = len(precisions)
    return out
