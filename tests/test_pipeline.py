"""Tests for the end-to-end pipeline orchestrator and recommendations."""

from __future__ import annotations

from movielens import features, recommend
from movielens.models import BiasModel, ModelSpec
from movielens.pipeline import run_pipeline


def test_run_pipeline_end_to_end(small_ds):
    result = run_pipeline(
        dataset=small_ds,
        sample_strategy="active_users",
        sample_size=80,
        split_strategy="temporal",
        model_specs=[ModelSpec("global_mean"), ModelSpec("bias")],
        k=5,
        verbose=False,
    )
    assert set(result.scores["model"]) == {"global_mean", "bias"}
    assert "rmse" in result.scores.columns
    # Results are sorted best-first by rmse.
    assert result.scores["rmse"].is_monotonic_increasing
    # bias should not be worse than the naive baseline.
    by_model = result.scores.set_index("model")["rmse"]
    assert by_model["bias"] <= by_model["global_mean"] + 1e-6


def test_run_pipeline_without_sampling(small_ds):
    result = run_pipeline(
        dataset=small_ds,
        sample_strategy=None,
        model_specs=[ModelSpec("bias")],
        verbose=False,
    )
    assert result.dataset.n_ratings == small_ds.n_ratings


def test_recommend_for_user_shape_and_excludes_seen(small_ds):
    from movielens import splitting

    sp = splitting.temporal_split(small_ds.ratings, test_frac=0.2)
    model = BiasModel().fit(sp.train)
    a_user = int(sp.train["userId"].iloc[0])

    recs = recommend.recommend_for_user(model, small_ds, a_user, n=5, exclude_seen=True)
    assert len(recs) <= 5
    assert list(recs.columns) == ["rank", "movieId", "title", "genres", "predicted_rating"]

    seen = set(small_ds.ratings.loc[small_ds.ratings["userId"] == a_user, "movieId"])
    assert not set(recs["movieId"]).intersection(seen)


def test_user_history(small_ds):
    a_user = int(small_ds.ratings["userId"].iloc[0])
    hist = recommend.user_history(small_ds, a_user, n=5)
    assert len(hist) <= 5
    assert "title" in hist.columns


def test_feature_engineering(small_ds):
    uf = features.user_features(small_ds)
    mf = features.movie_features(small_ds)
    assert "avg_rating" in uf.columns
    assert any(c.startswith("genre_") for c in mf.columns)
    years = features.extract_year(small_ds.movies)
    assert "year" in years.columns
