"""Turn a fitted model into human-readable movie recommendations.

The model speaks in ids and predicted ratings; this module joins those back to
movie titles and filters out films the user has already seen, so the notebook
can print a clean "Top 10 for user 42" table — the payoff moment of the demo.
"""

from __future__ import annotations

import pandas as pd

from .data import Dataset
from .models import BaseModel


def recommend_for_user(
    model: BaseModel,
    dataset: Dataset,
    user_id: int,
    n: int = 10,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """Return the top-``n`` recommended movies for one user, with titles.

    Parameters
    ----------
    model:
        A fitted model.
    dataset:
        Provides the movies table (titles/genres) and the user's seen history.
    user_id:
        Whom to recommend for.
    n:
        How many movies to return.
    exclude_seen:
        Drop movies the user has already rated (the usual behavior — you don't
        recommend a film someone already watched).
    """
    scores = model.predict_all_for_user(int(user_id)).sort_values(ascending=False)

    if exclude_seen:
        seen = set(
            dataset.ratings.loc[dataset.ratings["userId"] == user_id, "movieId"].astype(int)
        )
        scores = scores[~scores.index.isin(seen)]

    top = scores.head(n).rename_axis("movieId").reset_index()
    top.columns = ["movieId", "predicted_rating"]

    movies = dataset.movies[["movieId", "title", "genres"]]
    out = top.merge(movies, on="movieId", how="left")
    out["predicted_rating"] = out["predicted_rating"].round(3)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out[["rank", "movieId", "title", "genres", "predicted_rating"]]


def user_history(
    dataset: Dataset,
    user_id: int,
    n: int = 10,
) -> pd.DataFrame:
    """The user's own top-rated movies — useful to show *why* a rec makes sense.

    Putting "what they liked" next to "what we recommend" lets the class sanity
    check the model with their own eyes.
    """
    hist = dataset.ratings[dataset.ratings["userId"] == user_id]
    hist = hist.sort_values("rating", ascending=False).head(n)
    out = hist.merge(dataset.movies[["movieId", "title", "genres"]], on="movieId", how="left")
    return out[["movieId", "title", "genres", "rating"]].reset_index(drop=True)
