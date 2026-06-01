"""Feature engineering helpers.

The collaborative-filtering models work straight off the (user, movie, rating)
triples, so they don't strictly need engineered features. But a real pipeline
builds features, and they're useful for content-based ideas, for inspecting the
data, and as a talking point in class. Everything here is pure
DataFrame-in / DataFrame-out so it's easy to test and compose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset


def user_features(ds: Dataset) -> pd.DataFrame:
    """One row per user: how much and how generously they rate."""
    g = ds.ratings.groupby("userId")["rating"]
    feat = pd.DataFrame(
        {
            "n_ratings": g.size(),
            "avg_rating": g.mean(),
            "rating_std": g.std().fillna(0.0),
        }
    ).reset_index()
    return feat


def movie_features(ds: Dataset) -> pd.DataFrame:
    """One row per movie: popularity, average rating, and genre flags."""
    g = ds.ratings.groupby("movieId")["rating"]
    base = pd.DataFrame(
        {
            "n_ratings": g.size(),
            "avg_rating": g.mean(),
            "rating_std": g.std().fillna(0.0),
        }
    ).reset_index()

    genres = genre_dummies(ds.movies)
    return base.merge(genres, on="movieId", how="left").fillna(0)


def genre_dummies(movies: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the pipe-delimited ``genres`` column.

    Returns a DataFrame with ``movieId`` plus one 0/1 column per genre,
    prefixed ``genre_`` — handy as content-based features.
    """
    # str.get_dummies is purpose-built for delimited tags and is robust to the
    # pandas "string" dtype, unlike a crosstab over an exploded column.
    genres = movies["genres"].fillna("").astype(str)
    dummies = genres.str.get_dummies("|")
    dummies.columns = [f"genre_{c}" for c in dummies.columns]
    dummies.insert(0, "movieId", movies["movieId"].to_numpy())
    return dummies


def extract_year(movies: pd.DataFrame) -> pd.DataFrame:
    """Pull the release year out of titles like ``"Toy Story (1995)"``."""
    year = movies["title"].astype(str).str.extract(r"\((\d{4})\)")[0]
    out = movies[["movieId"]].copy()
    out["year"] = pd.to_numeric(year, errors="coerce").astype("Int64")
    return out
