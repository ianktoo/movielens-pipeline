"""Exploratory data analysis: summary tables and plots.

Every function takes a :class:`~movielens.data.Dataset` and returns either a
DataFrame (a table you can display) or a matplotlib ``Figure`` (a chart you can
show). Keeping plotting here — out of the notebook — means the notebook cells
stay short and readable, and the charts are reusable and testable.
"""

from __future__ import annotations

# Note: we deliberately do NOT call matplotlib.use(...) here. Forcing a backend
# at import time would override a notebook's "%matplotlib inline" and silently
# stop charts from displaying. In headless/test runs matplotlib auto-selects the
# non-interactive Agg backend on its own, and simply *creating* a Figure (which
# is all the tests do) works on any backend.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import Dataset


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def rating_distribution(ds: Dataset) -> pd.DataFrame:
    """Count of each star value (0.5 ... 5.0)."""
    counts = ds.ratings["rating"].value_counts().sort_index()
    return counts.rename_axis("rating").reset_index(name="count")


def activity_table(ds: Dataset) -> pd.DataFrame:
    """Per-user and per-movie activity stats, side by side."""
    per_user = ds.ratings.groupby("userId").size()
    per_movie = ds.ratings.groupby("movieId").size()

    def _describe(s: pd.Series) -> dict:
        return {
            "min": int(s.min()),
            "median": float(s.median()),
            "mean": round(float(s.mean()), 1),
            "max": int(s.max()),
        }

    return pd.DataFrame(
        {"ratings_per_user": _describe(per_user), "ratings_per_movie": _describe(per_movie)}
    )


def top_movies(ds: Dataset, n: int = 10, min_ratings: int = 50) -> pd.DataFrame:
    """Highest-rated movies that clear a minimum number of ratings.

    The ``min_ratings`` filter avoids the classic trap of a single 5-star
    rating topping the chart.
    """
    stats = (
        ds.ratings.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .query("count >= @min_ratings")
        .sort_values("mean", ascending=False)
        .head(n)
        .reset_index()
    )
    stats = stats.merge(ds.movies[["movieId", "title"]], on="movieId", how="left")
    stats["mean"] = stats["mean"].round(3)
    return stats[["movieId", "title", "mean", "count"]].rename(
        columns={"mean": "avg_rating", "count": "n_ratings"}
    )


def genre_counts(ds: Dataset) -> pd.DataFrame:
    """How many movies fall under each genre (genres are pipe-delimited)."""
    exploded = (
        ds.movies["genres"].fillna("")
        .str.split("|")
        .explode()
        .replace("", np.nan)
        .dropna()
    )
    return (
        exploded.value_counts()
        .rename_axis("genre")
        .reset_index(name="n_movies")
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_rating_distribution(ds: Dataset):
    """Bar chart of how often each star value appears."""
    dist = rating_distribution(ds)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dist["rating"], dist["count"], width=0.4, color="#4C72B0")
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Number of ratings")
    ax.set_title(f"Rating distribution — {ds.name}")
    fig.tight_layout()
    return fig


def plot_activity_long_tail(ds: Dataset):
    """Log-log plot showing the long-tail of ratings-per-movie.

    This is the single most important EDA chart for a recommender: it shows
    that a few blockbusters get most of the ratings while the vast majority of
    movies are rated rarely — the reason cold-start is hard.
    """
    per_movie = ds.ratings.groupby("movieId").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(per_movie) + 1), per_movie.to_numpy(), color="#C44E52")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Movie rank (most-rated first, log scale)")
    ax.set_ylabel("Number of ratings (log scale)")
    ax.set_title(f"The long tail of movie popularity — {ds.name}")
    fig.tight_layout()
    return fig


def plot_genre_counts(ds: Dataset, top: int = 15):
    """Horizontal bar chart of the most common genres."""
    gc = genre_counts(ds).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(gc["genre"], gc["n_movies"], color="#55A868")
    ax.set_xlabel("Number of movies")
    ax.set_title(f"Most common genres — {ds.name}")
    fig.tight_layout()
    return fig
