"""Recommendation models, from dead-simple baselines to matrix factorization.

Every model shares one tiny interface so they're interchangeable in the
notebook and in evaluation:

    model = BiasModel().fit(train_df)
    preds = model.predict(user_ids, movie_ids)   # -> np.ndarray of ratings

The progression is deliberately pedagogical:

1. :class:`GlobalMeanModel`  -- "predict the average rating for everything".
2. :class:`BiasModel`        -- add a per-user and per-movie offset. Surprisingly
                                strong, and the standard baseline to beat.
3. :class:`MatrixFactorization` -- learn latent taste/appeal vectors via SVD on
                                the bias residuals. The real recommender.
4. :class:`ItemItemCF`       -- "users who liked this also liked..." via cosine
                                similarity between movies.

Each model also exposes ``predict_all_for_user`` so :mod:`movielens.recommend`
can produce top-N lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from . import config


def _clip(x: np.ndarray) -> np.ndarray:
    """Keep predictions on the legal 0.5 - 5.0 rating scale."""
    return np.clip(x, config.RATING_MIN, config.RATING_MAX)


class BaseModel:
    """Shared interface + id<->index bookkeeping for all models."""

    def __init__(self) -> None:
        self.global_mean_: float = 0.0
        self.user_index_: dict[int, int] = {}
        self.movie_index_: dict[int, int] = {}

    # -- helpers ----------------------------------------------------------
    def _build_indexes(self, ratings: pd.DataFrame) -> None:
        users = np.sort(ratings["userId"].unique())
        movies = np.sort(ratings["movieId"].unique())
        self.user_index_ = {int(u): i for i, u in enumerate(users)}
        self.movie_index_ = {int(m): i for i, m in enumerate(movies)}
        self.users_ = users
        self.movies_ = movies

    def knows_user(self, user_id: int) -> bool:
        return int(user_id) in self.user_index_

    def knows_movie(self, movie_id: int) -> bool:
        return int(movie_id) in self.movie_index_

    # -- interface (subclasses implement fit/predict) ---------------------
    def fit(self, ratings: pd.DataFrame) -> "BaseModel":  # pragma: no cover
        raise NotImplementedError

    def predict(self, user_ids, movie_ids) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict_all_for_user(self, user_id: int) -> pd.Series:
        """Predicted rating for every known movie, indexed by movieId.

        Default implementation just calls :meth:`predict` for all movies; models
        with a cheaper bulk path can override this.
        """
        movie_ids = self.movies_
        preds = self.predict(np.full(len(movie_ids), user_id), movie_ids)
        return pd.Series(preds, index=movie_ids, name="pred")


class GlobalMeanModel(BaseModel):
    """The humblest baseline: predict the global average rating, always."""

    def fit(self, ratings: pd.DataFrame) -> "GlobalMeanModel":
        self._build_indexes(ratings)
        self.global_mean_ = float(ratings["rating"].mean())
        return self

    def predict(self, user_ids, movie_ids) -> np.ndarray:
        n = len(np.asarray(user_ids))
        return _clip(np.full(n, self.global_mean_))


class BiasModel(BaseModel):
    r"""Global mean + regularized user and movie biases.

    Prediction:  :math:`\hat r_{ui} = \mu + b_u + b_i`

    Biases are computed with shrinkage (regularization) so that users/movies
    with very few ratings are pulled toward zero instead of overfitting:

    .. math::
        b_i = \frac{\sum_u (r_{ui} - \mu)}{\lambda + n_i}, \qquad
        b_u = \frac{\sum_i (r_{ui} - \mu - b_i)}{\lambda + n_u}
    """

    def __init__(self, reg: float = 10.0) -> None:
        super().__init__()
        self.reg = reg
        self.user_bias_: np.ndarray = np.array([])
        self.movie_bias_: np.ndarray = np.array([])

    def fit(self, ratings: pd.DataFrame) -> "BiasModel":
        self._build_indexes(ratings)
        mu = float(ratings["rating"].mean())
        self.global_mean_ = mu

        df = ratings[["userId", "movieId", "rating"]].copy()
        df["dev"] = df["rating"] - mu

        # Movie bias first (shrunk toward 0 by self.reg).
        movie_stats = df.groupby("movieId")["dev"].agg(["sum", "count"])
        movie_bias = movie_stats["sum"] / (self.reg + movie_stats["count"])
        self.movie_bias_ = np.zeros(len(self.movie_index_))
        for mid, b in movie_bias.items():
            self.movie_bias_[self.movie_index_[int(mid)]] = b

        # User bias on the residual after removing movie bias.
        df["mb"] = df["movieId"].map(movie_bias).astype(float)
        df["udev"] = df["dev"] - df["mb"]
        user_stats = df.groupby("userId")["udev"].agg(["sum", "count"])
        user_bias = user_stats["sum"] / (self.reg + user_stats["count"])
        self.user_bias_ = np.zeros(len(self.user_index_))
        for uid, b in user_bias.items():
            self.user_bias_[self.user_index_[int(uid)]] = b

        return self

    def _bias_vector(self, user_ids, movie_ids) -> np.ndarray:
        # Vectorized with pandas reindex: gather each id's bias in one shot, with
        # unknown (cold-start) ids becoming NaN -> 0. Far faster than a Python
        # loop when scoring millions of test ratings.
        ub = pd.Series(self.user_bias_, index=self.users_)
        mb = pd.Series(self.movie_bias_, index=self.movies_)
        u = ub.reindex(np.asarray(user_ids)).to_numpy()
        m = mb.reindex(np.asarray(movie_ids)).to_numpy()
        return self.global_mean_ + np.nan_to_num(u) + np.nan_to_num(m)

    def predict(self, user_ids, movie_ids) -> np.ndarray:
        return _clip(self._bias_vector(user_ids, movie_ids))


class MatrixFactorization(BaseModel):
    """Latent-factor model via truncated SVD on the bias residuals.

    Idea: after subtracting the bias prediction, what's left is the
    *interaction* — does THIS user have a taste for what THIS movie offers?
    We factor that residual matrix into ``n_factors`` latent dimensions
    (think: "amount of action", "amount of romance", ... discovered
    automatically). The dot product of a user's taste and a movie's appeal
    reconstructs the residual.

    Final prediction:  :math:`\\hat r_{ui} = \\mu + b_u + b_i + p_u \\cdot q_i`
    """

    def __init__(self, n_factors: int = 20, reg: float = 10.0) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.bias = BiasModel(reg=reg)

    def fit(self, ratings: pd.DataFrame) -> "MatrixFactorization":
        self.bias.fit(ratings)
        self._build_indexes(ratings)
        self.global_mean_ = self.bias.global_mean_

        n_users = len(self.user_index_)
        n_movies = len(self.movie_index_)

        # Map ids -> row/col indices.
        rows = ratings["userId"].map(self.user_index_).to_numpy()
        cols = ratings["movieId"].map(self.movie_index_).to_numpy()

        # Residual = actual - bias prediction.
        bias_pred = self.bias._bias_vector(
            ratings["userId"].to_numpy(), ratings["movieId"].to_numpy()
        )
        resid = ratings["rating"].to_numpy() - bias_pred

        # k must be < min(matrix dims) for svds.
        k = int(min(self.n_factors, min(n_users, n_movies) - 1))
        k = max(k, 1)

        mat = csr_matrix((resid, (rows, cols)), shape=(n_users, n_movies))
        u, s, vt = svds(mat, k=k)
        # svds returns factors in ascending order of singular value; fold sqrt(s)
        # into both sides so prediction is a plain dot product.
        sqrt_s = np.sqrt(s)
        self.user_factors_ = u * sqrt_s          # (n_users, k)
        self.movie_factors_ = (vt.T * sqrt_s)    # (n_movies, k)
        self.k_ = k
        return self

    def predict(self, user_ids, movie_ids) -> np.ndarray:
        user_ids = np.asarray(user_ids)
        movie_ids = np.asarray(movie_ids)
        base = self.bias._bias_vector(user_ids, movie_ids)

        # Map ids -> integer row/col positions in one vectorized pass; unknown
        # ids become NaN and are skipped (their prediction stays at the bias).
        u_pos = pd.Series(np.arange(len(self.users_)), index=self.users_).reindex(user_ids).to_numpy()
        m_pos = pd.Series(np.arange(len(self.movies_)), index=self.movies_).reindex(movie_ids).to_numpy()
        valid = ~np.isnan(u_pos) & ~np.isnan(m_pos)
        if valid.any():
            ui = u_pos[valid].astype(int)
            mi = m_pos[valid].astype(int)
            # Row-wise dot product of the matched user/movie latent vectors.
            dots = np.sum(self.user_factors_[ui] * self.movie_factors_[mi], axis=1)
            base[valid] += dots
        return _clip(base)

    def predict_all_for_user(self, user_id: int) -> pd.Series:
        # Vectorized bulk path — much faster than looping for top-N.
        base = self.bias.global_mean_ + self.bias.movie_bias_.copy()
        u = int(user_id)
        if u in self.user_index_:
            base = base + self.bias.user_bias_[self.user_index_[u]]
            base = base + self.movie_factors_ @ self.user_factors_[self.user_index_[u]]
        return pd.Series(_clip(base), index=self.movies_, name="pred")


class ItemItemCF(BaseModel):
    """Item-based collaborative filtering: "users who liked X also liked Y".

    We compute cosine similarity between movies based on the (mean-centered)
    ratings they share, then predict a user's rating for a movie as the
    similarity-weighted average of how that user rated the most similar movies
    they've already seen.
    """

    def __init__(self, k_neighbors: int = 30, max_movies: int = 5000) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        # Item-item CF precomputes an n_movies x n_movies similarity matrix, which
        # is O(n_movies^2) memory. This guard keeps it honest: it's designed for
        # the *dense* sample (few hundred popular movies), not the full catalogue.
        self.max_movies = max_movies

    def fit(self, ratings: pd.DataFrame) -> "ItemItemCF":
        self._build_indexes(ratings)
        self.global_mean_ = float(ratings["rating"].mean())

        n_users = len(self.user_index_)
        n_movies = len(self.movie_index_)
        if n_movies > self.max_movies:
            raise ValueError(
                f"ItemItemCF builds a {n_movies}x{n_movies} similarity matrix, "
                f"which exceeds max_movies={self.max_movies}. Use the 'dense' "
                "sample (which trims to the most popular movies), or raise "
                "max_movies if you have the RAM."
            )

        rows = ratings["userId"].map(self.user_index_).to_numpy()
        cols = ratings["movieId"].map(self.movie_index_).to_numpy()

        # Per-movie mean, for centering.
        self.movie_mean_ = (
            ratings.groupby("movieId")["rating"].mean()
            .reindex(self.movies_).to_numpy()
        )
        centered = ratings["rating"].to_numpy() - self.movie_mean_[cols]

        # Movies x users sparse matrix of centered ratings.
        mat = csr_matrix((centered, (cols, rows)), shape=(n_movies, n_users))

        # Precompute the FULL cosine-similarity matrix once. For the dense sample
        # (~800 movies) this is a tiny, fast dense matrix, and it turns each later
        # prediction into cheap array indexing instead of a sparse matmul.
        norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
        norms[norms == 0] = 1e-9
        sim = (mat @ mat.T).toarray()
        sim /= np.outer(norms, norms)
        np.fill_diagonal(sim, 0.0)  # a movie is never its own neighbour
        self.sim_ = sim

        # Per-user: parallel arrays of (movie_idx, centered_rating) for fast lookup.
        self._user_items: dict[int, np.ndarray] = {}
        self._user_centered: dict[int, np.ndarray] = {}
        for u_idx, m_idx, c in zip(rows, cols, centered):
            self._user_items.setdefault(u_idx, []).append(m_idx)
            self._user_centered.setdefault(u_idx, []).append(c)
        self._user_items = {u: np.asarray(v) for u, v in self._user_items.items()}
        self._user_centered = {u: np.asarray(v) for u, v in self._user_centered.items()}
        return self

    def predict(self, user_ids, movie_ids) -> np.ndarray:
        user_ids = np.asarray(user_ids)
        movie_ids = np.asarray(movie_ids)
        out = np.full(len(user_ids), self.global_mean_, dtype=float)

        for idx in range(len(user_ids)):
            u = int(user_ids[idx])
            m = int(movie_ids[idx])
            if u not in self.user_index_ or m not in self.movie_index_:
                continue
            m_idx = self.movie_index_[m]
            u_idx = self.user_index_[u]
            rated = self._user_items.get(u_idx)
            if rated is None or len(rated) == 0:
                out[idx] = self.movie_mean_[m_idx]
                continue

            # Similarities from the target movie to everything this user rated.
            sims = self.sim_[m_idx, rated]
            centered = self._user_centered[u_idx]
            pos = sims > 0
            if not pos.any():
                out[idx] = self.movie_mean_[m_idx]
                continue
            sims, centered = sims[pos], centered[pos]

            # Keep the k most-similar neighbours (argpartition = O(n), no full sort).
            if len(sims) > self.k_neighbors:
                top = np.argpartition(sims, -self.k_neighbors)[-self.k_neighbors:]
                sims, centered = sims[top], centered[top]

            den = np.abs(sims).sum()
            adj = float((sims * centered).sum() / den) if den > 0 else 0.0
            out[idx] = self.movie_mean_[m_idx] + adj
        return _clip(out)

    def predict_all_for_user(self, user_id: int) -> pd.Series:
        # Vectorized bulk path for top-N: one matrix-vector product over all movies.
        u = int(user_id)
        base = self.movie_mean_.copy()
        if u in self.user_index_:
            u_idx = self.user_index_[u]
            rated = self._user_items.get(u_idx)
            if rated is not None and len(rated):
                centered = self._user_centered[u_idx]
                sub = self.sim_[:, rated]          # (n_movies, n_rated)
                sub_pos = np.where(sub > 0, sub, 0.0)
                num = sub_pos @ centered
                den = sub_pos.sum(axis=1)
                base = base + np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return pd.Series(_clip(base), index=self.movies_, name="pred")


# A registry so the notebook can instantiate by name and compare uniformly.
MODELS = {
    "global_mean": GlobalMeanModel,
    "bias": BiasModel,
    "matrix_factorization": MatrixFactorization,
    "item_item_cf": ItemItemCF,
}


@dataclass
class ModelSpec:
    """A named model + constructor kwargs, for sweeping several at once."""

    name: str
    kwargs: dict = field(default_factory=dict)

    def build(self) -> BaseModel:
        return MODELS[self.name](**self.kwargs)
