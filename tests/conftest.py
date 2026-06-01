"""Shared pytest fixtures.

Everything is built on a small synthetic dataset so tests are fast and never
touch the network or the 877 MB ratings file.
"""

from __future__ import annotations

import pytest

from movielens import data


@pytest.fixture(scope="session")
def small_ds():
    """A tiny, deterministic synthetic dataset reused across tests."""
    return data.make_synthetic(n_users=120, n_movies=60, n_ratings=4000, seed=7)


@pytest.fixture(scope="session")
def tiny_ds():
    """An even smaller dataset for cheap unit tests."""
    return data.make_synthetic(n_users=30, n_movies=20, n_ratings=500, seed=1)
