"""movielens — a modular, extensible recommender pipeline for MovieLens 32M.

The pipeline is built from small, single-purpose modules you can use on their
own or chain together:

    data      -- load real data or generate synthetic data (Dataset)
    sampling  -- cut the 32M ratings into manageable, swappable samples
    splitting -- train/test splits suited to recommenders (temporal/random)
    eda       -- summary tables and plots
    features  -- feature engineering helpers
    models    -- GlobalMean, Bias, MatrixFactorization, ItemItemCF
    evaluate  -- RMSE/MAE + precision/recall@k
    recommend -- top-N recommendations with titles
    pipeline  -- run_pipeline(): the whole thing in one call

Typical notebook usage:

    import movielens as ml
    ds = ml.data.load_or_synthesize()
    sample = ml.sampling.build_sample(ds, strategy="active_users", size=2000)
    result = ml.pipeline.run_pipeline(sample, sample_strategy=None)
    result.scores
"""

from __future__ import annotations

from . import (
    config,
    data,
    eda,
    evaluate,
    features,
    models,
    pipeline,
    recommend,
    sampling,
    splitting,
)
from .data import Dataset, load_or_synthesize, load_raw, make_synthetic
from .pipeline import PipelineResult, run_pipeline

__version__ = "0.1.0"

__all__ = [
    "config",
    "data",
    "eda",
    "evaluate",
    "features",
    "models",
    "pipeline",
    "recommend",
    "sampling",
    "splitting",
    "Dataset",
    "load_or_synthesize",
    "load_raw",
    "make_synthetic",
    "PipelineResult",
    "run_pipeline",
]
