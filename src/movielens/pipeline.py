"""The end-to-end orchestrator.

This is the one-call convenience layer that strings the whole pipeline
together — load -> sample -> split -> train several models -> evaluate — and
hands back a tidy results table. The notebook uses the individual modules for
the *teaching* walkthrough, then calls :func:`run_pipeline` to show "here's the
whole thing in one function".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config, data, sampling, splitting
from .data import Dataset
from .evaluate import evaluate_model
from .models import MODELS, BaseModel, ModelSpec
from .splitting import Split


# Sensible default line-up: each model is a rung on the ladder from naive to real.
DEFAULT_MODELS = [
    ModelSpec("global_mean"),
    ModelSpec("bias", {"reg": 10.0}),
    ModelSpec("matrix_factorization", {"n_factors": 20, "reg": 10.0}),
    ModelSpec("item_item_cf", {"k_neighbors": 30}),
]


@dataclass
class PipelineResult:
    """Everything the pipeline produced, kept together for inspection."""

    dataset: Dataset
    split: Split
    models: dict[str, BaseModel]
    scores: pd.DataFrame
    config: dict = field(default_factory=dict)


def run_pipeline(
    dataset: Dataset | None = None,
    sample_strategy: str | None = "active_users",
    sample_size: int = 2000,
    split_strategy: str = "temporal",
    test_frac: float = 0.2,
    model_specs: list[ModelSpec] | None = None,
    k: int = 10,
    seed: int = config.DEFAULT_SEED,
    verbose: bool = True,
) -> PipelineResult:
    """Run the full pipeline and return a :class:`PipelineResult`.

    Parameters
    ----------
    dataset:
        Starting data. If ``None``, loads real data when available else
        synthesizes (via :func:`movielens.data.load_or_synthesize`).
    sample_strategy:
        Name of a sampling strategy, or ``None`` to skip sampling.
    sample_size:
        Number of users to keep when sampling.
    split_strategy:
        ``"temporal"`` (realistic) or ``"random"``.
    model_specs:
        Which models to train; defaults to :data:`DEFAULT_MODELS`.
    k:
        Cutoff for ranking metrics.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # 1. Load --------------------------------------------------------------
    if dataset is None:
        dataset = data.load_or_synthesize()
    log(f"Loaded {dataset!r}")

    # 2. Sample ------------------------------------------------------------
    if sample_strategy is not None:
        dataset = sampling.build_sample(
            dataset, strategy=sample_strategy, size=sample_size, seed=seed
        )
        log(f"Sampled -> {dataset!r}")

    # 3. Split -------------------------------------------------------------
    sp = splitting.split(
        dataset.ratings, strategy=split_strategy, test_frac=test_frac, seed=seed
    )
    log(f"Split -> {sp!r}")

    # 4. Train + 5. Evaluate ----------------------------------------------
    model_specs = model_specs or DEFAULT_MODELS
    fitted: dict[str, BaseModel] = {}
    rows = []
    for spec in model_specs:
        if spec.name not in MODELS:
            raise KeyError(f"Unknown model {spec.name!r}; known: {list(MODELS)}")
        model = spec.build().fit(sp.train)
        fitted[spec.name] = model
        scores = evaluate_model(model, sp.test, k=k)
        scores = {"model": spec.name, **scores}
        rows.append(scores)
        log(f"  {spec.name:>22}: RMSE={scores['rmse']:.4f}  MAE={scores['mae']:.4f}")

    scores_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)

    return PipelineResult(
        dataset=dataset,
        split=sp,
        models=fitted,
        scores=scores_df,
        config={
            "sample_strategy": sample_strategy,
            "sample_size": sample_size,
            "split_strategy": split_strategy,
            "test_frac": test_frac,
            "k": k,
            "seed": seed,
        },
    )
