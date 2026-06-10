# Library architecture

The project is split into two halves on purpose: a reusable library (the
engine) and notebooks (the application). This note explains the design and how
to extend it. The library lives in `src/movielens/`.

## The split

| Part | Where | Role |
|------|-------|------|
| Library | `src/movielens/` | all the real logic, tested, reusable |
| Notebooks | `notebooks/` | short calls into the library plus narrative |
| Scripts | `scripts/` | download data, build samples, build notebooks |
| Tests | `tests/` | 50 tests on fast synthetic data, 93 percent coverage |

The point of the split is that the notebook stays readable. Every cell is a few
lines that call into the library, so a reader follows the *story* of the
pipeline instead of wading through implementation details.

## The modules

Each module has a single job and can be used on its own or chained together.

| Module | Responsibility |
|--------|----------------|
| `config.py` | project paths and shared constants (rating scale, default seed) |
| `data.py` | load real CSVs or generate synthetic data, both returning a `Dataset` |
| `sampling.py` | named, swappable sampling strategies, plus save and load to disk |
| `splitting.py` | train and test splits suited to recommenders (temporal or random) |
| `eda.py` | summary tables and plots |
| `features.py` | per-user and per-movie feature engineering |
| `models.py` | the four models, all sharing one interface |
| `evaluate.py` | RMSE, MAE, precision@k, recall@k |
| `recommend.py` | top-N recommendations joined back to movie titles |
| `pipeline.py` | `run_pipeline()`, the whole flow in one call |

## The Dataset object

Everything travels through the pipeline as a small `Dataset`: a `ratings`
table, a `movies` table, and a name. It also exposes convenience numbers
(`n_ratings`, `n_users`, `n_movies`, `sparsity`) and a `summary()` dict. Because
both the real loader and the synthetic generator return the same `Dataset`
shape, nothing downstream cares where the data came from.

## The synthetic fallback

`data.load_or_synthesize()` reads the real files when they are present and
otherwise generates a realistic synthetic dataset. The synthetic generator is
not pure noise: it bakes in user biases, movie biases, latent taste and appeal
vectors, and a long-tailed popularity curve, so models trained on it behave
like models trained on real data. This is what lets the notebooks run on a
machine that never downloaded the 836 MB ratings file, and it is also what the
test suite runs on (fast and deterministic).

## Two extension points

The design goal is that you add capability without touching the notebook.

**A new model.** Write a class with `.fit(ratings)` and
`.predict(user_ids, movie_ids)`, then register it in the `MODELS` dictionary in
`models.py`:

```python
MODELS["my_model"] = MyModel
```

**A new sampling strategy.** Write a function that returns the user ids to keep
and decorate it:

```python
@register("my_strategy")
def _my_strategy(ds, size, rng, **_):
    ...
    return user_ids
```

Both new pieces are immediately available to the pipeline and the notebooks by
name. The notebook never changes; it just uses whatever the library exposes.

## Tests

The suite runs entirely on synthetic data, so it never touches the network or
the large ratings file. That keeps it fast (about two seconds) and
reproducible. Run it with:

```bash
uv run pytest
uv run pytest --cov=movielens   # with coverage
```
