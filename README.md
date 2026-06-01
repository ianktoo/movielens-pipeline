# 🎬 MovieLens 32M — A Full Data-Science Pipeline

A class project that builds a **movie recommender** end-to-end on the
[MovieLens 32M dataset](https://grouplens.org/datasets/movielens/32m/)
(32 million ratings, ~200K users, ~84K movies).

The work is split into two parts:

- **`src/movielens/`** — a small, modular, reusable Python **library** (the engine).
- **`notebooks/movielens_pipeline.ipynb`** — the **application** we present in
  class. Each cell is a short call into the library plus a plain-English
  explanation.

---

## Quick start

### Option A — with [uv](https://docs.astral.sh/uv/) (recommended)

```bash
uv sync --all-extras                       # create the env + install everything
```

Get the data — **one command** downloads (~240 MB), unzips, and builds the samples:

```bash
uv run python scripts/download_data.py
```

> Don't want the real data? Skip the command — the pipeline falls back to
> realistic **synthetic** data automatically, so the notebook still runs.

Run the notebook:

```bash
uv run jupyter lab notebooks/movielens_pipeline.ipynb
```

### Option B — with plain pip (for teammates without uv)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt # installs deps + the movielens package (-e .)
python scripts/download_data.py # download + unzip + build samples
jupyter lab notebooks/movielens_pipeline.ipynb
```

> **No dataset? No problem.** If `data/raw/ml-32m/` is missing, every entry
> point (`load_or_synthesize`, `prepare_samples.py`) falls back to a realistic
> **synthetic** dataset, so the notebook always runs.

---

## Getting the real data

**Dataset:** MovieLens 32M — <https://grouplens.org/datasets/movielens/32m/>
(32M ratings · ~200K users · ~84K movies · ~240 MB zipped).

**Easiest — let the script do it (cross-platform, no curl/wget needed):**

```bash
uv run python scripts/download_data.py            # download + unzip + build samples
uv run python scripts/download_data.py --force    # re-download from scratch
uv run python scripts/download_data.py --skip-samples   # just fetch the data
```

**Manual alternative:**

1. Download `ml-32m.zip` from <https://grouplens.org/datasets/movielens/32m/>.
2. Unzip into `data/raw/` so files live at `data/raw/ml-32m/ratings.csv`, etc.
3. Run `uv run python scripts/prepare_samples.py` to build the sample sets.

The raw data and generated samples are **git-ignored** (too big to commit) —
each teammate regenerates them locally. The dataset is © GroupLens and subject to
[its own terms of use](https://grouplens.org/datasets/movielens/).

---

## The library

```text
src/movielens/
  config.py     paths + constants (rating scale, default seed)
  data.py       load real data OR generate synthetic data  -> Dataset
  sampling.py   smart, swappable sampling strategies (by user, dense, ...)
  splitting.py  train/test splits for recommenders (temporal / random)
  eda.py        summary tables + plots
  features.py   per-user / per-movie feature engineering
  models.py     GlobalMean, Bias, MatrixFactorization, ItemItemCF
  evaluate.py   RMSE / MAE + precision@k / recall@k
  recommend.py  top-N recommendations joined to movie titles
  pipeline.py   run_pipeline(): the whole thing in one call
```

Minimal example:

```python
import movielens as ml

ds      = ml.data.load_or_synthesize()
sample  = ml.sampling.build_sample(ds, strategy="dense", size=3000)
result  = ml.pipeline.run_pipeline(sample, sample_strategy=None)
print(result.scores)                       # leaderboard of all models
```

### Extending it

- **New model?** Add a class to `models.py` with `.fit()`/`.predict()` and
  register it in the `MODELS` dict.
- **New sampling idea?** Add a `@register("my_strategy")`-decorated function to
  `sampling.py`.

The notebook never changes — it just *uses* whatever the library exposes.

---

## The four models (the "ladder")

| Model | Idea | Typical RMSE* |
|-------|------|---------------|
| `global_mean` | predict the overall average rating | ~1.00 |
| `bias` | `μ + user_bias + movie_bias` | ~0.86 |
| `matrix_factorization` | learn latent taste/appeal vectors (SVD) | ~0.84 |
| `item_item_cf` | "users who liked X also liked Y" | ~0.82 |

\* on the `dense` sample, temporal split. **Lower RMSE is better**; each model
beats the previous one — the story we tell in class.

---

## Tests

```bash
uv run pytest                 # 45 tests, all on fast synthetic data
uv run pytest --cov=movielens # with coverage
```

Tests never touch the network or the 877 MB ratings file — they run on small,
deterministic synthetic datasets, so they're fast and reproducible.

---

## Project layout

```text
.
├── src/movielens/        the library
├── notebooks/            movielens_pipeline.ipynb  (the presentation)
├── scripts/              download_data.py, prepare_samples.py, build_notebook.py
├── tests/                pytest suite
├── data/                 raw/ + samples/  (git-ignored)
├── pyproject.toml        uv project + deps
└── requirements.txt      pinned deps for pip users
```
