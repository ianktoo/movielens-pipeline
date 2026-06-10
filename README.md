# MovieLens 32M: a movie recommender, built end to end

This is a class project. The question we set ourselves is simple to state and surprisingly deep to answer: **given the movies a person has already rated, can we predict what they would enjoy next?** That is the same question behind Netflix "Top Picks", Spotify "Discover Weekly", and Amazon "Customers also
bought".

We wanted to build the whole thing ourselves, not just call a library function, So we could see where the difficulty actually lives. It turns out most of it is not in the model. It is in the data: 32 million ratings, a grid that is more than 99.99 percent empty, and the discipline needed to test a recommender
honestly.

## Source data

All of the data comes from the **MovieLens 32M** dataset published by the GroupLens lab at the University of Minnesota.

- <https://grouplens.org/datasets/movielens/32m/>
- 32,000,204 ratings by 200,948 users on 87,585 movies, collected October 2023.
- The data is GroupLens property and subject to
  [their terms of use](https://grouplens.org/datasets/movielens/). We do not
  commit it; everyone regenerates it locally.

More detail on the files, the counts, and why the data is hard lives in
[docs/data-and-sampling.md](docs/data-and-sampling.md).

## How the project is organised

We split the work into an engine and an application, on purpose.

| Part | Where | What it is |
|------|-------|------------|
| Library (the engine) | `src/movielens/` | ten small, tested modules that do the real work |
| Notebooks (the application) | `notebooks/` | short calls into the library, with narrative |
| Scripts | `scripts/` | download data, build samples, build notebooks |
| Tests | `tests/` | 50 tests on synthetic data, 93 percent coverage |

There are two notebooks that do the same end-to-end run, for two audiences:

| Notebook | For | Style |
|----------|-----|-------|
| `notebooks/movielens_walkthrough.ipynb` | learning the pipeline | explained step by step |
| `notebooks/movielens_minimal.ipynb` | just running it | code only, minimal prose |

The full design and the two extension points (adding a model, adding a sampling
strategy) are written up in
[docs/library-architecture.md](docs/library-architecture.md).

## Quick start

### Option A: with uv (recommended)

```bash
uv sync --all-extras                      # create the env and install everything
uv run python scripts/download_data.py    # download, unzip, build the samples
uv run jupyter lab notebooks/movielens_walkthrough.ipynb
```

If you skip the download step, the pipeline falls back to realistic synthetic
data automatically, so the notebooks still run.

### Option B: with plain pip

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
jupyter lab notebooks/movielens_walkthrough.ipynb
```

## What we found

We never model all 32 million rows at once. We sample by user (keeping each kept
user's whole history) into three slices, then train four models on each. The
four models form a ladder, where each rung adds one idea.

| Model | The one idea it adds |
|-------|----------------------|
| `global_mean` | predict the overall average rating |
| `bias` | add a per-user and per-movie offset |
| `matrix_factorization` | learn hidden taste and appeal vectors |
| `item_item_cf` | "people who liked this also liked that" |

The headline result is that **each rung beats the one below it.** These are the
verified numbers from a temporal split (train on the past, predict the future).

On the `active_users` slice (5.19M ratings, 3,000 users, 77,334 movies):

| Model | RMSE | MAE |
|-------|------|-----|
| global_mean | 1.018 | 0.799 |
| bias | 0.826 | 0.623 |
| matrix_factorization | 0.800 | 0.601 |

Item-item collaborative filtering is left out of that slice on purpose: with
77,334 movies its similarity matrix would need about 6 billion cells. We run it
instead on the `dense` slice (301,416 ratings, 3,000 users, 800 movies), where
it takes the top spot:

| Model | RMSE | MAE |
|-------|------|-----|
| item_item_cf | 0.820 | 0.612 |
| matrix_factorization | 0.837 | 0.630 |
| bias | 0.860 | 0.650 |
| global_mean | 0.998 | 0.780 |

Why the models behave the way they do, and why ranking metrics like recall
should not be compared across slices, is covered in
[docs/models.md](docs/models.md) and [docs/evaluation.md](docs/evaluation.md).

## Minimal code example

```python
import movielens as ml

ds     = ml.data.load_or_synthesize()
sample = ml.sampling.build_sample(ds, strategy="dense", size=3000)
result = ml.pipeline.run_pipeline(sample, sample_strategy=None)
print(result.scores)   # the leaderboard of all models
```

## Tests

```bash
uv run pytest                  # 50 tests, all on fast synthetic data
uv run pytest --cov=movielens  # with coverage (93 percent)
```

The tests never touch the network or the 836 MB ratings file. They run on small,
deterministic synthetic datasets, so they are fast and reproducible.

## Further reading

| Document | What it covers |
|----------|----------------|
| [docs/data-and-sampling.md](docs/data-and-sampling.md) | the dataset, sparsity, and the three sampling strategies |
| [docs/models.md](docs/models.md) | the four models, their intuition, and trade-offs |
| [docs/evaluation.md](docs/evaluation.md) | splitting, metrics, and how to read the results |
| [docs/library-architecture.md](docs/library-architecture.md) | module layout and how to extend the library |
| [PRESENTATION.md](PRESENTATION.md) | the slides we present in class |

## Project layout

```text
.
├── src/movielens/   the library (the engine)
├── notebooks/       walkthrough and minimal notebooks (the application)
├── scripts/         download_data, prepare_samples, build_notebook scripts
├── docs/            deeper-dive notes referenced from this README
├── tests/           pytest suite
├── data/            raw and samples (git-ignored)
├── pyproject.toml   uv project and dependencies
└── requirements.txt pinned deps for pip users
```
