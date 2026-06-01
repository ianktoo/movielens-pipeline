"""Generate notebooks/movielens_pipeline.ipynb programmatically with nbformat.

Building the notebook from code (rather than hand-editing JSON) keeps it under
version control as readable Python and guarantees valid nbformat output. Run:

    uv run python scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "movielens_pipeline.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(new_code_cell(text.strip("\n")))


# ===========================================================================
# Title
# ===========================================================================
md(r"""
# 🎬 MovieLens 32M — A Full Data-Science Pipeline

**What we're building:** a *movie recommender* — a system that learns each
person's taste from the movies they've rated and predicts what they'd enjoy
next. This is the same idea behind Netflix's "Top Picks for You".

**The dataset:** [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/)
— **32 million** ratings from ~200,000 people on ~84,000 movies, collected by
the University of Minnesota's GroupLens research lab.

**How this notebook is organised** — a classic data-science pipeline, one step
per section:

| Step | Section | Question it answers |
|------|---------|--------------------|
| 1. Load | Get the data | What do we actually have? |
| 2. Sample | Cut it down | How do we make 32M rows runnable in class? |
| 3. Explore (EDA) | Look before you leap | What does the data look like? |
| 4. Features | Engineer signals | What can we compute about users & movies? |
| 5. Split | Train vs. test | How do we test honestly? |
| 6. Model | Learn patterns | Four models, simple → smart |
| 7. Evaluate | Measure | Which model is best, and by how much? |
| 8. Recommend | The payoff | Actual movie recommendations for a person |
| 9. Swap samples | Robustness | Do conclusions hold on a different slice? |

All the real work lives in a small custom library, **`movielens`** (in `src/`),
so each notebook cell stays short and readable. Think of the notebook as the
*application* and the library as the *engine*.
""")

# ===========================================================================
# 0. Setup
# ===========================================================================
md(r"""
## 0. Setup

We import our library as `ml`. Every module (`data`, `sampling`, `models`, …) is
a small, focused file you can open and read.
""")

code(r"""
%matplotlib inline
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

import movielens as ml

pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

print("movielens library version:", ml.__version__)
print("Available models:    ", list(ml.models.MODELS))
print("Sampling strategies: ", ml.sampling.available_strategies())
""")

# ===========================================================================
# 1. Load
# ===========================================================================
md(r"""
## 1. Load the data

`load_or_synthesize()` reads the real MovieLens files if they're present in
`data/raw/`. If they're missing (say, on a classmate's laptop that hasn't
downloaded the 240 MB archive), it transparently **generates a realistic
synthetic dataset** instead — so this notebook *always* runs.

Either way we get back a `Dataset` object that bundles the `ratings` table and
the `movies` table together.
""")

code(r"""
full = ml.data.load_or_synthesize()
print(full)

# The two tables inside a Dataset:
display(full.ratings.head())
display(full.movies.head())
""")

md(r"""
A quick headline summary. The **sparsity** number is the single most important
fact about recommender data: if we laid every user against every movie in a
giant grid, what fraction of the cells would be *empty* (un-rated)? The answer
is almost always >99% — nobody watches even 1% of all movies. Learning taste
from such a sparse grid is the central challenge.
""")

code(r"""
pd.DataFrame([full.summary()]).T.rename(columns={0: "value"})
""")

# ===========================================================================
# 2. Sample
# ===========================================================================
md(r"""
## 2. Sampling — making 32 million rows manageable

32M rows is too much to model live in a classroom. So we **sample**. But *how*
we sample matters enormously:

- ❌ **Naive row sampling** (pick 100k random ratings) shatters every user's
  history into a few disconnected ratings — useless for learning taste.
- ✅ **Sample by *user***: keep a subset of users but *all* of their ratings.
  Each kept user brings their whole history, so the data stays coherent.

Our library offers several **named, swappable strategies**:

| Strategy | What it keeps | Good for |
|----------|---------------|----------|
| `random_users`  | a random subset of users | an unbiased baseline slice |
| `active_users`  | the users who rate the most | lots of signal per user — **our main slice** |
| `dense`         | active users **×** popular movies | a *dense* slice where item-item CF shines (see §9) |

We pre-built these to `data/samples/` with `scripts/prepare_samples.py`, so we
can load one instantly instead of re-reading 877 MB every time.

We'll work with **`active_users`**: the 3,000 most prolific raters, with their
*entire* histories. These users give us the richest signal to learn from.
""")

code(r"""
saved = ml.sampling.list_saved_samples()
print("Pre-built samples on disk:", saved)

if "active_users" in saved:
    sample = ml.sampling.load_sample("active_users")   # instant load
else:
    sample = ml.sampling.build_sample(full, strategy="active_users", size=3000)

print(sample)
print(f"sparsity: {sample.sparsity:.1%} empty  (the full set is {full.sparsity:.2%} empty)")
""")

md(r"""
Even after focusing on the most active users, the grid is still **~98% empty** —
3,000 people simply can't watch a meaningful fraction of 77,000 movies. That
sparsity is the realistic challenge our models have to handle.

Let's compare the three sample strategies side by side. Notice that `dense`
deliberately trades catalogue size for a much *fuller* grid — that's what makes
item-item collaborative filtering practical on it later in §9.
""")

code(r"""
rows = []
for key in ["random_users", "active_users", "dense"]:
    if key in saved:
        s = ml.sampling.load_sample(key)
        rows.append({**s.summary()})
pd.DataFrame(rows)
""")

# ===========================================================================
# 3. EDA
# ===========================================================================
md(r"""
## 3. Exploratory Data Analysis (EDA)

Before modelling, *look at the data*. Plotting functions live in `ml.eda` and
each returns a chart we can display. We're working with the `active_users` sample.

### 3.1 How do people rate?
Most ratings cluster at 3–4 stars, and people prefer whole stars to half stars.
Knowing the *average* rating already gives us our dumbest possible model
("just predict the average") — the baseline everything else must beat.
""")

code(r"""
ml.eda.plot_rating_distribution(sample)
plt.show()
""")

md(r"""
### 3.2 The long tail of popularity
A handful of blockbusters collect a huge share of all ratings, while most movies
are rated rarely. On a log-log scale this shows up as a near-straight line —
the famous **long tail**. It's *why* recommending obscure movies (the
"cold-start" problem) is hard: there's barely any data on them.
""")

code(r"""
ml.eda.plot_activity_long_tail(sample)
plt.show()
""")

md(r"""
### 3.3 Genres and top movies
""")

code(r"""
ml.eda.plot_genre_counts(sample, top=15)
plt.show()
""")

code(r"""
# Highest-rated movies (with a minimum number of ratings so a single
# 5-star vote can't top the chart):
ml.eda.top_movies(sample, n=10, min_ratings=50)
""")

code(r"""
# Activity stats: how many ratings per user / per movie?
ml.eda.activity_table(sample)
""")

# ===========================================================================
# 4. Features
# ===========================================================================
md(r"""
## 4. Feature engineering

Collaborative-filtering models learn straight from the (user, movie, rating)
triples, so they don't strictly *need* engineered features. But a real pipeline
builds them — they're useful for inspection, for content-based ideas, and for
understanding *who* and *what* we're modelling.

`ml.features` turns the raw tables into per-user and per-movie feature tables,
including one-hot genre flags and the release year parsed out of the title.
""")

code(r"""
user_feats = ml.features.user_features(sample)
movie_feats = ml.features.movie_features(sample)

print("User features:")
display(user_feats.head())
print("Movie features (popularity, avg rating, + one-hot genres):")
display(movie_feats.head())
""")

# ===========================================================================
# 5. Split
# ===========================================================================
md(r"""
## 5. Train / test split — testing honestly

To know if a model is any good, we hide some ratings, train on the rest, then
check whether the model would have predicted the hidden ones. Two strategies:

- **Random split** — hide a random 20% of each user's ratings. Simple, but it
  lets the model peek at "future" ratings to predict the "past".
- **Temporal split** — for each user, hide their *most recent* 20%. This mirrors
  reality: *train on the past, predict the future.* It's the fairer test for a
  recommender, so it's our default.

We always keep every test user in the training set too — you can't fairly test a
recommender on someone it has never seen.
""")

code(r"""
split = ml.splitting.split(sample.ratings, strategy="temporal", test_frac=0.2)
print(split)
print(f"  train: {split.n_train:,} ratings")
print(f"  test:  {split.n_test:,} ratings (the most recent 20% per user)")
""")

# ===========================================================================
# 6. Models
# ===========================================================================
md(r"""
## 6. The models — from naive to smart

We climb a ladder of four models, each adding one idea:

1. **Global mean** — predict the overall average rating for *everything*.
   The humblest baseline.
2. **Bias model** — add a per-user offset (some people are generous, some
   harsh) and a per-movie offset (some films are just better). Prediction =
   `μ + user_bias + movie_bias`. Surprisingly strong — the baseline to beat.
3. **Matrix factorization** — discover hidden "taste dimensions" (action-y?
   romantic? quirky?) for every user *and* movie, then match them up. This is
   the heart of modern recommenders.
4. **Item-item collaborative filtering** — "people who liked *this* also liked
   *that*": predict from the movies most similar to ones you already rated.

Each model exposes the same `.fit()` / `.predict()` interface, so they're fully
interchangeable.

> **A real engineering constraint:** item-item CF builds an
> *n_movies × n_movies* similarity matrix. On `active_users` that's
> 77,000² ≈ 6 billion cells — far too big. So we **skip it automatically** on
> large catalogues and showcase it on the compact `dense` sample in §9. This is
> a genuine lesson: the "best" algorithm on paper isn't always the one that fits
> in memory.
""")

code(r"""
# Build the line-up, skipping item-item CF when the catalogue is too large.
to_train = [
    ("global_mean",          ml.models.GlobalMeanModel()),
    ("bias",                 ml.models.BiasModel(reg=10.0)),
    ("matrix_factorization", ml.models.MatrixFactorization(n_factors=20, reg=10.0)),
]
if sample.n_movies <= 5000:
    to_train.append(("item_item_cf", ml.models.ItemItemCF(k_neighbors=30)))
else:
    print(f"(Skipping item_item_cf — {sample.n_movies:,} movies is too many for a "
          f"similarity matrix. We'll demo it on the 'dense' sample in §9.)\n")

models = {}
for name, model in to_train:
    model.fit(split.train)
    models[name] = model
    print(f"fitted {name}")
""")

# ===========================================================================
# 7. Evaluate
# ===========================================================================
md(r"""
## 7. Evaluation — which model wins?

Two kinds of metric, because a recommender has two jobs:

- **Accuracy** — when we predict a rating, how close are we?
  - **RMSE** (root mean squared error): typical miss in stars, punishing big
    misses extra. *Lower is better.*
  - **MAE** (mean absolute error): average miss in stars. *Lower is better.*
- **Ranking** — of the movies we put at the *top*, how many are genuinely good?
  - **Precision@10**: of our top 10 picks, what fraction the user actually liked.
  - **Recall@10**: of everything the user liked, how much we surfaced in the top 10.
  - *Higher is better* for both.
""")

code(r"""
results = []
for name, model in models.items():
    scores = ml.evaluate.evaluate_model(model, split.test, k=10)
    results.append({"model": name, **scores})

leaderboard = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
leaderboard
""")

md(r"""
The story should be clear: **each smarter model beats the previous one.** Let's
visualise the RMSE drop — shorter bars are better.
""")

code(r"""
order = leaderboard.sort_values("rmse", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(order["model"], order["rmse"], color="#4C72B0")
ax.set_xlabel("RMSE  (lower = better)")
ax.set_title("Rating-prediction accuracy by model")
for i, v in enumerate(order["rmse"]):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center")
plt.tight_layout(); plt.show()
""")

# ===========================================================================
# 8. Recommend
# ===========================================================================
md(r"""
## 8. The payoff — real recommendations for a real person

Numbers are nice, but the *point* of all this is to recommend movies. Let's pick
a user, look at what they already love, and see what our best model suggests
they watch next (excluding films they've already seen).

Put "what they liked" next to "what we recommend" and you can sanity-check the
model with your own eyes — do the recommendations *feel* right?
""")

code(r"""
best_name = leaderboard.iloc[0]["model"]
best_model = models[best_name]
print(f"Using our best model: {best_name}\n")

# Pick an active user from the sample so they have a rich history to show.
example_user = int(sample.ratings["userId"].value_counts().index[0])
print(f"Example user: {example_user}")

print("\n⭐ Movies THIS USER already rated highest:")
display(ml.recommend.user_history(sample, example_user, n=8))
""")

code(r"""
print(f"🍿 Top 10 recommendations for user {example_user} (movies they haven't seen):")
ml.recommend.recommend_for_user(best_model, sample, example_user, n=10)
""")

# ===========================================================================
# 9. Swap samples
# ===========================================================================
md(r"""
## 9. Swap the sample set — do our conclusions hold?

A good experiment is reproducible on a *different* slice of the data. Because
our samples are swappable and the whole pipeline is one function call, we can
re-run everything on another sample in a single line and check that the model
ranking is stable.

`run_pipeline()` does load → (sample) → split → train all models → evaluate, and
returns a tidy leaderboard.
""")

code(r"""
for key in ["dense", "random_users"]:
    if key not in ml.sampling.list_saved_samples():
        continue
    ds = ml.sampling.load_sample(key)
    # 'random_users' has thousands of movies; item-item CF is meant for dense
    # data, so we let run_pipeline use the default model line-up and simply skip
    # item_item_cf when the catalogue is too big.
    specs = list(ml.pipeline.DEFAULT_MODELS)
    if ds.n_movies > 5000:
        specs = [s for s in specs if s.name != "item_item_cf"]
    print(f"\n===== sample: {key}  ({ds.n_ratings:,} ratings, {ds.n_movies:,} movies) =====")
    res = ml.pipeline.run_pipeline(ds, sample_strategy=None, model_specs=specs, verbose=False)
    display(res.scores[["model", "rmse", "mae", "precision@10", "recall@10"]])
""")

# ===========================================================================
# 10. Wrap up
# ===========================================================================
md(r"""
## 10. Wrap-up

**What we did, end to end:**

1. **Loaded** 32M real MovieLens ratings (with an automatic synthetic fallback).
2. **Sampled** smartly — by user, into dense/swappable slices — to stay runnable.
3. **Explored** the data: rating distribution, the long tail, genres.
4. **Engineered** per-user and per-movie features.
5. **Split** train/test the honest way (temporal: predict the future).
6. **Trained** four models, naive → smart.
7. **Evaluated** them on accuracy *and* ranking — each model beat the last.
8. **Recommended** actual movies for a real user.
9. **Swapped** samples to confirm the conclusions are robust.

**Why the library matters:** every step above was a one-liner because the logic
lives in the modular `movielens` package. Want to add a deep-learning model? Add
a class to `models.py` and register it. Want a new sampling idea? Add a
`@register`-decorated function to `sampling.py`. The notebook never changes — it
just *uses* the engine. That's what "modular and extensible" buys you.

**Ideas to extend:** SVD++ or neural collaborative filtering, content-based
recommendations using the genre features, hyper-parameter tuning of
`n_factors`, or A/B-comparing temporal vs. random splits.
""")


def main() -> None:
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"Wrote {OUT}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
