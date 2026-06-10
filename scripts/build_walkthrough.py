"""Generate notebooks/movielens_walkthrough.ipynb programmatically with nbformat.

This is the explained, teaching version of the pipeline. It walks through every
step with narrative and comparison tables, while keeping each code cell short by
calling into the movielens library. It contains no emojis and no long dashes, so
it reads cleanly when presented live in class.

The minimal, run-only counterpart is built by scripts/build_minimal.py and does
the exact same end-to-end run with almost no prose.

Run:

    uv run python scripts/build_walkthrough.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "movielens_walkthrough.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(new_code_cell(text.strip("\n")))


# ===========================================================================
# Title
# ===========================================================================
md(r"""
# MovieLens 32M: a full data-science pipeline (walkthrough)

**The question.** Given the movies a person has already rated, can we predict
what they would enjoy next? This is the idea behind Netflix "Top Picks for You".

**The data.** [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/),
which is 32 million ratings by about 200,000 people on about 87,000 movies,
collected by the GroupLens lab at the University of Minnesota.

**How to read this notebook.** It is a classic data-science pipeline, one step
per section. The real work lives in a small library called `movielens` (in
`src/`), so each cell stays short. Think of the notebook as the application and
the library as the engine.

| Step | Section | Question it answers |
|------|---------|---------------------|
| 1. Load | Get the data | What do we actually have? |
| 2. Sample | Cut it down | How do we make 32M rows runnable? |
| 3. Explore | Look first | What does the data look like? |
| 4. Features | Engineer signals | What can we compute about users and movies? |
| 5. Split | Train vs test | How do we test honestly? |
| 6. Model | Learn patterns | Four models, simple to smart |
| 7. Evaluate | Measure | Which model is best, and by how much? |
| 8. Recommend | The payoff | Actual recommendations for a person |
| 9. Swap samples | Robustness | Do the conclusions hold on another slice? |
""")

# ===========================================================================
# 0. Setup
# ===========================================================================
md(r"""
## 0. Setup

We import our library as `ml`. Every module (`data`, `sampling`, `models`, and
so on) is a small, focused file you can open and read.
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

`load_or_synthesize()` reads the real MovieLens files if they are present in
`data/raw/`. If they are missing, it generates a realistic synthetic dataset
instead, so this notebook always runs. Either way we get back a `Dataset` that
bundles the `ratings` table and the `movies` table together.
""")

code(r"""
full = ml.data.load_or_synthesize()
print(full)

# The two tables inside a Dataset:
display(full.ratings.head())
display(full.movies.head())
""")

md(r"""
The **sparsity** number is the single most important fact about recommender
data. If we laid every user against every movie in a grid, what fraction of the
cells would be empty? The answer is almost always above 99 percent. Nobody
watches even 1 percent of all movies, and learning taste from such an empty grid
is the central challenge.
""")

code(r"""
pd.DataFrame([full.summary()]).T.rename(columns={0: "value"})
""")

# ===========================================================================
# 2. Sample
# ===========================================================================
md(r"""
## 2. Sampling: making 32 million rows manageable

We cannot model 32M rows live, so we sample. How we sample matters a great deal.

| Approach | What it does | Result |
|----------|--------------|--------|
| Random rows | pick N ratings at random | breaks each user's history into a few disconnected ratings, useless |
| Random users | pick N users, keep all their ratings | each user brings a full history, so the data stays coherent |

We always sample by user. Our library offers three named, swappable strategies.

| Strategy | What it keeps | Good for |
|----------|---------------|----------|
| `random_users` | a random subset of users | an unbiased baseline slice |
| `active_users` | the users who rate the most | the most signal per user, our main slice |
| `dense` | active users restricted to popular movies | a fuller grid where item-item CF works (see section 9) |

We pre-built these to `data/samples/` with `scripts/prepare_samples.py`, so we
load one instantly instead of re-reading the full file. We will work with
`active_users`: the 3,000 most prolific raters, with their entire histories.
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
Even after focusing on the most active users, the grid is still about 98 percent
empty. Three thousand people simply cannot watch a meaningful fraction of 77,000
movies. Let us compare the three strategies side by side. Notice that `dense`
deliberately trades catalogue size for a much fuller grid, which is what makes
item-item collaborative filtering practical on it later.
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
## 3. Exploratory data analysis

Before modelling, look at the data. Plotting functions live in `ml.eda` and each
returns a chart we can display. We are working with the `active_users` sample.

### 3.1 How do people rate?

Most ratings cluster at 3 to 4 stars, and people prefer whole stars to half
stars. Knowing the average rating already gives us our dumbest possible model
(just predict the average), the baseline everything else must beat.
""")

code(r"""
ml.eda.plot_rating_distribution(sample)
plt.show()
""")

md(r"""
### 3.2 The long tail of popularity

A handful of blockbusters collect a huge share of all ratings, while most movies
are rated rarely. On a log-log scale this shows up as a near-straight line, the
famous long tail. It is why recommending obscure movies (the cold-start problem)
is hard: there is barely any data on them.
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
# Highest-rated movies, with a minimum number of ratings so a single 5-star
# vote cannot top the chart:
ml.eda.top_movies(sample, n=10, min_ratings=50)
""")

code(r"""
# Activity stats: how many ratings per user and per movie?
ml.eda.activity_table(sample)
""")

# ===========================================================================
# 4. Features
# ===========================================================================
md(r"""
## 4. Feature engineering

Collaborative-filtering models learn straight from the (user, movie, rating)
triples, so they do not strictly need engineered features. But a real pipeline
builds them, and they are useful for inspection and for content-based ideas.
`ml.features` turns the raw tables into per-user and per-movie feature tables,
including one-hot genre flags and the release year parsed from the title.
""")

code(r"""
user_feats = ml.features.user_features(sample)
movie_feats = ml.features.movie_features(sample)

print("User features:")
display(user_feats.head())
print("Movie features (popularity, avg rating, plus one-hot genres):")
display(movie_feats.head())
""")

# ===========================================================================
# 5. Split
# ===========================================================================
md(r"""
## 5. Train and test split: testing honestly

To know if a model is any good, we hide some ratings, train on the rest, then
check whether the model would have predicted the hidden ones.

| Strategy | What it hides | Pros | Cons |
|----------|---------------|------|------|
| Random | a random 20% per user | simple | lets the model peek at the future |
| Temporal | the most recent 20% per user | matches reality, predict the future | needs timestamps |

We default to the temporal split. We always keep every test user in the training
set too, because you cannot fairly test a recommender on someone it has never
seen.
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
## 6. The models: from naive to smart

We climb a ladder of four models, each adding exactly one idea.

| Rung | Model | Idea | Strength | Limitation |
|------|-------|------|----------|------------|
| 1 | Global mean | guess the overall average | a true floor | ignores everyone |
| 2 | Bias | mu plus user and movie offsets | simple and strong | no personal taste |
| 3 | Matrix factorization | learn hidden taste and appeal vectors | personalised, modern core | needs enough ratings |
| 4 | Item-item CF | "liked this, so liked that" | strong on dense data | does not scale to big catalogues |

Each model exposes the same `.fit()` and `.predict()` interface, so they are
fully interchangeable.

A real engineering constraint: item-item CF builds an n_movies by n_movies
similarity matrix. On `active_users` that is about 77,000 squared, roughly 6
billion cells, far too big. So we skip it automatically on large catalogues and
showcase it on the compact `dense` sample in section 9. The best algorithm on
paper is not always the one that fits in memory.
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
    print(f"(Skipping item_item_cf: {sample.n_movies:,} movies is too many for a "
          f"similarity matrix. We demo it on the 'dense' sample in section 9.)\n")

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
## 7. Evaluation: which model wins?

Two kinds of metric, because a recommender has two jobs.

| Job | Question | Metric | Better |
|-----|----------|--------|--------|
| Accuracy | how close is a predicted rating? | RMSE, MAE | lower |
| Ranking | are the top picks actually good? | Precision@10, Recall@10 | higher |

RMSE punishes big misses extra. MAE is the plain average miss in stars.
Precision@10 is the fraction of our top ten that the user liked. Recall@10 is
the fraction of everything they liked that we surfaced in the top ten.
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
The story should be clear: each smarter model beats the previous one. Let us
visualise the RMSE drop. Shorter bars are better.
""")

code(r"""
order = leaderboard.sort_values("rmse", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(order["model"], order["rmse"], color="#4C72B0")
ax.set_xlabel("RMSE  (lower is better)")
ax.set_title("Rating-prediction accuracy by model")
for i, v in enumerate(order["rmse"]):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center")
plt.tight_layout(); plt.show()
""")

# ===========================================================================
# 8. Recommend
# ===========================================================================
md(r"""
## 8. The payoff: real recommendations for a real person

Numbers are nice, but the point is to recommend movies. We pick a user, look at
what they already love, and see what our best model suggests they watch next,
excluding films they have already seen. Putting "what they liked" next to "what
we recommend" lets us sanity-check the model by eye.
""")

code(r"""
best_name = leaderboard.iloc[0]["model"]
best_model = models[best_name]
print(f"Using our best model: {best_name}\n")

# Pick the most active user in the sample so they have a rich history to show.
example_user = int(sample.ratings["userId"].value_counts().index[0])
print(f"Example user: {example_user}")

print("\nMovies this user already rated highest:")
display(ml.recommend.user_history(sample, example_user, n=8))
""")

code(r"""
print(f"Top 10 recommendations for user {example_user} (movies they have not seen):")
ml.recommend.recommend_for_user(best_model, sample, example_user, n=10)
""")

# ===========================================================================
# 9. Swap samples
# ===========================================================================
md(r"""
## 9. Swap the sample set: do our conclusions hold?

A good experiment is reproducible on a different slice of the data. Because our
samples are swappable and the whole pipeline is one function call, we can re-run
everything on another sample in a single line and check that the ranking is
stable. `run_pipeline()` does load, then sample, then split, then train all
models, then evaluate, and returns a tidy leaderboard.

One caution when reading the output: recall@10 is not comparable across samples.
With far fewer movies in the `dense` slice, ten picks cover much more of what a
user liked, so recall looks much higher there. Compare models within a sample,
not across samples.
""")

code(r"""
for key in ["dense", "random_users"]:
    if key not in ml.sampling.list_saved_samples():
        continue
    ds = ml.sampling.load_sample(key)
    # 'random_users' has thousands of movies; item-item CF is meant for dense
    # data, so we drop it when the catalogue is too big.
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

What we did, end to end:

1. Loaded 32M real MovieLens ratings (with an automatic synthetic fallback).
2. Sampled smartly, by user, into swappable slices, to stay runnable.
3. Explored the data: rating distribution, the long tail, genres.
4. Engineered per-user and per-movie features.
5. Split train and test the honest way (temporal: predict the future).
6. Trained four models, naive to smart.
7. Evaluated them on accuracy and ranking. Each model beat the last.
8. Recommended actual movies for a real user.
9. Swapped samples to confirm the conclusions are robust.

Every step was a one-liner because the logic lives in the modular `movielens`
package. To add a deep-learning model, add a class to `models.py` and register
it. To add a new sampling idea, add a registered function to `sampling.py`. The
notebook never changes; it just uses the engine.
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
