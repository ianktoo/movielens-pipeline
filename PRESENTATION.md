---
marp: true
title: MovieLens 32M — A Full Data-Science Pipeline
author: ianktoo
paginate: true
theme: default
---

# 🎬 Predicting What You'll Watch Next
## A full data-science pipeline on MovieLens 32M

A movie **recommender** built end-to-end — from 32 million raw ratings to
personalised "Top 10 for you" lists.

*Notebook = the application · `movielens` library = the engine*

---

## The big idea

> **Given the movies a person has rated, predict the movies they'll love next.**

This is the engine behind Netflix "Top Picks", Spotify "Discover Weekly",
Amazon "Customers also bought", and YouTube's home feed.

We build the same core idea from scratch — and, just as importantly, the
**pipeline and engineering discipline** around it.

---

## Why it matters

- **Recommenders run the modern internet.** ~80% of Netflix watch-time and ~35%
  of Amazon sales come from recommendations.
- **It's a perfect teaching problem:** real, messy, *sparse* data; multiple
  modelling approaches; clear ways to measure success.
- **The skills transfer:** loading big data, sampling, train/test discipline,
  baselines→models, honest evaluation, and shipping reusable code — that's *any*
  data-science job.

---

## The dataset — MovieLens 32M

| | |
|---|---|
| Source | GroupLens Lab, University of Minnesota |
| Ratings | **32,000,204** |
| Users | **200,948** |
| Movies | **84,432** |
| Scale | 0.5 – 5.0 stars (half-star steps) |
| Tables | `ratings`, `movies` (title + genres), `tags`, `links` |

**The defining trait: sparsity.** The user × movie grid is **>99.99% empty** —
nobody rates even 1% of all movies. Learning taste from mostly-blank data is the
whole challenge.

---

## Key features of our solution

- 🧩 **Modular library (`src/movielens/`)** — 10 focused modules; the notebook is
  just short calls into it.
- ✂️ **Smart, swappable sampling** — sample *by user* (keep full histories), with
  named strategies you can hot-swap with one string.
- 🔌 **Pluggable models** — all share `.fit()/.predict()`; add a new one by
  registering a class.
- 🛟 **Always runs** — auto-falls back to realistic **synthetic data** if the 240 MB
  download is missing.
- ⚡ **Fast enough to demo live** — vectorised throughout (one fix took a step from
  **242 s → 1.6 s**).
- ✅ **50 unit tests, 93% coverage** — on fast synthetic data, no network needed.

---

## Sampling — taming 32 million rows

32M rows won't model live in class. **How** you cut matters:

- ❌ Random *rows* → shatters each user's history → useless.
- ✅ Random *users* → keep their **whole** history → coherent.

| Strategy | Keeps | Use |
|---|---|---|
| `random_users` | random users | unbiased baseline |
| `active_users` | top raters | **main slice** — richest signal |
| `dense` | active users × popular movies | dense grid → item-item CF shines |

Samples are pre-built to disk → **instant load**, and swappable to re-test
conclusions on a different slice.

---

## Model choice — a deliberate "ladder"

Each model adds exactly **one** idea, so we can see what each idea is worth:

1. **Global Mean** — predict the overall average. *The humblest baseline.*
2. **Bias** — `μ + user_bias + movie_bias`. Generous vs. harsh raters; good vs.
   bad films. *Simple, strong — the baseline to beat.*
3. **Matrix Factorization (SVD)** — discover hidden "taste dimensions"
   (action-y? romantic?) for users **and** movies; match them. *The heart of
   modern recommenders.*
4. **Item-Item CF** — "people who liked X also liked Y" via movie similarity.

We start simple **on purpose**: you can't claim a fancy model is good until it
beats the dumb one.

---

## Evaluation — measuring honestly

**Split:** *temporal* — hide each user's **most recent** ratings → train on the
past, predict the future (realistic, not "peeking").

**Two kinds of metric (a recommender has two jobs):**

- **Accuracy** — how close is a predicted rating?
  - **RMSE** / **MAE** — typical miss in stars *(lower = better)*.
- **Ranking** — are the *top* picks actually good?
  - **Precision@10 / Recall@10** *(higher = better)*.

---

## Results — the ladder pays off

**`active_users` slice** (3,000 top raters · 77K movies · 98% sparse):

| Model | RMSE ↓ | MAE ↓ |
|---|---|---|
| global_mean | 1.018 | 0.799 |
| bias | 0.826 | 0.623 |
| **matrix_factorization** | **0.800** | **0.601** |

**`dense` slice** (where item-item CF is feasible) — it takes the crown:

| Model | RMSE ↓ |
|---|---|
| **item_item_cf** | **0.820** |
| matrix_factorization | 0.837 |
| bias | 0.860 |

*Every added idea lowers the error.* ✔

---

## Other considerations

- **Scalability is a real constraint.** Item-item CF needs an *n_movies²*
  matrix — 77K² ≈ 6B cells. So we **skip it on huge catalogues** and demo it on
  `dense`. *The best algorithm on paper isn't always the one that fits in RAM.*
- **Cold start.** New users/movies have no history → we fall back to biases /
  averages instead of failing.
- **Reproducibility.** Fixed seeds; pinned `requirements.txt`; `uv` lockfile.
- **Regularisation.** Biases & factors are shrunk so rarely-rated items don't
  overfit.
- **Ethics, briefly.** Recommenders can create filter bubbles & popularity bias —
  worth measuring coverage/diversity, not just accuracy.

---

## How it's built

```
src/movielens/         the engine (library)
  data · sampling · splitting · eda · features
  models · evaluate · recommend · pipeline
notebooks/             movielens_pipeline.ipynb   ← the app
scripts/               prepare_samples · build_notebook
tests/                 50 tests · 93% coverage
```

Want a deep-learning model? Add a class to `models.py`.
Want a new sampling idea? Register a function in `sampling.py`.
**The notebook never changes — it just uses the engine.**

---

## Live demo

1. **Load** 32M ratings (synthetic fallback if missing).
2. **Sample** → `active_users`.
3. **Explore** → rating distribution, the long tail, genres.
4. **Split** → temporal (predict the future).
5. **Train** four models → **evaluate** → leaderboard + chart.
6. **Recommend** → real Top-10 for a real user.
7. **Swap** the sample → conclusions still hold.

```bash
uv run jupyter lab notebooks/movielens_pipeline.ipynb
```

---

## Takeaways & future work

**Takeaways**
- A recommender is *learnable from scratch* — and the **pipeline discipline**
  matters as much as the model.
- Start with a baseline; measure honestly; respect engineering limits.

**Future work**
- SVD++ / neural collaborative filtering
- Content-based recs from genre features (helps cold start)
- Tune `n_factors`; add diversity/coverage metrics
- Hybrid model (factorization + content)

---

# Thank you 🎬
### Questions?

*Repo: github.com/ianktoo · Proprietary — All Rights Reserved*
