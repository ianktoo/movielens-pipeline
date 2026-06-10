---
marp: true
title: MovieLens 32M, a full data-science pipeline
author: ianktoo
paginate: true
theme: default
---

# Predicting what you will watch next

## A full data-science pipeline on MovieLens 32M

We take 32 million raw ratings and turn them into a personalised "Top 10 for
you" list, building every step ourselves.

The notebook is the application. The `movielens` library is the engine.

---

## Outline

1. Abstract: what we built and what we found
2. The problem and why it matters
3. The dataset: MovieLens 32M
4. The pipeline, one step at a time
5. Sampling: taming 32 million rows
6. Splitting: testing honestly
7. The models: a deliberate ladder
8. Evaluation: two kinds of metric
9. Results: does the ladder pay off?
10. A real recommendation
11. Engineering considerations
12. How it is built
13. Takeaways and future work

---

## Abstract

We build a movie recommender end to end on the MovieLens 32M dataset: 32 million
ratings by about 200,000 people on about 87,000 movies.

The work has two parts. First, a small reusable library that loads, samples,
splits, models, and evaluates. Second, a notebook that calls the library and
tells the story.

We train four models that form a ladder, from "always guess the average" to
collaborative filtering. The finding is that **each added idea lowers the
error**, and that the hardest parts of the problem are the data and the
discipline, not the model itself.

---

## The problem

> Given the movies a person has already rated, predict the movies they will
> love next.

This is the engine behind Netflix "Top Picks", Spotify "Discover Weekly",
Amazon "Customers also bought", and the YouTube home feed.

We build the same core idea from scratch, and just as importantly we build the
pipeline and engineering discipline around it.

---

## Why it matters

| Reason | Detail |
|--------|--------|
| Recommenders run the modern internet | a large share of what people watch and buy online is suggested, not searched for |
| It is an excellent teaching problem | real, messy, sparse data, several modelling approaches, and clear ways to measure success |
| The skills transfer | loading big data, sampling, train and test discipline, baselines before models, and honest evaluation are the core of any data-science job |

---

## The dataset: MovieLens 32M

Source: GroupLens lab, University of Minnesota.
Link: https://grouplens.org/datasets/movielens/32m/

| Quantity | Value |
|----------|-------|
| Ratings | 32,000,204 |
| Users | 200,948 |
| Movies in catalogue | 87,585 |
| Movies actually rated | 84,432 |
| Rating scale | 0.5 to 5.0 stars, half-star steps |
| Tables | ratings, movies, tags, links |

Collected October 2023. The data is GroupLens property and is used here under
their terms of use.

---

## The defining trait: sparsity

Lay every user against every movie in a grid: about 200,948 times 84,432, which
is roughly 17 billion cells. Only 32 million are filled.

> The grid is more than 99.99 percent empty.

Nobody rates even a tiny fraction of all movies. Learning taste from
mostly-blank data is the whole challenge, and sparsity is the single best
one-number description of how hard the problem is.

---

## The pipeline, one step at a time

| Step | Question it answers |
|------|---------------------|
| 1. Load | what do we actually have? |
| 2. Sample | how do we make 32M rows runnable? |
| 3. Explore | what does the data look like? |
| 4. Features | what can we compute about users and movies? |
| 5. Split | how do we test honestly? |
| 6. Model | four models, simple to smart |
| 7. Evaluate | which model is best, and by how much? |
| 8. Recommend | actual movies for a real person |
| 9. Swap samples | do the conclusions hold on a different slice? |

---

## Sampling: how you cut matters

We cannot model 32M rows live. We sample. The key decision is what to sample.

| Approach | What it does | Result |
|----------|--------------|--------|
| Random rows | pick N ratings at random | shatters each user's history, useless |
| Random users | pick N users, keep all their ratings | full histories, coherent data |

We always sample by user. Each kept user brings their whole history, which is
what keeps the rating grid usable.

---

## Three sampling strategies

| Strategy | Keeps | Best for |
|----------|-------|----------|
| `random_users` | a random subset of users | an unbiased baseline slice |
| `active_users` | the most prolific raters | richest signal, our main slice |
| `dense` | active users restricted to popular movies | a fuller grid where item-item CF works |

The slices we actually build, from the real data:

| Sample | Ratings | Users | Movies | Empty |
|--------|---------|-------|--------|-------|
| `active_users` | 5,190,485 | 3,000 | 77,334 | 97.8% |
| `dense` | 301,416 | 3,000 | 800 | 87.4% |
| `random_users` | 464,411 | 3,000 | 20,047 | 99.2% |

---

## Splitting: testing honestly

We hide some ratings, train on the rest, and check whether we would have
predicted the hidden ones.

| Strategy | What it hides | Pros | Cons |
|----------|---------------|------|------|
| Random | a random 20% per user | simple | lets the model peek at the future |
| Temporal | the most recent 20% per user | matches reality, predict the future | needs timestamps |

We default to the temporal split. Every test user is also in the training set,
because you cannot fairly test a recommender on someone it has never seen.

---

## The models: a deliberate ladder

Each rung adds exactly one idea, so we can see what each idea is worth.

| Rung | Model | Idea | Strength | Limitation |
|------|-------|------|----------|------------|
| 1 | Global mean | guess the overall average | a true floor | ignores everyone |
| 2 | Bias | mu plus user and movie offsets | simple and strong | no personal taste |
| 3 | Matrix factorization | learn hidden taste and appeal vectors | personalised, modern core | needs enough ratings |
| 4 | Item-item CF | "liked this, so liked that" | strong on dense data | does not scale to big catalogues |

We start simple on purpose: you cannot claim a fancy model is good until it
beats the dumb one.

---

## Evaluation: two kinds of metric

A recommender has two jobs, so we measure both.

| Job | Question | Metric | Better |
|-----|----------|--------|--------|
| Accuracy | how close is a predicted rating? | RMSE, MAE | lower |
| Ranking | are the top picks actually good? | Precision@10, Recall@10 | higher |

RMSE punishes big misses extra. MAE is the plain average miss in stars.
Precision@10 asks how many of our top ten the user liked. Recall@10 asks how
much of what they liked we surfaced.

---

## Results: the active_users slice

5,190,485 ratings, 3,000 users, 77,334 movies, 97.8 percent empty, temporal
split.

| Model | RMSE | MAE | Precision@10 | Recall@10 |
|-------|------|-----|--------------|-----------|
| global_mean | 1.018 | 0.799 | 0.299 | 0.031 |
| bias | 0.826 | 0.623 | 0.598 | 0.086 |
| matrix_factorization | 0.800 | 0.601 | 0.649 | 0.099 |

Item-item CF is skipped here: 77,334 squared is about 6 billion cells, too large
for a similarity matrix.

---

## Results: the dense slice

301,416 ratings, 3,000 users, 800 movies, 87.4 percent empty, temporal split.
This is where item-item CF becomes feasible, and it takes the top spot.

| Model | RMSE | MAE | Precision@10 | Recall@10 |
|-------|------|-----|--------------|-----------|
| item_item_cf | 0.820 | 0.612 | 0.684 | 0.758 |
| matrix_factorization | 0.837 | 0.630 | 0.684 | 0.757 |
| bias | 0.860 | 0.650 | 0.676 | 0.749 |
| global_mean | 0.998 | 0.780 | 0.616 | 0.702 |

Every added idea lowers the error.

---

## Reading the numbers with care

One honest caveat for anyone studying the tables.

Recall@10 looks far higher on `dense` (about 0.76) than on `active_users`
(about 0.10). Nothing got better. The candidate pool changed: ten picks cover
much more of 800 movies than of 77,334 movies.

> Compare models within a sample, not across samples. A metric only means
> something relative to the slice it was computed on.

---

## A real recommendation

For an active user, we put what they already love next to what the model
suggests they watch next, excluding films they have already seen.

| Step | Output |
|------|--------|
| Their top-rated films | a list of movies they scored 5.0 |
| Model recommendations | a ranked top 10 they have not seen, with predicted ratings |

This is the payoff. The numbers are a means; the recommendation is the end. It
also lets the room sanity-check the model by eye.

---

## Engineering considerations

| Concern | How we handle it |
|---------|------------------|
| Scalability | item-item CF needs an n_movies by n_movies matrix, so we skip it on huge catalogues and demo it on `dense` |
| Cold start | new users or movies have no history, so we fall back to biases and averages instead of failing |
| Reproducibility | fixed seeds, a pinned `requirements.txt`, and a uv lockfile |
| Regularisation | biases and factors are shrunk so rarely-rated items do not overfit |
| Ethics | recommenders can create filter bubbles and popularity bias, worth measuring coverage and diversity, not just accuracy |

---

## How it is built

```text
src/movielens/    the engine (library)
  data, sampling, splitting, eda, features
  models, evaluate, recommend, pipeline
notebooks/        walkthrough (explained) and minimal (run only)
scripts/          download_data, prepare_samples, build_notebook
tests/            50 tests, 93 percent coverage
```

Want a deep-learning model? Add a class to `models.py`.
Want a new sampling idea? Register a function in `sampling.py`.
The notebook never changes. It just uses the engine.

---

## Live demo

1. Load 32M ratings (synthetic fallback if the files are missing).
2. Sample to `active_users`.
3. Explore: rating distribution, the long tail, genres.
4. Split temporally (predict the future).
5. Train four models, evaluate, show the leaderboard and chart.
6. Recommend a real top 10 for a real user.
7. Swap the sample and confirm the conclusions hold.

```bash
uv run jupyter lab notebooks/movielens_walkthrough.ipynb
```

---

## Takeaways and future work

**Takeaways**

- A recommender is learnable from scratch, and the pipeline discipline matters
  as much as the model.
- Start with a baseline, measure honestly, and respect engineering limits.

**Future work**

- SVD++ or neural collaborative filtering.
- Content-based recommendations from genres and tags, which would help cold
  start.
- Tune the number of factors, and add diversity and coverage metrics.

---

# Thank you

## Questions?

Source data: https://grouplens.org/datasets/movielens/32m/
