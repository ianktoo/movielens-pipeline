# Data and sampling

This note explains where the data comes from, what makes it hard to model, and
why we sample the way we do. It expands on the short version in the
[README](../README.md).

## Source data

Everything starts from the **MovieLens 32M** dataset published by the GroupLens
research lab at the University of Minnesota.

- Homepage and license: <https://grouplens.org/datasets/movielens/32m/>
- Collected October 2023, released May 2024.
- Download: `ml-32m.zip` (about 228 MB zipped, roughly 1 GB unzipped).

We do not commit the data. It is git-ignored because it is large and because it
is GroupLens property, distributed under
[their own terms of use](https://grouplens.org/datasets/movielens/). Each person
regenerates it locally with `scripts/download_data.py`.

### What is actually in the archive

| File | Rows | What it holds |
|------|------|---------------|
| `ratings.csv` | 32,000,204 | one row per rating: `userId, movieId, rating, timestamp` |
| `movies.csv` | 87,585 | one row per movie: `movieId, title, genres` |
| `tags.csv` | about 2,000,000 | free-text tags users applied to movies |
| `links.csv` | 87,585 | IMDb and TMDB ids for each movie |

Our pipeline only loads `ratings.csv` and `movies.csv`. The tags and links are
available for future content-based work but are not used yet.

### A detail worth noticing

The catalogue lists **87,585** movies, but only **84,432** of them appear in the
ratings file. About 3,000 movies were never rated by anyone in this snapshot.
This gap is a small but real reminder that the catalogue and the observed data
are not the same thing, which matters when you reason about coverage and the
cold-start problem.

### Headline numbers

| Quantity | Value |
|----------|-------|
| Ratings | 32,000,204 |
| Users | 200,948 |
| Movies in catalogue | 87,585 |
| Movies actually rated | 84,432 |
| Rating scale | 0.5 to 5.0 stars, in half-star steps |

## Why this data is hard: sparsity

If you arrange users on one axis and movies on the other, you get a grid with
about 200,948 times 84,432, which is roughly 17 billion cells. Only 32 million
of those cells are filled. That means the grid is more than **99.99 percent
empty**. Nobody rates even a tiny fraction of all movies, so the central
modelling challenge is learning taste from mostly-blank data.

Sparsity is the single most useful one-number summary of "how hard is this
recommender problem", and the `Dataset.sparsity` property reports it for any
slice we look at.

## Why we sample

Thirty-two million rows is too much to train and re-train live in a classroom.
So we cut the data down. The important idea is that **how you cut matters**.

| Approach | What it does | Problem or benefit |
|----------|--------------|--------------------|
| Random rows | pick N ratings at random | breaks each user into a few disconnected ratings, so there is no history to learn from. Avoid this. |
| Random users | pick N users, keep all their ratings | every kept user brings a full history, so the data stays coherent. This is what we do. |

We always sample **by user**. Each strategy below decides *which* users to keep,
then we keep all of their ratings.

## The three sampling strategies

| Strategy | Which users it keeps | Why you would use it |
|----------|----------------------|----------------------|
| `random_users` | a uniform random subset | an unbiased baseline slice |
| `active_users` | the most prolific raters | the richest signal per user, our main slice |
| `dense` | active users restricted to popular movies | a much fuller grid, which is what item-item collaborative filtering needs |

The `dense` strategy is special: after picking active users it also trims the
movie set down to the most-rated movies. That trade gives up catalogue size in
exchange for a far less empty grid.

### The samples we actually build

These are the numbers produced by `scripts/prepare_samples.py` on the real
32M data. They are the slices the notebooks load.

| Sample | Ratings | Users | Movies | Sparsity (empty) |
|--------|---------|-------|--------|------------------|
| `active_users` | 5,190,485 | 3,000 | 77,334 | 97.8 percent |
| `dense` | 301,416 | 3,000 | 800 | 87.4 percent |
| `random_users` | 464,411 | 3,000 | 20,047 | 99.2 percent |

Notice the story these three rows tell. Keeping the same 3,000 users but
changing which movies we keep moves sparsity from 99.2 percent (random users,
spread across 20,000 movies) down to 87.4 percent (dense, concentrated on 800
popular movies). That single design choice is what later makes item-item
collaborative filtering feasible.

## Reproducibility

Samples are written to `data/samples/<key>/` as two Parquet files (`ratings`
and `movies`) so the notebooks load them instantly instead of re-reading the
836 MB ratings file every run. Sampling uses a fixed random seed, so everyone
who runs the scripts gets the same slices.
