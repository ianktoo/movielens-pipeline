# Splitting and evaluation

A recommender is only as trustworthy as the way you test it. This note covers
how we split the data and how we score the models. The code lives in
`src/movielens/splitting.py` and `src/movielens/evaluate.py`.

## Splitting: testing honestly

To know whether a model is any good, we hide some ratings, train on the rest,
and check whether the model would have predicted the hidden ones. There are two
honest ways to choose which ratings to hide.

| Strategy | What it hides | Pros | Cons |
|----------|---------------|------|------|
| Random | a random 20 percent of each user's ratings | simple, easy to reason about | lets the model peek at "future" ratings to predict the "past" |
| Temporal | each user's most recent 20 percent of ratings | mirrors reality: train on the past, predict the future | needs a timestamp column |

We default to the **temporal** split because it matches how a recommender is
actually used. You only ever know a user's past when you recommend their future.

Both strategies split *within* each user, so every user who appears in the test
set also appears in training. You cannot fairly test a recommender on a person
it has never seen, so we never let that happen.

## Two kinds of metric

A recommender has two jobs, and we measure both.

| Job | Question | Metric | Direction |
|-----|----------|--------|-----------|
| Accuracy | when we predict a rating, how close are we? | RMSE, MAE | lower is better |
| Ranking | of the items we put at the top, how many are good? | Precision@10, Recall@10 | higher is better |

- **RMSE** (root mean squared error) is the typical miss in stars, and it
  punishes big misses extra because of the squaring.
- **MAE** (mean absolute error) is the plain average miss in stars, which is
  easier to explain to a non-technical audience.
- **Precision@10** is, of our top ten picks, the fraction the user actually
  liked (rated 4.0 or higher in the held-out set).
- **Recall@10** is, of everything the user liked, the fraction we surfaced in
  the top ten.

Accuracy is computed over the full test set. Ranking metrics loop per user and
are the slow part, so we cap them at 200 users per model for a snappy demo.

## Reading the results carefully

The verified leaderboards are in the [README](../README.md) and the
[presentation](../PRESENTATION.md). Two points are worth stressing for anyone
studying the numbers.

First, **RMSE and MAE always agree on the ordering** of our models, and each
rung of the ladder beats the one below it. That consistency is the headline
result.

Second, **recall@10 is not comparable across samples**, and this trips people
up. On the `active_users` slice recall@10 sits near 0.10, while on the `dense`
slice it is near 0.76. Nothing got better or worse between them. The difference
is the size of the candidate pool: with 77,334 movies in play, ten picks can
only ever cover a tiny share of everything a user liked, so recall is
mechanically small. With 800 movies, ten picks cover much more. The lesson is
that a metric value only means something relative to the slice it was computed
on, so compare models within a sample, not across samples.
