# The models

We build four models and arrange them as a deliberate ladder. Each rung adds
exactly one idea, so when the error drops we know which idea earned the drop.
This note explains each model, its intuition, and its trade-offs. The code lives
in `src/movielens/models.py`.

All four share the same tiny interface, which is what lets the notebook and the
evaluation code treat them interchangeably:

```python
model = BiasModel().fit(train_df)
preds = model.predict(user_ids, movie_ids)   # numpy array of ratings
```

## The ladder at a glance

| Rung | Model | One-line idea | Strengths | Limitations |
|------|-------|---------------|-----------|-------------|
| 1 | Global mean | predict the overall average rating for everything | trivial, instant, a true floor | ignores every user and every movie |
| 2 | Bias | average plus a per-user and per-movie offset | simple, strong, hard to beat | cannot capture personal taste, only overall generosity and quality |
| 3 | Matrix factorization | learn hidden taste and appeal vectors, then match them | captures personalised taste, the core of modern recommenders | needs enough ratings per user and movie, factors are not directly interpretable |
| 4 | Item-item collaborative filtering | "people who liked this also liked that" | intuitive, strong on dense data | builds an n_movies by n_movies matrix, so it does not scale to large catalogues |

## 1. Global mean

The humblest baseline. Fit by computing one number, the mean of all training
ratings, and predict that number for every user and movie. It is useless as a
product, but it is the floor every other model has to clear. If a fancy model
cannot beat "always guess the average", the fancy model is broken.

## 2. Bias model

The prediction is

```
r_hat(u, i) = mu + b_u + b_i
```

where `mu` is the global mean, `b_u` is how generous or harsh user `u` is on
average, and `b_i` is how good or bad movie `i` is on average. The biases are
computed with shrinkage (regularisation) so that a user or movie with only a
couple of ratings is pulled toward zero instead of being trusted completely:

```
b_i = sum_over_users(r_ui - mu) / (lambda + n_i)
b_u = sum_over_movies(r_ui - mu - b_i) / (lambda + n_u)
```

The `lambda` term (we use 10) is the strength of that pull. This model is
surprisingly strong and is the standard baseline that any real recommender has
to beat.

## 3. Matrix factorization

After the bias model explains "this user is generous" and "this movie is good",
what is left over is the **interaction**: does this particular user have a taste
for what this particular movie offers? Matrix factorization learns that.

We take the residual (actual rating minus the bias prediction), arrange it as a
sparse user-by-movie matrix, and factor it with a truncated SVD into a small
number of latent dimensions (we use 20). Each user gets a taste vector `p_u` and
each movie gets an appeal vector `q_i`. You can loosely read the dimensions as
discovered themes such as "amount of action" or "amount of romance", although
the model finds them on its own and they are not labelled.

The final prediction adds the matched interaction back onto the bias:

```
r_hat(u, i) = mu + b_u + b_i + p_u . q_i
```

This is the heart of modern collaborative filtering. The main cost is that it
needs a reasonable number of ratings per user and per movie to estimate the
vectors well.

## 4. Item-item collaborative filtering

This model formalises "people who liked X also liked Y". We compute the cosine
similarity between every pair of movies, based on the mean-centred ratings they
share. To predict how a user would rate a movie, we take a similarity-weighted
average of how that user rated the most similar movies they have already seen.

The catch is scale. The similarity matrix has `n_movies` rows and `n_movies`
columns. On the `active_users` slice that would be about 77,334 squared, which
is roughly 6 billion cells, far too large to hold in memory. The model therefore
refuses to fit when the catalogue is bigger than a configurable cap
(`max_movies`, default 5,000), and we showcase it on the `dense` slice (800
movies) where the matrix is tiny. This is a genuine engineering lesson: the best
algorithm on paper is not always the one that fits in RAM.

## How they are registered

Every model is registered in a `MODELS` dictionary keyed by name, so the
pipeline can build and compare them uniformly:

```python
MODELS = {
    "global_mean": GlobalMeanModel,
    "bias": BiasModel,
    "matrix_factorization": MatrixFactorization,
    "item_item_cf": ItemItemCF,
}
```

To add a fifth model, write a class with `.fit()` and `.predict()` and add it to
this dictionary. Nothing else in the pipeline has to change. See
[library-architecture.md](library-architecture.md) for the full extension story.
