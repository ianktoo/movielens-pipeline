"""Pre-build swappable sample sets from the full MovieLens 32M dataset.

Run this once after downloading the data:

    uv run python scripts/prepare_samples.py

It loads the full 32M ratings, carves out a few named samples using different
strategies, and saves each to ``data/samples/<key>``. The notebook can then load
any of them instantly with ``sampling.load_sample("active_users")`` instead of
re-reading 877 MB every time.

If the raw files are missing, it builds the samples from synthetic data instead,
so the script never fails.
"""

from __future__ import annotations

import time

from movielens import config, data, sampling

# (key, strategy, size, kwargs) — edit freely to add/replace sample sets.
SAMPLE_PLAN = [
    ("active_users", "active_users", 3000, {}),
    ("dense", "dense", 3000, {"top_movies": 800, "min_user_ratings": 30}),
    ("random_users", "random_users", 3000, {}),
]


def main() -> None:
    config.ensure_dirs()

    t0 = time.time()
    if data.raw_available():
        print("Loading full MovieLens 32M (this reads ~877 MB, give it a moment)...")
        ds = data.load_raw()
    else:
        print("Raw data not found — building samples from SYNTHETIC data instead.")
        ds = data.make_synthetic(n_users=8000, n_movies=3000, n_ratings=600_000)
    print(f"Source: {ds!r}  (loaded in {time.time() - t0:.1f}s)\n")

    for key, strategy, size, kwargs in SAMPLE_PLAN:
        t = time.time()
        sample = sampling.build_sample(ds, strategy=strategy, size=size, **kwargs)
        path = sampling.save_sample(sample, key=key)
        print(
            f"[{key:>13}] {sample!r}\n"
            f"               sparsity={sample.sparsity:.4f}  "
            f"saved -> {path}  ({time.time() - t:.1f}s)"
        )

    print(f"\nDone. Saved samples: {sampling.list_saved_samples()}")


if __name__ == "__main__":
    main()
