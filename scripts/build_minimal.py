"""Generate notebooks/movielens_minimal.ipynb programmatically with nbformat.

This is the minimal, run-only version of the pipeline. It performs the exact
same end-to-end run as the walkthrough notebook (load, sample, explore, split,
train, evaluate, recommend, swap), but with almost no prose: just short section
headers and the code. Use it when you want to run the whole thing top to bottom
without the teaching narrative.

The explained counterpart is built by scripts/build_walkthrough.py.

Run:

    uv run python scripts/build_minimal.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "movielens_minimal.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(new_code_cell(text.strip("\n")))


md(r"""
# MovieLens 32M pipeline (minimal)

Run-only version of the full pipeline. Same end-to-end run as
`movielens_walkthrough.ipynb`, without the narrative. Run all cells top to
bottom. Data source: https://grouplens.org/datasets/movielens/32m/
""")

code(r"""
%matplotlib inline
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

import movielens as ml

pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
""")

md("## 1. Load")
code(r"""
full = ml.data.load_or_synthesize()
print(full)
pd.DataFrame([full.summary()]).T.rename(columns={0: "value"})
""")

md("## 2. Sample (active_users)")
code(r"""
saved = ml.sampling.list_saved_samples()
if "active_users" in saved:
    sample = ml.sampling.load_sample("active_users")
else:
    sample = ml.sampling.build_sample(full, strategy="active_users", size=3000)
print(sample, f"| sparsity: {sample.sparsity:.1%} empty")
""")

md("## 3. Explore")
code(r"""
ml.eda.plot_rating_distribution(sample); plt.show()
ml.eda.plot_activity_long_tail(sample); plt.show()
ml.eda.plot_genre_counts(sample, top=15); plt.show()
display(ml.eda.top_movies(sample, n=10, min_ratings=50))
""")

md("## 4. Features")
code(r"""
display(ml.features.user_features(sample).head())
display(ml.features.movie_features(sample).head())
""")

md("## 5. Split (temporal)")
code(r"""
split = ml.splitting.split(sample.ratings, strategy="temporal", test_frac=0.2)
print(split)
""")

md("## 6. Train")
code(r"""
to_train = [
    ("global_mean",          ml.models.GlobalMeanModel()),
    ("bias",                 ml.models.BiasModel(reg=10.0)),
    ("matrix_factorization", ml.models.MatrixFactorization(n_factors=20, reg=10.0)),
]
if sample.n_movies <= 5000:
    to_train.append(("item_item_cf", ml.models.ItemItemCF(k_neighbors=30)))

models = {}
for name, model in to_train:
    model.fit(split.train)
    models[name] = model
    print(f"fitted {name}")
""")

md("## 7. Evaluate")
code(r"""
results = [{"model": name, **ml.evaluate.evaluate_model(m, split.test, k=10)}
           for name, m in models.items()]
leaderboard = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
display(leaderboard)

order = leaderboard.sort_values("rmse", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(order["model"], order["rmse"], color="#4C72B0")
ax.set_xlabel("RMSE  (lower is better)")
ax.set_title("Rating-prediction accuracy by model")
for i, v in enumerate(order["rmse"]):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center")
plt.tight_layout(); plt.show()
""")

md("## 8. Recommend")
code(r"""
best_name = leaderboard.iloc[0]["model"]
best_model = models[best_name]
example_user = int(sample.ratings["userId"].value_counts().index[0])
print(f"best model: {best_name} | example user: {example_user}")
display(ml.recommend.user_history(sample, example_user, n=8))
display(ml.recommend.recommend_for_user(best_model, sample, example_user, n=10))
""")

md("## 9. Swap samples")
code(r"""
for key in ["dense", "random_users"]:
    if key not in ml.sampling.list_saved_samples():
        continue
    ds = ml.sampling.load_sample(key)
    specs = list(ml.pipeline.DEFAULT_MODELS)
    if ds.n_movies > 5000:
        specs = [s for s in specs if s.name != "item_item_cf"]
    print(f"\n===== {key}  ({ds.n_ratings:,} ratings, {ds.n_movies:,} movies) =====")
    res = ml.pipeline.run_pipeline(ds, sample_strategy=None, model_specs=specs, verbose=False)
    display(res.scores[["model", "rmse", "mae", "precision@10", "recall@10"]])
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
