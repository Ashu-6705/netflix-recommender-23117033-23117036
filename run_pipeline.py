"""
Run the full recommendation pipeline from the command line.

Usage:
    python run_pipeline.py --ratings data/ratings_sample.csv --movies data/movie_titles_clean.csv

This loads and filters the data, trains all three models, evaluates RMSE and
ranking metrics (MAP@10, etc.), and prints a sample of Top-10 recommendations.
"""
import argparse
import numpy as np

from src.data import prepare
from src.models import BiasModel, MatrixFactorization, ItemCF
from src.evaluate import (rmse, mae, build_eval_sets,
                          evaluate_allitems, evaluate_sampled)
from src.recommend import recommend_topk, movie_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="data/ratings_sample.csv")
    ap.add_argument("--movies", default="data/movie_titles_clean.csv")
    ap.add_argument("--min_user", type=int, default=15)
    ap.add_argument("--min_movie", type=int, default=10)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--relevance", type=float, default=3.5)
    ap.add_argument("--max_eval_users", type=int, default=2500)
    args = ap.parse_args()

    print("Loading and preparing data...")
    d = prepare(args.ratings, args.movies, args.min_user, args.min_movie, args.test_frac)
    train, test = d["train"], d["test"]
    n_users, n_items, mu = d["n_users"], d["n_items"], d["mu"]
    print(f"  train={len(train):,}  test={len(test):,}  users={n_users:,}  items={n_items:,}")

    print("\nTraining models...")
    bias = BiasModel(n_users, n_items, mu).fit(train)
    print("  bias model done")
    mf = MatrixFactorization(n_users, n_items, mu).fit(train)
    print("  matrix factorization done")
    icf = ItemCF(n_users, n_items, mu).fit(train)
    print("  item-based CF done")

    # ---- rating accuracy ----
    print("\n=== RATING ACCURACY ===")
    pb = bias.predict(test.u.values, test.i.values)
    pm = mf.predict(test.u.values, test.i.values)
    samp = test.sample(min(15000, len(test)), random_state=1)
    pc = icf.predict(samp.u.values, samp.i.values)
    print(f"  Bias  : RMSE={rmse(pb, test.rating):.4f}  MAE={mae(pb, test.rating):.4f}")
    print(f"  MF    : RMSE={rmse(pm, test.rating):.4f}  MAE={mae(pm, test.rating):.4f}")
    print(f"  ItemCF: RMSE={rmse(pc, samp.rating):.4f}  MAE={mae(pc, samp.rating):.4f}")

    # ---- ranking ----
    print("\n=== RANKING QUALITY ===")
    train_by_u, relevant, eval_users = build_eval_sets(train, test, args.relevance)
    rng = np.random.default_rng(7)
    if len(eval_users) > args.max_eval_users:
        eval_users = list(rng.choice(eval_users, args.max_eval_users, replace=False))
    print(f"  evaluating on {len(eval_users):,} users (>=1 relevant held-out item)")
    for name, model in [("Popularity", bias), ("MF", mf), ("ItemCF", icf)]:
        a = evaluate_allitems(model, eval_users, train_by_u, relevant, n_items)
        s = evaluate_sampled(model, eval_users, train_by_u, relevant, n_items)
        print(f"  {name:10} | all-items MAP@10={a['MAP10']:.4f} Cov={a['Coverage']:.2%}"
              f" | sampled MAP@10={s['MAP10']:.4f} HR@10={s['HitRate10']:.4f}")

    # ---- sample recommendations ----
    print("\n=== SAMPLE MF TOP-10 ===")
    cand = [u for u in eval_users if len(relevant[u]) >= 3 and len(train_by_u.get(u, [])) >= 15]
    if cand:
        u = sorted(cand)[0]
        title_of = dict(zip(d["movies"].movie_id, d["movies"].title))
        year_of = dict(zip(d["movies"].movie_id, d["movies"].year))
        for rank, i in enumerate(recommend_topk(mf, u, train_by_u), 1):
            tag = "  <-- relevant (held-out)" if i in relevant[u] else ""
            print(f"  {rank:2}. {movie_name(int(i), d['inv_m'], title_of, year_of)}{tag}")


if __name__ == "__main__":
    main()
