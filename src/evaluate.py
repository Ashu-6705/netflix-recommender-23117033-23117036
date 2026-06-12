"""Evaluation: rating accuracy (RMSE/MAE) and ranking metrics (MAP@10, etc.)."""
import numpy as np


# --------------------------- rating accuracy ------------------------------- #
def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


# --------------------------- ranking helpers ------------------------------- #
def average_precision_at_k(ranked, relevant, k=10):
    hits, score = 0, 0.0
    for n, item in enumerate(ranked[:k], 1):
        if item in relevant:
            hits += 1
            score += hits / n
    return score / min(len(relevant), k) if relevant else 0.0


def ndcg_at_k(ranked, relevant, k=10):
    dcg = sum(1 / np.log2(n + 1) for n, item in enumerate(ranked[:k], 1) if item in relevant)
    idcg = sum(1 / np.log2(n + 1) for n in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def build_eval_sets(train, test, relevance_threshold=3.5):
    """Return (train_by_user, relevant_by_user, eval_users)."""
    train_by_u = {u: np.array(sorted(s)) for u, s in train.groupby("u").i.agg(set).items()}
    relevant = {u: set(g[g.rating >= relevance_threshold].i.values)
                for u, g in test.groupby("u")}
    eval_users = [u for u in relevant if len(relevant[u]) > 0]
    return train_by_u, relevant, eval_users


# --------------------------- ranking protocols ----------------------------- #
def evaluate_allitems(model, eval_users, train_by_u, relevant, n_items, k=10):
    """Rank every catalog item; report MAP/P/R/NDCG@k and catalog coverage."""
    APs, Ps, Rs, Ns, covered = [], [], [], [], set()
    for u in eval_users:
        sc = model.rank_scores(u).copy()
        sc[train_by_u.get(u, [])] = -1e9
        top = np.argpartition(sc, -k)[-k:]
        top = top[np.argsort(sc[top])[::-1]].tolist()
        rel = relevant[u]
        hits = len(set(top) & rel)
        APs.append(average_precision_at_k(top, rel, k))
        Ps.append(hits / k)
        Rs.append(hits / len(rel))
        Ns.append(ndcg_at_k(top, rel, k))
        covered.update(top)
    return dict(MAP10=float(np.mean(APs)), P10=float(np.mean(Ps)),
                R10=float(np.mean(Rs)), NDCG10=float(np.mean(Ns)),
                Coverage=len(covered) / n_items)


def evaluate_sampled(model, eval_users, train_by_u, relevant, n_items,
                     n_neg=100, k=10, seed=11):
    """Rank held-out positives against `n_neg` random negatives per user."""
    rng = np.random.default_rng(seed)
    all_items = set(range(n_items))
    APs, HRs, Ns = [], [], []
    for u in eval_users:
        seen, rel = set(train_by_u.get(u, [])), relevant[u]
        pool = list(all_items - seen - rel)
        negs = rng.choice(pool, min(n_neg, len(pool)), replace=False)
        cand = np.array(list(rel) + list(negs))
        ranked = cand[np.argsort(model.rank_scores(u)[cand])[::-1]]
        hits = sum(1 for it in ranked[:k] if it in rel)
        APs.append(average_precision_at_k(list(ranked), rel, k))
        HRs.append(1.0 if hits else 0.0)
        Ns.append(ndcg_at_k(list(ranked), rel, k))
    return dict(MAP10=float(np.mean(APs)), HitRate10=float(np.mean(HRs)),
                NDCG10=float(np.mean(Ns)))
