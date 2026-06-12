"""Top-K recommendation generation and item-based explanations."""
import numpy as np


def movie_name(i, inv_m, title_of, year_of):
    mid = inv_m[i]
    return f"{title_of.get(mid, '?')} ({year_of.get(mid, '?')})"


def recommend_topk(model, user, train_by_u, k=10):
    """Return the indices of the top-k recommended items for a user,
    excluding items already seen in training."""
    sc = model.rank_scores(user).copy()
    sc[train_by_u.get(user, [])] = -1e9
    top = np.argsort(sc)[::-1][:k]
    return top.tolist()


def explain_item(item_cf, movie_id, mids, inv_m, title_of, year_of, k=5):
    """Return the k most similar items to `movie_id` (item-CF neighbours),
    supporting 'because you watched X' explanations."""
    i = mids.get(movie_id)
    if i is None:
        return []
    s = item_cf.sim[i]
    nb = np.argsort(s)[::-1][:k]
    return [(movie_name(int(j), inv_m, title_of, year_of), float(s[j]))
            for j in nb if s[j] > 0]
