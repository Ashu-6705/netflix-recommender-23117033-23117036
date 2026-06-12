"""Data loading, k-core filtering, and temporal train/test split."""
import numpy as np
import pandas as pd


def load_data(ratings_path, movies_path):
    """Load ratings and movie metadata."""
    ratings = pd.read_csv(ratings_path, parse_dates=["date"])
    movies = pd.read_csv(movies_path)
    return ratings, movies


def kcore_filter(df, min_user=15, min_movie=10, max_iters=10):
    """Iteratively keep users with >= min_user ratings and movies with
    >= min_movie ratings until the set stabilises (a k-core)."""
    for _ in range(max_iters):
        before = len(df)
        mc = df.groupby("movie_id").size()
        df = df[df.movie_id.isin(mc[mc >= min_movie].index)]
        uc = df.groupby("user_id").size()
        df = df[df.user_id.isin(uc[uc >= min_user].index)]
        if len(df) == before:
            break
    return df.reset_index(drop=True)


def reindex(df):
    """Map raw user/movie ids to contiguous integer indices.
    Returns (df_with_u_i, uids, mids, inv_m, n_users, n_items)."""
    uids = {u: i for i, u in enumerate(df.user_id.unique())}
    mids = {m: i for i, m in enumerate(df.movie_id.unique())}
    inv_m = {v: k for k, v in mids.items()}
    df = df.copy()
    df["u"] = df.user_id.map(uids)
    df["i"] = df.movie_id.map(mids)
    return df, uids, mids, inv_m, len(uids), len(mids)


def temporal_split(df, test_frac=0.20):
    """Per-user temporal hold-out: the most recent `test_frac` of each
    user's ratings becomes the test set. Cold test items/users are dropped."""
    df = df.sort_values(["u", "date"]).reset_index(drop=True)
    mask = np.zeros(len(df), bool)
    for _, idx in df.groupby("u").indices.items():
        k = max(1, int(round(len(idx) * test_frac)))
        mask[idx[-k:]] = True
    train = df[~mask].reset_index(drop=True)
    test = df[mask].reset_index(drop=True)
    train_items, train_users = set(train.i.unique()), set(train.u.unique())
    test = test[test.i.isin(train_items) & test.u.isin(train_users)].reset_index(drop=True)
    return train, test


def prepare(ratings_path, movies_path, min_user=15, min_movie=10, test_frac=0.20):
    """End-to-end: load -> filter -> reindex -> split. Returns a dict bundle."""
    ratings, movies = load_data(ratings_path, movies_path)
    df = kcore_filter(ratings, min_user, min_movie)
    df, uids, mids, inv_m, n_users, n_items = reindex(df)
    train, test = temporal_split(df, test_frac)
    return dict(train=train, test=test, movies=movies, uids=uids, mids=mids,
                inv_m=inv_m, n_users=n_users, n_items=n_items,
                mu=float(train.rating.mean()))
