"""Three recommendation models: bias baseline, matrix factorization, item-based CF."""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


# --------------------------------------------------------------------------- #
# Model 1: Bias baseline   r_hat = mu + b_u + b_i
# --------------------------------------------------------------------------- #
class BiasModel:
    def __init__(self, n_users, n_items, mu, reg_u=10, reg_i=10, n_iter=15):
        self.n_users, self.n_items, self.mu = n_users, n_items, mu
        self.reg_u, self.reg_i, self.n_iter = reg_u, reg_i, n_iter

    def fit(self, train):
        bu = np.zeros(self.n_users)
        bi = np.zeros(self.n_items)
        u, i, r = train.u.values, train.i.values, train.rating.values
        for _ in range(self.n_iter):
            dev = r - self.mu - bu[u]
            bi = np.bincount(i, dev, self.n_items) / (self.reg_i + np.bincount(i, None, self.n_items))
            dev = r - self.mu - bi[i]
            bu = np.bincount(u, dev, self.n_users) / (self.reg_u + np.bincount(u, None, self.n_users))
        self.bu, self.bi = np.nan_to_num(bu), np.nan_to_num(bi)
        return self

    def predict(self, u, i):
        return np.clip(self.mu + self.bu[u] + self.bi[i], 1, 5)

    def rank_scores(self, u):
        """Item scores for ranking (popularity ranker = item bias)."""
        return self.bi


# --------------------------------------------------------------------------- #
# Model 2: Matrix Factorization (biased SVD, SGD)
# --------------------------------------------------------------------------- #
class MatrixFactorization:
    def __init__(self, n_users, n_items, mu, k=15, lr=0.012, reg=0.1,
                 epochs=10, seed=42):
        self.n_users, self.n_items, self.mu = n_users, n_items, mu
        self.k, self.lr, self.reg, self.epochs, self.seed = k, lr, reg, epochs, seed

    def fit(self, train, verbose=False):
        rng = np.random.default_rng(self.seed)
        P = rng.normal(0, 0.1, (self.n_users, self.k))
        Q = rng.normal(0, 0.1, (self.n_items, self.k))
        bu = np.zeros(self.n_users)
        bi = np.zeros(self.n_items)
        u, i, r = train.u.values, train.i.values, train.rating.values.astype(float)
        for ep in range(self.epochs):
            order = rng.permutation(len(u))
            for idx in order:
                a, b, c = u[idx], i[idx], r[idx]
                err = c - (self.mu + bu[a] + bi[b] + P[a] @ Q[b])
                bu[a] += self.lr * (err - self.reg * bu[a])
                bi[b] += self.lr * (err - self.reg * bi[b])
                pa = P[a].copy()
                P[a] += self.lr * (err * Q[b] - self.reg * P[a])
                Q[b] += self.lr * (err * pa - self.reg * Q[b])
            if verbose:
                print(f"  epoch {ep + 1}/{self.epochs} done")
        self.P, self.Q, self.bu, self.bi = P, Q, bu, bi
        return self

    def predict(self, u, i):
        dot = np.sum(self.P[u] * self.Q[i], axis=1)
        return np.clip(self.mu + self.bu[u] + self.bi[i] + dot, 1, 5)

    def rank_scores(self, u):
        return self.bi + self.Q @ self.P[u]


# --------------------------------------------------------------------------- #
# Model 3: Item-based Collaborative Filtering
# --------------------------------------------------------------------------- #
class ItemCF:
    def __init__(self, n_users, n_items, mu, neighbors=50):
        self.n_users, self.n_items, self.mu = n_users, n_items, mu
        self.neighbors = neighbors

    def fit(self, train):
        cnt = np.bincount(train.i, None, self.n_items)
        ssum = np.bincount(train.i, train.rating, self.n_items)
        self.item_base = (ssum + self.mu * 10) / (cnt + 10)

        dev = train.rating.values - self.item_base[train.i.values]
        Rc = csr_matrix((dev, (train.u, train.i)), shape=(self.n_users, self.n_items))
        Rn = normalize(Rc.tocsc(), axis=0)
        sim = (Rn.T @ Rn).toarray().astype(np.float32)
        np.fill_diagonal(sim, 0)
        sim[sim < 0] = 0
        # prune to top-K neighbours per item
        for i in range(self.n_items):
            row = sim[i]
            if np.count_nonzero(row) > self.neighbors:
                thr = np.partition(row, -self.neighbors)[-self.neighbors]
                row[row < thr] = 0
        self.sim = sim
        self.Rdev = csr_matrix((dev, (train.u, train.i)),
                               shape=(self.n_users, self.n_items)).tocsr()
        return self

    def predict(self, u_arr, i_arr):
        out = np.empty(len(u_arr))
        for j, (a, b) in enumerate(zip(u_arr, i_arr)):
            row = self.Rdev[a]
            s = self.sim[b, row.indices]
            d = s.sum()
            out[j] = self.item_base[b] + (s @ row.data) / d if d > 1e-8 else self.item_base[b]
        return np.clip(out, 1, 5)

    def rank_scores(self, u):
        return np.asarray(self.Rdev[u].dot(self.sim)).ravel()
