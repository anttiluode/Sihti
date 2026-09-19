"""Deterministic k-means (k-means++ seeding + Lloyd), numpy only, vectorised."""
import numpy as np


def _kmeanspp(X, xx, k, rng):
    n = X.shape[0]
    C = np.empty((k, X.shape[1]))
    C[0] = X[rng.integers(n)]
    d2 = np.maximum(xx - 2.0 * X @ C[0] + C[0] @ C[0], 0.0)
    for t in range(1, k):
        s = d2.sum()
        p = d2 / s if s > 0 else np.full(n, 1.0 / n)
        C[t] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, np.maximum(xx - 2.0 * X @ C[t] + C[t] @ C[t], 0.0))
    return C


def kmeans(X, k, n_init=4, iters=60, seed=0):
    """Returns (labels, centers, inertia) of the best of `n_init` runs."""
    X = np.asarray(X, dtype=np.float64)
    n, dim = X.shape
    k = int(max(1, min(k, n)))
    xx = (X * X).sum(1)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = _kmeanspp(X, xx, k, rng)
        lab = None
        for _ in range(iters):
            D = xx[:, None] - 2.0 * X @ C.T + (C * C).sum(1)[None, :]
            new = D.argmin(1)
            if lab is not None and np.array_equal(new, lab):
                break
            lab = new
            counts = np.bincount(lab, minlength=k)
            sums = np.stack([np.bincount(lab, weights=X[:, j], minlength=k) for j in range(dim)], 1)
            C = sums / np.maximum(counts, 1)[:, None]
            empty = np.flatnonzero(counts == 0)
            if empty.size:                      # re-seed empty clusters at the worst-fit points
                worst = np.argsort(D[np.arange(n), lab])[::-1][:empty.size]
                C[empty] = X[worst]
        D = xx[:, None] - 2.0 * X @ C.T + (C * C).sum(1)[None, :]
        lab = D.argmin(1)
        inertia = float(np.maximum(D[np.arange(n), lab], 0).sum())
        if best is None or inertia < best[2]:
            best = (lab.copy(), C.copy(), inertia)
    return best
