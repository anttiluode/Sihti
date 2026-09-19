"""The slowest-dying modes of the self-EQ -- the objects.

For the normalized Laplacian  L = I - D^-1/2 W D^-1/2  the smallest eigenvalues
mu belong to the modes that survive diffusion longest (the lazy walk keeps a
mode with per-pass gain 1 - mu/2).  Mode 0 is trivial (constant); the next few
are close to piecewise constant on regions that hang together (normalized cuts,
Shi & Malik 2000).  They come out as an arbitrary rotation of the region
indicators -- the same indeterminacy as in BSS -- and a rotation that gives each
pixel one owner (NJW k-means on the unit-normalised rows) undoes it.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .cluster import kmeans


class Modes:
    def __init__(self, mu, U, d, H, W):
        self.mu = mu            # eigenvalues of L, ascending (mu[0] ~ 0)
        self.U = U              # eigenvectors of the symmetric normalized operator
        self.d = d              # degrees
        self.H, self.W = H, W

    @property
    def V(self):
        """Random-walk modes D^-1/2 U: these are the near piecewise-constant maps."""
        return self.U / np.sqrt(self.d)[:, None]

    def lifetimes(self, laziness=0.5):
        """Passes for each mode to fall to 1/e under the lazy walk."""
        g = np.clip(1.0 - laziness * self.mu, 1e-12, 1.0)
        with np.errstate(divide="ignore"):
            return np.where(g < 1.0, -1.0 / np.log(g), np.inf)


def spectral_modes(Wm, H, W, k=16, shift=-1e-3, seed=0):
    """k smallest eigenpairs of the normalized Laplacian (shift-invert Lanczos)."""
    Wm = Wm.tocsr()
    n = Wm.shape[0]
    d = np.asarray(Wm.sum(axis=1)).ravel()
    s = 1.0 / np.sqrt(d)
    S = sp.diags(s) @ Wm @ sp.diags(s)
    S = 0.5 * (S + S.T)
    L = (sp.identity(n, format="csr") - S).tocsc()
    k = int(min(k, n - 2))
    v0 = np.random.default_rng(seed).standard_normal(n)
    try:
        mu, U = eigsh(L, k=k, sigma=shift, which="LM", v0=v0, tol=1e-9, maxiter=5000)
    except Exception:
        # fallback for tiny graphs or solver trouble
        mu_all, U_all = np.linalg.eigh(L.toarray())
        mu, U = mu_all[:k], U_all[:, :k]
    order = np.argsort(mu)
    mu, U = mu[order], U[:, order]
    # deterministic signs: largest-magnitude entry positive
    idx = np.abs(U).argmax(axis=0)
    U = U * np.sign(U[idx, np.arange(U.shape[1])])[None, :]
    return Modes(mu, U, d, H, W)


def eigengap_k(mu, kmin=2, kmax=12):
    """k with the largest gap mu[k] - mu[k-1] (k clusters = k small eigenvalues)."""
    kmax = int(min(kmax, len(mu) - 1))
    if kmax < kmin:
        return kmin
    gaps = [mu[k] - mu[k - 1] for k in range(kmin, kmax + 1)]
    return int(kmin + int(np.argmax(gaps)))


def segments(modes, k, seed=0, n_init=4, use=None):
    """Ng-Jordan-Weiss discretisation: unit-normalise rows of k modes, k-means.

    `use` picks which modes (default: the first k, including the trivial one).
    """
    use = list(range(k)) if use is None else list(use)[:k]
    k = len(use)
    X = modes.U[:, use]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    lab, _, _ = kmeans(X, k, n_init=n_init, seed=seed)
    return lab.reshape(modes.H, modes.W)


def scene_modes(shares, tau=0.9):
    """Indices of modes that are NOT substrate (blank-frame share below tau).

    `shares[i]` is the share of mode i+1.  Mode 0 (trivial) is always kept first.
    """
    return [0] + [m for m, s in enumerate(shares, start=1) if np.isfinite(s) and s < tau]


def paint(modes, which=(1, 2, 3), lo=1.0, hi=99.0):
    which = list(which)[:3]
    """Paint three non-trivial modes as R, G, B (objects show up as flat patches)."""
    V = modes.V
    out = np.zeros((V.shape[0], 3))
    for c, m in enumerate(which):
        if m >= V.shape[1]:
            continue
        v = V[:, m]
        a, b = np.percentile(v, [lo, hi])
        out[:, c] = np.clip((v - a) / (b - a + 1e-12), 0.0, 1.0)
    return out.reshape(modes.H, modes.W, 3)


def align(U_new, U_ref):
    """Orthogonal Procrustes: rotate U_new's columns to best match U_ref (same grid).

    Used in live mode so the painted colours do not flicker between frames.  This
    is continuous frame matching -- the MovingProblem warning applies: after a
    closed path it can come back with the wrong orientation while every local
    check looks fine.  It only stabilises colours; nothing downstream relies on it.
    """
    M = U_new.T @ U_ref
    P, _, Qt = np.linalg.svd(M)
    return U_new @ (P @ Qt)


def lattice_basis(lattice_modes, count=12):
    """Orthonormal basis of the first `count` non-trivial blank-frame modes."""
    V = lattice_modes.V[:, 1:1 + count]
    Q, _ = np.linalg.qr(V)
    return Q


def substrate_share(modes, basis, which=(1, 2, 3)):
    """Fraction of each mode that the blank lattice's own modes already explain."""
    V = modes.V
    out = []
    for m in which:
        if m >= V.shape[1]:
            out.append(float("nan"))
            continue
        v = V[:, m] - V[:, m].mean()
        nv = float(v @ v)
        p = basis.T @ v
        out.append(float(p @ p / nv) if nv > 0 else float("nan"))
    return out
