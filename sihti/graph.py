"""The image's own EQ: a pixel graph whose couplings come from the image itself.

Every pixel is coupled to its neighbours within `radius`.  The coupling is

    w_ij = exp(-|lab_i - lab_j|^2 / (2 sigma^2)) * exp(-|p_i - p_j|^2 / (2 sigma_x^2))

so pixels that look alike share strongly, pixels across an edge barely share.
On a blank frame every colour distance is zero and the graph becomes the plain
lattice -- the substrate.  `GridGraph` builds the sparse *pattern* once per
(H, W, radius) and only refills the weights per frame, which is what makes the
live mode cheap.
"""
import numpy as np
import scipy.sparse as sp


def neighbor_offsets(radius):
    """Half-plane offsets (dy, dx) with dy^2 + dx^2 <= radius^2 (each pair once)."""
    r = int(np.floor(radius))
    offs = []
    for dy in range(0, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx <= 0:
                continue
            if dy * dy + dx * dx <= radius * radius + 1e-9:
                offs.append((dy, dx))
    return offs


class GridGraph:
    """Fixed neighbour pattern on an H x W grid.  Weights are supplied per image."""

    def __init__(self, H, W, radius=2.0):
        self.H, self.W, self.radius = int(H), int(W), float(radius)
        self.n = self.H * self.W
        idx = np.arange(self.n).reshape(self.H, self.W)
        ii, jj, s2 = [], [], []
        for dy, dx in neighbor_offsets(radius):
            y1 = self.H - dy
            x0, x1 = max(0, -dx), min(self.W, self.W - dx)
            if y1 <= 0 or x1 <= x0:
                continue
            a = idx[0:y1, x0:x1].ravel()
            b = idx[dy:dy + y1, x0 + dx:x1 + dx].ravel()
            ii.append(a)
            jj.append(b)
            s2.append(np.full(a.size, float(dy * dy + dx * dx)))
        self.i = np.concatenate(ii)
        self.j = np.concatenate(jj)
        self.s2 = np.concatenate(s2)
        self.E = self.i.size
        rows = np.concatenate([self.i, self.j])
        cols = np.concatenate([self.j, self.i])
        # Store (edge index + 1) as data so the CSR ordering can be recovered.
        pat = sp.coo_matrix((np.arange(2 * self.E, dtype=np.float64) + 1.0, (rows, cols)),
                            shape=(self.n, self.n)).tocsr()
        self._perm = (pat.data - 1.0).astype(np.int64)
        self._indptr = pat.indptr.copy()
        self._indices = pat.indices.copy()
        # nearest-neighbour (4-connected) edges, used for the colour-step statistic
        self._nn = self.s2 == 1.0

    # ------------------------------------------------------------------
    def weights(self, feat, sigma, sigma_x=None, floor=1e-6):
        """Edge weights for an image feature map `feat` (H, W, C)."""
        f = np.asarray(feat, dtype=np.float64).reshape(self.n, -1)
        d2 = ((f[self.i] - f[self.j]) ** 2).sum(axis=1)
        w = np.exp(-d2 / (2.0 * float(sigma) ** 2))
        if sigma_x is not None and sigma_x > 0:
            w = w * np.exp(-self.s2 / (2.0 * float(sigma_x) ** 2))
        return np.maximum(w, floor)

    def lattice_weights(self, sigma_x=None, floor=1e-6):
        """The blank-frame weights: colour plays no part -- only the grid remains."""
        w = np.ones(self.E)
        if sigma_x is not None and sigma_x > 0:
            w = w * np.exp(-self.s2 / (2.0 * float(sigma_x) ** 2))
        return np.maximum(w, floor)

    def matrix(self, w):
        """Symmetric sparse affinity matrix W from edge weights."""
        data = np.concatenate([w, w])[self._perm]
        return sp.csr_matrix((data, self._indices, self._indptr), shape=(self.n, self.n))

    def colour_steps(self, feat):
        """Distances between 4-connected neighbours (for choosing sigma)."""
        f = np.asarray(feat, dtype=np.float64).reshape(self.n, -1)
        a, b = self.i[self._nn], self.j[self._nn]
        return np.sqrt(((f[a] - f[b]) ** 2).sum(axis=1))


_CACHE = {}


def grid_graph(H, W, radius=2.0):
    """Cached GridGraph (the pattern only depends on shape and radius)."""
    key = (int(H), int(W), float(radius))
    g = _CACHE.get(key)
    if g is None:
        if len(_CACHE) > 8:
            _CACHE.clear()
        g = GridGraph(H, W, radius)
        _CACHE[key] = g
    return g


def self_affinity(lab, radius=2.0, sigma=8.0, sigma_x=None, floor=1e-6):
    """Affinity matrix of an image (given in Lab) with itself."""
    H, W = lab.shape[:2]
    g = grid_graph(H, W, radius)
    if sigma_x is None:
        sigma_x = max(1.0, radius)
    return g.matrix(g.weights(lab, sigma, sigma_x, floor))


def lattice_affinity(H, W, radius=2.0, sigma_x=None, floor=1e-6):
    """Affinity matrix of a blank frame: the substrate the self-EQ lives on."""
    g = grid_graph(H, W, radius)
    if sigma_x is None:
        sigma_x = max(1.0, radius)
    return g.matrix(g.lattice_weights(sigma_x, floor))
