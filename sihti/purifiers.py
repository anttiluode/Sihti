"""Purifiers: one pass of "purify" = one call  x -> step(x),  x of shape (H, W, C).

The sieve (sieve.py) does not care which purifier it is given.  What changes is
what *persistence* means:

  FourierEQ        Sigh's own operator (10-band gain curve indexed by normalized
                   k^2, exactly as in sigh_image_live_loop.py).  Persistence is
                   frequency.
  GraphDiffusion   averaging over the image's own graph (the self-EQ), or over
                   the blank lattice.  Persistence is grouping: pixels that
                   hang together equalise first, objects merge last.
  RebuiltSelfEQ    the self-EQ rebuilt from the *current* state every pass --
                   nonlinear, like an iterated bilateral filter.
  MedianPurifier   a plain nonlinear purifier (used to show the sum does not
                   care about linearity).
"""
import numpy as np
from scipy import ndimage

from .color import srgb_to_lab
from .graph import grid_graph

# Sigh's presets, copied from sigh_image_live_loop.py
SIGH_PRESETS = {
    "low pass": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005],
    "high pass": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    "band pass": [0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 0.8, 0.4, 0.2, 0.1],
    "notch": [1.0, 1.0, 1.0, 0.2, 0.05, 0.05, 0.2, 1.0, 1.0, 1.0],
    "all pass": [1.0] * 10,
}


def sigh_multiplier(H, W, gains, points=256):
    """Sigh's Fourier multiplier: gain curve over bands, looked up by normalized k^2."""
    ky = np.fft.fftfreq(H, d=1.0 / H)
    kx = np.fft.fftfreq(W, d=1.0 / W)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    k2 = k2 / k2.max()
    table = np.interp(np.linspace(0.0, 1.0, points),
                      np.linspace(0.0, 1.0, len(gains)), np.asarray(gains, float))
    idx = np.clip((k2 * (points - 1)).astype(np.int64), 0, points - 1)
    return table[idx]


class FourierEQ:
    """x -> ifft2( H(k) * fft2(x) ): Sigh's recursive filter, one pass."""
    kind = "fourier"

    def __init__(self, H, W, gains):
        self.gains = np.asarray(gains, float)
        self.M = sigh_multiplier(H, W, self.gains)

    def __call__(self, x):
        X = np.fft.fft2(x, axes=(0, 1))
        return np.fft.ifft2(X * self.M[..., None], axes=(0, 1)).real


class GraphDiffusion:
    """Lazy random walk on a graph:  x -> x + a (D^-1 W x - x).

    Each pass moves every pixel part of the way towards the weighted mean of the
    neighbours it is coupled to.  With a = 0.5 all eigenvalues lie in [0, 1], so
    nothing oscillates; with a = 1 on a 4-neighbour lattice the checkerboard
    (eigenvalue -1) survives forever -- Sigh's endpoint, reappearing as the
    lattice's own bipartite mode.
    """
    kind = "graph"

    def __init__(self, Wm, H, W, laziness=0.5):
        self.Wm = Wm.tocsr()
        self.H, self.W = int(H), int(W)
        self.d = np.asarray(self.Wm.sum(axis=1)).ravel()
        self.a = float(laziness)

    def __call__(self, x):
        n = self.H * self.W
        X = x.reshape(n, -1)
        Y = X + self.a * ((self.Wm @ X) / self.d[:, None] - X)
        return Y.reshape(x.shape)


class RebuiltSelfEQ:
    """Nonlinear purifier: rebuild the image's own graph from the current state each pass."""
    kind = "nonlinear"

    def __init__(self, H, W, radius=2.0, sigma=8.0, laziness=0.5):
        self.g = grid_graph(H, W, radius)
        self.H, self.W = int(H), int(W)
        self.sigma, self.a = float(sigma), float(laziness)
        self.sigma_x = max(1.0, float(radius))

    def __call__(self, x):
        lab = srgb_to_lab(np.clip(x, 0.0, 1.0))
        Wm = self.g.matrix(self.g.weights(lab, self.sigma, self.sigma_x))
        return GraphDiffusion(Wm, self.H, self.W, self.a)(x)


class MedianPurifier:
    """Nonlinear purifier: 3x3 median per channel."""
    kind = "nonlinear"

    def __call__(self, x):
        return ndimage.median_filter(x, size=(3, 3, 1), mode="nearest")
