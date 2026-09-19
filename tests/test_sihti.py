"""pytest -q"""
import numpy as np
import pytest

from sihti import (sift, octave_depths, GraphDiffusion, FourierEQ, MedianPurifier, RebuiltSelfEQ,
                   SIGH_PRESETS, self_affinity, lattice_affinity, srgb_to_lab, analyse,
                   grid_graph, within_fraction)
from sihti import scenes, metrics, baselines
from sihti.purifiers import sigh_multiplier


@pytest.fixture(scope="module")
def img():
    return np.random.default_rng(0).uniform(0, 1, (24, 32, 3))


def test_graph_symmetric_and_connected(img):
    Wm = self_affinity(srgb_to_lab(img), radius=2.0, sigma=6.0)
    assert abs(Wm - Wm.T).max() == 0
    assert (np.asarray(Wm.sum(1)).ravel() > 0).all()


@pytest.mark.parametrize("kind", ["self", "lattice", "sigh", "median", "rebuilt"])
def test_sum_is_lossless(img, kind):
    H, W = img.shape[:2]
    step = {"self": lambda: GraphDiffusion(self_affinity(srgb_to_lab(img), 2.0, 6.0), H, W),
            "lattice": lambda: GraphDiffusion(lattice_affinity(H, W, 2.0), H, W),
            "sigh": lambda: FourierEQ(H, W, SIGH_PRESETS["high pass"]),
            "median": lambda: MedianPurifier(),
            "rebuilt": lambda: RebuiltSelfEQ(H, W, 2.0, 6.0)}[kind]()
    res = sift(img, step, octave_depths(5))
    assert np.abs(res.reconstruct() - img).max() <= 1e-12


def test_fixed_fourier_eq_has_no_new_axis(img):
    H, W = img.shape[:2]
    gains = SIGH_PRESETS["band pass"]
    depths = octave_depths(5)
    res = sift(img, FourierEQ(H, W, gains), depths)
    g = np.linspace(-1, 2, res.channels)
    y = res.reconstruct(g, 0.3)
    Hm = sigh_multiplier(H, W, gains)
    G = 0.3 * Hm ** depths[-1] + sum(g[j] * (Hm ** depths[j] - Hm ** depths[j + 1])
                                     for j in range(res.channels))
    yf = np.fft.ifft2(np.fft.fft2(img, axes=(0, 1)) * G[..., None], axes=(0, 1)).real
    assert np.abs(y - yf).max() < 1e-9


def test_checkerboard_is_the_lattice_bipartite_mode():
    H, W = 12, 16
    g = grid_graph(H, W, 1.0)
    Wm = g.matrix(g.lattice_weights())
    yy, xx = np.mgrid[0:H, 0:W]
    c = ((-1.0) ** (yy + xx))[..., None]
    assert np.abs(GraphDiffusion(Wm, H, W, 1.0)(c) + c).max() < 1e-12
    assert np.abs(GraphDiffusion(Wm, H, W, 0.5)(c)).max() < 1e-12


def test_metrics_on_identity():
    gt = scenes.overlap()[1]
    assert metrics.covering(gt, gt) == pytest.approx(1.0)
    assert metrics.ari(gt, gt) == pytest.approx(1.0)
    assert metrics.vi(gt, gt) == pytest.approx(0.0, abs=1e-12)


def test_within_fraction_bounds():
    gt = scenes.twins()[1]
    flat = np.stack([gt.astype(float)] * 3, -1)
    assert within_fraction(flat, gt) == pytest.approx(0.0, abs=1e-12)
    noise = np.random.default_rng(1).standard_normal(flat.shape)
    assert 0.9 < within_fraction(noise, gt) <= 1.0


def test_twins_cohesion_beats_colour():
    x, gt = scenes.twins()
    out = analyse(x, radius=2.0, sigma=5.0, J=5, k=3)
    assert metrics.ari(out["segments"], gt) > 0.95
    assert metrics.ari(baselines.kmeans_lab_xy(out["lab"], 3, 0.0), gt) < 0.6


def test_shade_cohesion_beats_colour():
    x, gt = scenes.shade()
    out = analyse(x, radius=2.0, sigma=5.0, J=5, k=2)
    assert metrics.ari(out["segments"], gt) > 0.95
    assert metrics.ari(baselines.kmeans_lab_xy(out["lab"], 2, 0.0), gt) < 0.6
