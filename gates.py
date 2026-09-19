"""Sihti gates G0-G2 (sanity, theorem check, substrate).  Writes results/GATES.json.

    python gates.py                  # uses BSDS500 val images if --bsds points at them
    python gates.py --bsds path/to/BSDS500/data

G0  LOSSLESS      sum of residues + core == input, for linear, nonlinear, changing
                  and noisy purifiers.  An identity -- a SANITY check, not a result.
G1  NO NEW AXIS   for a fixed Fourier EQ (Sigh's operator) every persistence-slider
                  setting equals one frequency EQ, G(H(k)).  Confirms the correction:
                  persistence only becomes a new axis when the operator stops
                  being a fixed Fourier multiplier.
G2  SUBSTRATE     a blank frame turns the self-EQ into the plain lattice; its slow
                  modes are cosines; and Sigh's checkerboard is the lattice's own
                  bipartite mode (eigenvalue -1 under the non-lazy 4-neighbour walk).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from PIL import Image

from sihti import (resize_to, srgb_to_lab, sift, octave_depths, grid_graph,
                   self_affinity, lattice_affinity, GraphDiffusion, FourierEQ,
                   RebuiltSelfEQ, MedianPurifier, SIGH_PRESETS, spectral_modes)
from sihti import scenes
from sihti.purifiers import sigh_multiplier

TOL0 = 1e-12
TOL1 = 1e-9


def load_images(bsds, count=8, long_side=96):
    imgs = [(name, fn()[0]) for name, fn in scenes.SCENES.items()]
    rng = np.random.default_rng(0)
    imgs.append(("noise", rng.uniform(0, 1, (48, 64, 3))))
    if bsds:
        files = sorted(glob.glob(os.path.join(bsds, "images", "val", "*.jpg")))[:count]
        for f in files:
            x = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64) / 255.0
            imgs.append((os.path.basename(f), resize_to(x, long_side)))
    return imgs


class Noisy:
    """A deliberately bad purifier: the self-EQ plus fresh noise every pass."""
    def __init__(self, inner, sd=0.01, seed=0):
        self.inner, self.sd, self.rng = inner, sd, np.random.default_rng(seed)

    def __call__(self, x):
        return self.inner(x) + self.sd * self.rng.standard_normal(x.shape)


class Alternating:
    """A changing purifier: switch operator every pass (the order matters for the
    channels, never for the sum)."""
    def __init__(self, ops):
        self.ops, self.t = ops, 0

    def __call__(self, x):
        y = self.ops[self.t % len(self.ops)](x)
        self.t += 1
        return y


def g0(imgs):
    rows, worst = [], 0.0
    for name, x in imgs:
        H, W = x.shape[:2]
        lab = srgb_to_lab(x)
        Wm = self_affinity(lab, radius=2.0, sigma=6.0)
        purifiers = {
            "self": (GraphDiffusion(Wm, H, W), 9),
            "lattice": (GraphDiffusion(lattice_affinity(H, W, 2.0), H, W), 9),
            "rebuilt (nonlinear)": (RebuiltSelfEQ(H, W, 2.0, 6.0), 6),
            "median (nonlinear)": (MedianPurifier(), 6),
            "noisy self": (Noisy(GraphDiffusion(Wm, H, W)), 6),
            "alternating sigh EQs": (Alternating([FourierEQ(H, W, SIGH_PRESETS["low pass"]),
                                                  FourierEQ(H, W, SIGH_PRESETS["notch"])]), 7),
        }
        for pname, g in SIGH_PRESETS.items():
            purifiers[f"sigh {pname}"] = (FourierEQ(H, W, g), 9)
        for pname, (step, J) in purifiers.items():
            res = sift(x, step, octave_depths(J))
            err = float(np.abs(res.reconstruct() - x).max())
            worst = max(worst, err)
            rows.append({"image": name, "purifier": pname, "J": J, "max_abs_error": err})
    return {"gate": "G0 lossless", "tolerance": TOL0, "worst": worst,
            "pass": worst <= TOL0, "cases": len(rows), "rows": rows}


def g1(imgs):
    rng = np.random.default_rng(1)
    worst, cases = 0.0, 0
    for name, x in imgs[:9]:
        H, W = x.shape[:2]
        curves = [SIGH_PRESETS[p] for p in ("low pass", "high pass", "band pass", "notch")]
        curves += [list(rng.uniform(0.05, 1.0, 10)) for _ in range(3)]
        for gains in curves:
            J = 7
            depths = octave_depths(J)
            res = sift(x, FourierEQ(H, W, gains), depths)
            g = rng.uniform(-1.0, 2.0, res.channels)
            gc = float(rng.uniform(-1.0, 2.0))
            y = res.reconstruct(g, gc)
            Hm = sigh_multiplier(H, W, gains)
            G = gc * Hm ** depths[-1]
            for j in range(res.channels):
                G = G + g[j] * (Hm ** depths[j] - Hm ** depths[j + 1])
            yf = np.fft.ifft2(np.fft.fft2(x, axes=(0, 1)) * G[..., None], axes=(0, 1)).real
            worst = max(worst, float(np.abs(y - yf).max()))
            cases += 1
    # how many distinct gains can a persistence EQ reach on Sigh's own 64x64 field?
    Hm = sigh_multiplier(64, 64, SIGH_PRESETS["band pass"])
    ky, kx = np.meshgrid(np.fft.fftfreq(64) * 64, np.fft.fftfreq(64) * 64, indexing="ij")
    distinct_k2 = len(np.unique(np.round(ky ** 2 + kx ** 2, 6)))
    return {"gate": "G1 no new axis for a fixed Fourier EQ", "tolerance": TOL1,
            "worst": worst, "cases": cases, "pass": worst <= TOL1,
            "sigh_64x64": {"frequency_bins": 64 * 64, "distinct_k2": distinct_k2,
                           "distinct_gains_reachable": int(len(np.unique(Hm)))}}


def g2():
    H, W = 48, 64
    g = grid_graph(H, W, 2.0)
    blank = srgb_to_lab(np.full((H, W, 3), 0.5))
    w_self = g.weights(blank, sigma=6.0, sigma_x=2.0)
    w_lat = g.lattice_weights(sigma_x=2.0)
    same = bool(np.array_equal(w_self, w_lat))
    md = spectral_modes(lattice_affinity(H, W, 2.0), H, W, k=6)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx = np.cos(np.pi * (xx + 0.5) / W).ravel()
    cy = np.cos(np.pi * (yy + 0.5) / H).ravel()
    B = np.stack([cx - cx.mean(), cy - cy.mean()], 1)
    Q, _ = np.linalg.qr(B)
    corr = []
    for m in (1, 2):
        v = md.V[:, m] - md.V[:, m].mean()
        corr.append(float(np.linalg.norm(Q.T @ v) / np.linalg.norm(v)))
    # the checkerboard: non-lazy 4-neighbour walk
    g4 = grid_graph(H, W, 1.0)
    Wm = g4.matrix(g4.lattice_weights(sigma_x=None))
    c = ((-1.0) ** (yy + xx))[..., None]
    walk = GraphDiffusion(Wm, H, W, laziness=1.0)
    lazy = GraphDiffusion(Wm, H, W, laziness=0.5)
    flip = float(np.abs(walk(c) + c).max())
    after100 = c.copy()
    for _ in range(100):
        after100 = walk(after100)
    kept = float(np.abs(after100).mean())
    killed = float(np.abs(lazy(c)).max())
    ok = same and flip <= 1e-12 and killed <= 1e-12 and abs(kept - 1.0) <= 1e-12
    return {"gate": "G2 substrate", "pass": ok,
            "blank_self_equals_lattice": same,
            "lattice_modes_1_2_in_cosine_span": corr,
            "checkerboard": {"non_lazy_one_step_error_vs_minus_c": flip,
                             "non_lazy_mean_abs_after_100_passes": kept,
                             "lazy_one_step_max_abs": killed}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsds", default=None, help="path to BSDS500/data (optional)")
    ap.add_argument("--out", default="results/GATES.json")
    a = ap.parse_args()
    t = time.time()
    imgs = load_images(a.bsds)
    report = {"g0": g0(imgs), "g1": g1(imgs), "g2": g2()}
    report["seconds"] = round(time.time() - t, 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    for k in ("g0", "g1", "g2"):
        r = report[k]
        extra = {kk: vv for kk, vv in r.items() if kk not in ("rows", "gate", "pass")}
        print(f"[{'PASS' if r['pass'] else 'FAIL'}] {r['gate']}: {extra}")
    print(f"({report['seconds']} s) -> {a.out}")


if __name__ == "__main__":
    main()
