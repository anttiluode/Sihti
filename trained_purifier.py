"""Train the purifier's coupling rules so a chosen feature survives purification.

Frozen in TRAINED_PREREG.md before any score was computed.

    python trained_purifier.py

Every purifier (fixed or trained) removes exactly the energy the fixed own EQ
removes at depth d, so learning can only move WHERE the smoothing happens.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

import eye_persistence as E
from sihti.color import srgb_to_lab
from sihti.graph import grid_graph
from sihti.sieve import sift

N = E.SIZE
G = grid_graph(N, N, 2.0)
SPATIAL = np.exp(-G.s2 / 8.0)
FEATURES = ("L", "a", "b", "grad", "local_std", "centre_surround")
BASE_T = np.array([np.log(1 / 128)] * 3 + [np.log(1e-6)] * 3)
BASE_U = np.zeros(6)
BOUNDS = [(-16.0, 2.0)] * 6 + [(-3.0, 3.0)] * 6
MAXFEV = 800
DEPTHS = (16, 32, 64)
TRAIN_DEPTH = 32
MAX_STEPS = 4096


# ---------------------------------------------------------------- features / readers

def features(rgb):
    lab = srgb_to_lab(np.clip(rgb, 0, 1))
    L = lab[..., 0]
    grad = np.hypot(ndimage.sobel(L, 0), ndimage.sobel(L, 1)) / 8.0
    m3 = ndimage.uniform_filter(L, 3)
    std = np.sqrt(np.maximum(ndimage.uniform_filter(L * L, 3) - m3 * m3, 0))
    cs = L - ndimage.uniform_filter(L, 5)
    return np.concatenate([lab, grad[..., None], std[..., None], cs[..., None]],
                          axis=-1).reshape(-1, 6)


def region_patch(rgb, landmarks, idx, widen):
    p = landmarks[idx]
    x0, x1 = p[:, 0].min(), p[:, 0].max()
    y0, y1 = p[:, 1].min(), p[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    width = (x1 - x0) * widen
    height = width / 2
    xs = cx - width / 2 + (np.arange(E.PATCH_W) + 0.5) * width / E.PATCH_W
    ys = cy - height / 2 + (np.arange(E.PATCH_H) + 0.5) * height / E.PATCH_H
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    v = ndimage.map_coordinates(E.luma(rgb), [yy.ravel(), xx.ravel()], order=1,
                                mode="nearest")
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


READERS = {"eye": (slice(36, 48), 1.4), "mouth": (slice(48, 68), 1.3)}


def patch(rgb, f, which):
    idx, widen = READERS[which]
    return region_patch(rgb, f["landmarks"], idx, widen)


def eye_box_mask(f):
    p = f["landmarks"][36:48]
    x0, x1 = p[:, 0].min(), p[:, 0].max()
    cx, cy = (x0 + x1) / 2, (p[:, 1].min() + p[:, 1].max()) / 2
    w = (x1 - x0) * 1.4
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    return (np.abs(xx - cx) <= w / 2) & (np.abs(yy - cy) <= w / 4)


# ---------------------------------------------------------------- purifier

class Face:
    def __init__(self, f, mean, std):
        self.f = f
        self.x0 = f["rgb"].reshape(-1, 3)
        phi = features(f["rgb"])
        self.D = (phi[G.i] - phi[G.j]) ** 2              # (E, 6)
        self.phihat = (phi - mean) / std
        self.target = {}
        for d in DEPTHS:
            x = run_steps(self, BASE_T, BASE_U, d)
            self.target[d] = float(((self.x0 - x) ** 2).sum())


def matrix(face, t, u):
    w = np.exp(-(face.D @ np.exp(t))) * SPATIAL
    g = np.exp(np.clip(face.phihat @ u, -3, 3))
    w = np.maximum(w * g[G.i] * g[G.j], 1e-6)
    Wm = G.matrix(w)
    return Wm, np.asarray(Wm.sum(axis=1)).ravel()


def step_fn(Wm, deg):
    return lambda X: X + 0.5 * ((Wm @ X) / deg[:, None] - X)


def run_steps(face, t, u, n):
    step = step_fn(*matrix(face, t, u))
    x = face.x0.copy()
    for _ in range(n):
        x = step(x)
    return x


def run_matched(face, t, u, depth):
    """Purify until the removed energy equals the fixed own EQ's at `depth`."""
    target = face.target[depth]
    step = step_fn(*matrix(face, t, u))
    x, e = face.x0.copy(), 0.0
    for n in range(1, MAX_STEPS + 1):
        xn = step(x)
        en = float(((face.x0 - xn) ** 2).sum())
        if en >= target:
            # removed energy is quadratic along the segment: solve it exactly
            r0, dlt = face.x0 - x, xn - x
            b, c = float((r0 * dlt).sum()), float((dlt * dlt).sum())
            disc = max(b * b - c * (e - target), 0.0)
            a = (b + np.sqrt(disc)) / c if c > 0 else 1.0
            return x + min(max(a, 0.0), 1.0) * (xn - x), n
        x, e = xn, en
    return x, MAX_STEPS


def states(faces, t, u, depth):
    return [run_matched(fc, t, u, depth) for fc in faces]


def score(faces, xs, which, gallery=None):
    pats = np.stack([patch(x.reshape(N, N, 3), fc.f, which) for x, fc in zip(xs, faces)])
    if gallery is None:
        gallery = np.stack([patch(fc.f["rgb"], fc.f, which) for fc in faces])
    acc, c_own = E.identify(pats, gallery)
    return acc, c_own


# ---------------------------------------------------------------- experiment

def train(faces, which):
    calls = {"n": 0}

    def loss(p):
        calls["n"] += 1
        xs = [s[0] for s in states(faces, p[:6], p[6:], TRAIN_DEPTH)]
        return -float(score(faces, xs, which)[1].mean())

    p0 = np.concatenate([BASE_T, BASE_U])
    start = -loss(p0)
    res = minimize(loss, p0, method="Powell", bounds=BOUNDS,
                   options={"maxfev": MAXFEV, "xtol": 1e-3, "ftol": 1e-6})
    return res.x, {"train_c_own_start": start, "train_c_own_end": -float(res.fun),
                   "evaluations": calls["n"], "message": str(res.message)}


def describe(p):
    return {"theta": dict(zip(FEATURES, np.exp(p[:6]).round(6).tolist())),
            "gate_u": dict(zip(FEATURES, p[6:].round(4).tolist()))}


def sign_p(k, n):
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def run(out=Path("results/trained/trained_results.json")):
    train_raw = E.load_split("training_with_face_landmarks.xml")
    test_raw = E.load_split("testing_with_face_landmarks.xml")
    allphi = np.concatenate([features(f["rgb"]) for f in train_raw])
    mean, std = allphi.mean(axis=0), allphi.std(axis=0) + 1e-9
    tr = [Face(f, mean, std) for f in train_raw]
    te = [Face(f, mean, std) for f in test_raw]

    # the fixed own EQ, re-run through run_matched, must land on its own depth state
    chk = max(float(np.abs(run_matched(fc, BASE_T, BASE_U, 32)[0] -
                           run_steps(fc, BASE_T, BASE_U, 32)).max()) for fc in te[:5])

    purifiers = {"fixed_own_eq": np.concatenate([BASE_T, BASE_U])}
    training = {}
    for which in ("eye", "mouth"):
        p, info = train(tr, which)
        purifiers[f"trained_{which}"] = p
        training[which] = {**info, **describe(p)}
        print(f"trained on {which}: train c_own {info['train_c_own_start']:.4f} -> "
              f"{info['train_c_own_end']:.4f} ({info['evaluations']} evals)", flush=True)

    results = {}
    for name, p in purifiers.items():
        results[name] = {}
        for d in DEPTHS:
            ss = states(te, p[:6], p[6:], d)
            xs = [s[0] for s in ss]
            row = {"steps_mean": float(np.mean([s[1] for s in ss]))}
            for which in ("eye", "mouth"):
                acc, c = score(te, xs, which)
                row[which] = {"accuracy": acc, "correct": int(round(acc * len(te))),
                              "mean_c_own": float(c.mean()), "c_own": c.tolist()}
            energy = [float(((fc.x0 - x) ** 2).sum()) / fc.target[d] for fc, x in zip(te, xs)]
            row["energy_ratio_min_max"] = [min(energy), max(energy)]
            inside = [float(((fc.x0 - x) ** 2)[eye_box_mask(fc.f).ravel()].sum() /
                            ((fc.x0 - x) ** 2).sum()) for fc, x in zip(te, xs)]
            row["eye_box_energy_fraction"] = float(np.mean(inside))
            row["bytes_q64"] = float(np.mean([E.quantise(x.reshape(N, N, 3), 1 / 64)[0]
                                              for x in xs]))
            results[name][str(d)] = row
        print(name, {d: (results[name][str(d)]["eye"]["correct"],
                         results[name][str(d)]["mouth"]["correct"]) for d in DEPTHS},
              flush=True)

    # lossless: the trained purifier's residues still sum to the input
    p = purifiers["trained_eye"]
    fc = te[0]
    res = sift(fc.f["rgb"], lambda X: step_fn(*matrix(fc, p[:6], p[6:]))(
        X.reshape(-1, 3)).reshape(N, N, 3), E.octave_depths(5))
    lossless = float(np.abs(res.reconstruct() - fc.f["rgb"]).max())

    def compare(a, b, which, d):
        ca = np.array(results[a][str(d)][which]["c_own"])
        cb = np.array(results[b][str(d)][which]["c_own"])
        k = int((ca > cb).sum())
        return {"faces_higher": k, "of": len(ca), "one_sided_sign_p": sign_p(k, len(ca)),
                "identification_gain": results[a][str(d)][which]["correct"] -
                results[b][str(d)][which]["correct"],
                "mean_c_own_gain": float(ca.mean() - cb.mean())}

    p1 = compare("trained_eye", "fixed_own_eq", "eye", 32)
    verdict = {
        "P1_eye_trained_vs_fixed_d32": {**p1, "pass": p1["identification_gain"] >= 4
                                        and p1["faces_higher"] >= 18},
        "P2_double_dissociation_d32": {
            "eye_gain_by_eye_trained": compare("trained_eye", "fixed_own_eq", "eye", 32),
            "eye_gain_by_mouth_trained": compare("trained_mouth", "fixed_own_eq", "eye", 32),
            "mouth_gain_by_mouth_trained": compare("trained_mouth", "fixed_own_eq", "mouth", 32),
            "mouth_gain_by_eye_trained": compare("trained_eye", "fixed_own_eq", "mouth", 32)},
        "P3_depth_transfer": {str(d): compare("trained_eye", "fixed_own_eq", "eye", d)
                              for d in DEPTHS},
    }
    dd = verdict["P2_double_dissociation_d32"]
    dd["holds"] = (dd["eye_gain_by_eye_trained"]["mean_c_own_gain"] >
                   dd["eye_gain_by_mouth_trained"]["mean_c_own_gain"] and
                   dd["mouth_gain_by_mouth_trained"]["mean_c_own_gain"] >
                   dd["mouth_gain_by_eye_trained"]["mean_c_own_gain"])
    report = {"prereg": "TRAINED_PREREG.md", "train_faces": len(tr), "test_faces": len(te),
              "matched_run_consistency_max_error": chk, "lossless_trained_sieve": lossless,
              "training": training, "results": results, "verdicts": verdict}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(verdict, indent=1)[:3000])
    return report


if __name__ == "__main__":
    run()
