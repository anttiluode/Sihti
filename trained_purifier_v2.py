"""Trained purifier v2: equal-storage budget (TRAINED_PREREG_V2.md).

    python trained_purifier_v2.py

v1 matched removed L2 energy and was gamed by a brightness shift. Here every
purifier must reach the fixed own EQ's compressed size (zlib-9, q = 1/64), and
the verdict adds a raw (unnormalised) eye-patch fidelity reader.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

import eye_persistence as E
import trained_purifier as T

MAX_STEPS = 512
Q = 1 / 64


def qbytes(x):
    return E.quantise(x.reshape(T.N, T.N, 3), Q)[0]


def raw_eye_patch(rgb, f):
    p = f["landmarks"][36:48]
    x0, x1 = p[:, 0].min(), p[:, 0].max()
    cx, cy = (x0 + x1) / 2, (p[:, 1].min() + p[:, 1].max()) / 2
    w = (x1 - x0) * 1.4
    xs = cx - w / 2 + (np.arange(E.PATCH_W) + 0.5) * w / E.PATCH_W
    ys = cy - w / 4 + (np.arange(E.PATCH_H) + 0.5) * (w / 2) / E.PATCH_H
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return ndimage.map_coordinates(E.luma(rgb), [yy.ravel(), xx.ravel()], order=1,
                                   mode="nearest")


def prepare(faces):
    for fc in faces:
        fc.target_bytes = {d: qbytes(T.run_steps(fc, T.BASE_T, T.BASE_U, d))
                           for d in T.DEPTHS}
        fc.raw_eye = raw_eye_patch(fc.f["rgb"], fc.f)


def run_bytes(fc, t, u, depth):
    """Purify until the quantised state compresses to <= the fixed EQ's size."""
    step = T.step_fn(*T.matrix(fc, t, u))
    x = fc.x0.copy()
    for n in range(1, MAX_STEPS + 1):
        x = step(x)
        if qbytes(x) <= fc.target_bytes[depth]:
            return x, n, True
    return x, MAX_STEPS, False


def train(faces, which):
    calls = {"n": 0}

    def loss(p):
        calls["n"] += 1
        xs = [run_bytes(fc, p[:6], p[6:], T.TRAIN_DEPTH)[0] for fc in faces]
        return -float(T.score(faces, xs, which)[1].mean())

    p0 = np.concatenate([T.BASE_T, T.BASE_U])
    start = -loss(p0)
    res = minimize(loss, p0, method="Powell", bounds=T.BOUNDS,
                   options={"maxfev": T.MAXFEV, "xtol": 1e-3, "ftol": 1e-6})
    return res.x, {"train_c_own_start": start, "train_c_own_end": -float(res.fun),
                   "evaluations": calls["n"], "message": str(res.message)}


def run(out=Path("results/trained/trained_v2_results.json")):
    train_raw = E.load_split("training_with_face_landmarks.xml")
    test_raw = E.load_split("testing_with_face_landmarks.xml")
    allphi = np.concatenate([T.features(f["rgb"]) for f in train_raw])
    mean, std = allphi.mean(axis=0), allphi.std(axis=0) + 1e-9
    tr = [T.Face(f, mean, std) for f in train_raw]
    te = [T.Face(f, mean, std) for f in test_raw]
    prepare(tr)
    prepare(te)

    purifiers = {"fixed_own_eq": np.concatenate([T.BASE_T, T.BASE_U])}
    training = {}
    for which in ("eye", "mouth"):
        p, info = train(tr, which)
        purifiers[f"trained_{which}"] = p
        training[which] = {**info, **T.describe(p)}
        print(f"trained on {which}: train c_own {info['train_c_own_start']:.4f} -> "
              f"{info['train_c_own_end']:.4f} ({info['evaluations']} evals)", flush=True)

    results = {}
    for name, p in purifiers.items():
        results[name] = {}
        for d in T.DEPTHS:
            ss = [run_bytes(fc, p[:6], p[6:], d) for fc in te]
            xs = [s[0] for s in ss]
            row = {"steps_mean": float(np.mean([s[1] for s in ss])),
                   "unreached": int(sum(not s[2] for s in ss))}
            for which in ("eye", "mouth"):
                acc, c = T.score(te, xs, which)
                row[which] = {"accuracy": acc, "correct": int(round(acc * len(te))),
                              "mean_c_own": float(c.mean()), "c_own": c.tolist()}
            row["raw_eye_rmse"] = [float(np.sqrt(np.mean(
                (raw_eye_patch(x.reshape(T.N, T.N, 3), fc.f) - fc.raw_eye) ** 2)))
                for fc, x in zip(te, xs)]
            row["energy_removed_mean"] = float(np.mean(
                [((fc.x0 - x) ** 2).sum() for fc, x in zip(te, xs)]))
            row["bytes_ratio_mean"] = float(np.mean(
                [qbytes(x) / fc.target_bytes[d] for fc, x in zip(te, xs)]))
            row["eye_box_energy_fraction"] = float(np.mean(
                [((fc.x0 - x) ** 2)[T.eye_box_mask(fc.f).ravel()].sum() /
                 max(((fc.x0 - x) ** 2).sum(), 1e-12) for fc, x in zip(te, xs)]))
            results[name][str(d)] = row
        print(name, {d: (results[name][str(d)]["eye"]["correct"],
                         results[name][str(d)]["mouth"]["correct"],
                         round(results[name][str(d)]["steps_mean"], 1),
                         results[name][str(d)]["unreached"]) for d in T.DEPTHS}, flush=True)

    def compare(a, b, which, d):
        ra, rb = results[a][str(d)], results[b][str(d)]
        ca, cb = np.array(ra[which]["c_own"]), np.array(rb[which]["c_own"])
        k = int((ca > cb).sum())
        out = {"faces_higher_c_own": k, "of": len(ca),
               "one_sided_sign_p": T.sign_p(k, len(ca)),
               "identification_gain": ra[which]["correct"] - rb[which]["correct"],
               "mean_c_own_gain": float(ca.mean() - cb.mean())}
        if which == "eye":
            ea, eb = np.array(ra["raw_eye_rmse"]), np.array(rb["raw_eye_rmse"])
            out["faces_lower_raw_rmse"] = int((ea < eb).sum())
            out["mean_raw_rmse"] = [float(ea.mean()), float(eb.mean())]
        return out

    p1 = compare("trained_eye", "fixed_own_eq", "eye", 32)
    dd = {"eye_gain_by_eye_trained": compare("trained_eye", "fixed_own_eq", "eye", 32),
          "eye_gain_by_mouth_trained": compare("trained_mouth", "fixed_own_eq", "eye", 32),
          "mouth_gain_by_mouth_trained": compare("trained_mouth", "fixed_own_eq", "mouth", 32),
          "mouth_gain_by_eye_trained": compare("trained_eye", "fixed_own_eq", "mouth", 32)}
    gap_eye = (dd["eye_gain_by_eye_trained"]["mean_c_own_gain"] -
               dd["eye_gain_by_mouth_trained"]["mean_c_own_gain"])
    gap_mouth = (dd["mouth_gain_by_mouth_trained"]["mean_c_own_gain"] -
                 dd["mouth_gain_by_eye_trained"]["mean_c_own_gain"])
    dd["gaps"] = {"eye": gap_eye, "mouth": gap_mouth,
                  "specific": gap_eye >= 0.02 and gap_mouth >= 0.02}
    verdict = {
        "P1_eye_trained_vs_fixed_d32": {
            **p1, "pass": p1["identification_gain"] >= 4 and p1["faces_higher_c_own"] >= 18
            and p1["faces_lower_raw_rmse"] >= 18},
        "P2_double_dissociation_d32": dd,
        "P3_depth_transfer": {str(d): compare("trained_eye", "fixed_own_eq", "eye", d)
                              for d in T.DEPTHS}}
    report = {"prereg": "TRAINED_PREREG_V2.md", "budget": "equal zlib-9 bytes at q=1/64",
              "training": training, "results": results, "verdicts": verdict}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(verdict, indent=1)[:2500])
    return report


if __name__ == "__main__":
    run()
