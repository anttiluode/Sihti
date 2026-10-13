"""Eye persistence: when do the eyes stop being *these* eyes, and can Sihti bring them back cheaply?

Frozen in EYE_PREREG.md before any score was computed.

    python eye_persistence.py            # downloads dlib's annotated example faces (~1 MB)

Data: dlib's example faces with 68 hand-annotated landmarks (eyes = points 36-47);
dlib's own train/test split is kept. Images are cached in data/dlib_faces/ and not
committed.
"""
from __future__ import annotations

import io
import json
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from sihti.color import srgb_to_lab
from sihti.graph import lattice_affinity, self_affinity
from sihti.purifiers import GraphDiffusion
from sihti.sieve import octave_depths, sift

BASE = "https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/"
CACHE = Path("data/dlib_faces")
SIZE = 64
J = 9
QS = (1 / 16, 1 / 32, 1 / 64, 1 / 128)
LEARN_Q = 1 / 64
JPEG_Q = (2, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90)
PATCH_W, PATCH_H = 32, 16
TARGET_ACC = 23 / 25


# ---------------------------------------------------------------- data

def fetch(name: str) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        with urllib.request.urlopen(BASE + name, timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def crop_face(img: np.ndarray, box: dict, pts: np.ndarray):
    """Square crop 1.6x the box, reflect-padded, Lanczos to SIZE; landmarks follow."""
    top, left, w, h = (float(box[k]) for k in ("top", "left", "width", "height"))
    side = 1.6 * max(w, h)
    cx, cy = left + w / 2, top + h / 2
    pad = int(np.ceil(side))
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    x0, y0 = cx - side / 2 + pad, cy - side / 2 + pad
    crop = Image.fromarray(padded).crop((x0, y0, x0 + side, y0 + side))
    crop = crop.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    rgb = np.asarray(crop, dtype=np.float64) / 255.0
    scale = SIZE / side
    local = np.column_stack([(pts[:, 0] - (cx - side / 2)) * scale,
                             (pts[:, 1] - (cy - side / 2)) * scale])
    return rgb, local


def load_split(xml_name: str) -> list[dict]:
    root = ET.fromstring(fetch(xml_name))
    faces = []
    for im in root.find("images"):
        img = np.asarray(Image.open(io.BytesIO(fetch(im.get("file")))).convert("RGB"))
        for box in im.findall("box"):
            parts = sorted(box.findall("part"), key=lambda p: int(p.get("name")))
            pts = np.array([[float(p.get("x")), float(p.get("y"))] for p in parts])
            rgb, local = crop_face(img, box.attrib, pts)
            faces.append({"photo": im.get("file"), "rgb": rgb, "landmarks": local,
                          "box_px": float(max(float(box.get("width")),
                                              float(box.get("height"))))})
    return faces


# ---------------------------------------------------------------- readers

def luma(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def eye_patch(rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Both eyes, sampled to PATCH_W x PATCH_H luma, zero mean, unit norm."""
    p = landmarks[36:48]
    x0, x1 = p[:, 0].min(), p[:, 0].max()
    y0, y1 = p[:, 1].min(), p[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    width = (x1 - x0) * 1.4
    height = width / 2
    xs = cx - width / 2 + (np.arange(PATCH_W) + 0.5) * width / PATCH_W
    ys = cy - height / 2 + (np.arange(PATCH_H) + 0.5) * height / PATCH_H
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    v = ndimage.map_coordinates(luma(rgb), [yy.ravel(), xx.ravel()], order=1,
                                mode="nearest")
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def identify(patches: np.ndarray, gallery: np.ndarray):
    """Top-1 identification against the original patches; returns (accuracy, c_own)."""
    corr = patches @ gallery.T
    acc = float(np.mean(np.argmax(corr, axis=1) == np.arange(len(patches))))
    return acc, np.diag(corr).copy()


# ---------------------------------------------------------------- sieves and codecs

def sieve(rgb: np.ndarray, purifier: str):
    H, W = rgb.shape[:2]
    if purifier == "self":
        Wm = self_affinity(srgb_to_lab(rgb), radius=2.0, sigma=8.0)
    else:
        Wm = lattice_affinity(H, W, 2.0)
    return sift(rgb, GraphDiffusion(Wm, H, W, 0.5), octave_depths(J))


def quantise(arr: np.ndarray, q: float):
    z = np.round(arr / q).astype(np.int16)
    return 4 + len(zlib.compress(z.tobytes(), 9)), z.astype(np.float64) * q


_K = np.array([1, 4, 6, 4, 1], float) / 16


def _blur(x):
    for ax in (0, 1):
        x = ndimage.convolve1d(x, _K, axis=ax, mode="reflect")
    return x


def _reduce(x):
    return _blur(x)[::2, ::2]


def _expand(x, shape):
    up = np.zeros(shape)
    up[::2, ::2] = x
    return 4 * _blur(up)


def pyramid(rgb: np.ndarray, levels=4):
    g = [rgb]
    for _ in range(levels):
        g.append(_reduce(g[-1]))
    bands = [g[i] - _expand(g[i + 1], g[i].shape) for i in range(levels)]
    return g[-1], bands                      # top (4x4), band-pass fine..coarse


def pyramid_rebuild(top, bands):
    x = top
    for b in reversed(bands):
        x = _expand(x, b.shape) + b
    return x


def jpeg(rgb: np.ndarray, quality: int):
    buf = io.BytesIO()
    Image.fromarray(np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)).save(
        buf, format="JPEG", quality=quality, optimize=True)
    n = buf.tell()
    buf.seek(0)
    return n, np.asarray(Image.open(buf).convert("RGB"), float) / 255.0


# ---------------------------------------------------------------- experiment

def coded_sieve(res, q):
    nb, core = quantise(res.core, q)
    parts = [quantise(R, q) for R in res.residues]
    return nb, core, [p[0] for p in parts], [p[1] for p in parts]


def learn_order(train, train_sieves):
    """Greedy on train only: add the residue with the best mean c_own gain per byte."""
    gal = np.stack([f["patch"] for f in train])
    coded = [coded_sieve(r, LEARN_Q) for r in train_sieves]
    chosen, remaining = [], list(range(J + 1))
    cur = [c[1].copy() for c in coded]

    def mean_cown(recons):
        pats = np.stack([eye_patch(x, f["landmarks"]) for x, f in zip(recons, train)])
        return float(np.mean(np.sum(pats * gal, axis=1)))

    base = mean_cown(cur)
    trace = []
    while remaining:
        best = None
        for j in remaining:
            trial = [x + c[3][j] for x, c in zip(cur, coded)]
            gain = mean_cown(trial) - base
            cost = float(np.mean([c[2][j] for c in coded]))
            score = gain / cost
            if best is None or score > best[0]:
                best = (score, j, gain, cost)
        _, j, gain, cost = best
        chosen.append(j)
        remaining.remove(j)
        cur = [x + c[3][j] for x, c in zip(cur, coded)]
        base += gain
        trace.append({"residue": j, "train_c_own_after": round(base, 5),
                      "mean_bytes": round(cost, 1)})
    return chosen, trace


def evaluate(recons_by_config, test, gal, template):
    """recons_by_config: {label: (mean_bytes, [recon per face])}"""
    rows = []
    orig_generic = np.array([f["patch"] @ template for f in test])
    for label, (nb, recons) in recons_by_config.items():
        pats = np.stack([eye_patch(x, f["landmarks"]) for x, f in zip(recons, test)])
        acc, c_own = identify(pats, gal)
        generic = (pats @ template) / orig_generic
        rows.append({"config": label, "mean_bytes": float(nb), "accuracy": acc,
                     "mean_c_own": float(c_own.mean()),
                     "mean_generic_rel": float(np.mean(generic))})
    return rows


def envelope(rows):
    pts = sorted((r["mean_bytes"], r["accuracy"]) for r in rows)
    out, best = [], 0.0
    for b, a in pts:
        best = max(best, a)
        out.append((b, best))
    b92 = next((b for b, a in out if a >= TARGET_ACC - 1e-12), None)
    return out, b92


def run(out_dir=Path("results/eye")):
    train, test = load_split("training_with_face_landmarks.xml"), \
        load_split("testing_with_face_landmarks.xml")
    for f in train + test:
        f["patch"] = eye_patch(f["rgb"], f["landmarks"])
    gal = np.stack([f["patch"] for f in test])
    template = np.mean([f["patch"] for f in train], axis=0)
    template /= np.linalg.norm(template)
    depths = octave_depths(J)
    report = {"prereg": "EYE_PREREG.md", "faces": {"train": len(train), "test": len(test)},
              "face_box_px": [min(f["box_px"] for f in train + test),
                              max(f["box_px"] for f in train + test)],
              "depths": depths}

    # sanity: originals identify themselves
    report["original_identification"] = identify(gal, gal)[0]

    sieves = {}
    report["persistence"] = {}
    for pur in ("self", "lattice"):
        tr = [sieve(f["rgb"], pur) for f in train]
        te = [sieve(f["rgb"], pur) for f in test]
        sieves[pur] = (tr, te)
        lossless = max(float(np.abs(r.reconstruct() - f["rgb"]).max())
                       for r, f in zip(te, test))
        curve = []
        cown = np.zeros((len(test), len(depths)))
        for k, d in enumerate(depths):
            pats = np.stack([eye_patch(r.frames[d], f["landmarks"])
                             for r, f in zip(te, test)])
            acc, c = identify(pats, gal)
            cown[:, k] = c
            generic = np.mean((pats @ template) /
                              np.array([f["patch"] @ template for f in test]))
            curve.append({"depth": d, "accuracy": acc, "mean_c_own": float(c.mean()),
                          "mean_generic_rel": float(generic)})
        half = []
        for i in range(len(test)):
            below = [depths[k] for k in range(len(depths)) if cown[i, k] < 0.5]
            half.append(below[0] if below else None)
        report["persistence"][pur] = {"lossless_max_error": lossless, "curve": curve,
                                      "eye_half_life_per_face": half}
        print(pur, "lossless", f"{lossless:.1e}", "| acc by depth",
              [round(c["accuracy"], 2) for c in curve], flush=True)

    order, trace = learn_order(train, sieves["self"][0])
    report["learned_order"] = {"order": order, "trace": trace,
                               "natural_order": list(range(J, -1, -1))}
    print("learned order", order, flush=True)

    methods = {}
    for name, pur, seq in (("sihti_natural", "self", list(range(J, -1, -1))),
                           ("sihti_learned", "self", order),
                           ("sihti_lattice", "lattice", list(range(J, -1, -1)))):
        cfg = {}
        for q in QS:
            coded = [coded_sieve(r, q) for r in sieves[pur][1]]
            for k in range(J + 2):
                use = seq[:k]
                recons = [c[1] + sum((c[3][j] for j in use), np.zeros_like(c[1]))
                          for c in coded]
                nb = np.mean([c[0] + sum(c[2][j] for j in use) for c in coded])
                cfg[f"q=1/{round(1 / q)} k={k}"] = (nb, recons)
        methods[name] = evaluate(cfg, test, gal, template)

    cfg = {}
    for q in QS:
        per = []
        for f in test:
            top, bands = pyramid(f["rgb"])
            tb, tq = quantise(top, q)
            coded = [quantise(b, q) for b in bands]
            per.append((tb, tq, coded, bands))
        for k in range(5):
            recons, nbs = [], []
            for tb, tq, coded, bands in per:
                keep = set(range(4 - k, 4))          # k coarsest band-pass levels
                use = [coded[i][1] if i in keep else np.zeros_like(bands[i])
                       for i in range(4)]
                recons.append(pyramid_rebuild(tq, use))
                nbs.append(tb + sum(coded[i][0] for i in keep))
            cfg[f"q=1/{round(1 / q)} k={k}"] = (np.mean(nbs), recons)
    methods["laplacian_pyramid"] = evaluate(cfg, test, gal, template)

    cfg = {}
    for qual in JPEG_Q:
        coded = [jpeg(f["rgb"], qual) for f in test]
        cfg[f"quality={qual}"] = (np.mean([c[0] for c in coded]), [c[1] for c in coded])
    methods["jpeg"] = evaluate(cfg, test, gal, template)

    report["methods"] = methods
    b92 = {}
    for name, rows in methods.items():
        env, b = envelope(rows)
        b92[name] = b
        report.setdefault("envelopes", {})[name] = env
    report["B92_mean_bytes_per_face"] = b92

    def ratio(a, b):
        return None if a is None or b is None else a / b
    sihti_best = min([v for v in (b92["sihti_learned"], b92["sihti_natural"]) if v],
                     default=None)
    classical = min([v for v in (b92["jpeg"], b92["laplacian_pyramid"]) if v],
                    default=None)
    v1 = ratio(b92["sihti_learned"], b92["sihti_natural"])
    v2 = ratio(sihti_best, classical)
    imagining = [{"method": m, **r} for m, rows in methods.items() for r in rows
                 if r["mean_generic_rel"] >= 0.9 and r["accuracy"] < 0.5]
    report["verdicts"] = {
        "V1_learning_helps": {"ratio_learned_over_natural": v1,
                              "pass": v1 is not None and v1 <= 0.9},
        "V2_sihti_vs_classical": {"ratio_sihti_over_best_classical": v2,
                                  "reading": None if v2 is None else
                                  ("win" if v2 <= 0.8 else "lose" if v2 > 1.0 else "tie")},
        "V3_imagining_without_remembering": imagining}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eye_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("B92 bytes/face:", {k: (round(v) if v else None) for k, v in b92.items()})
    print("verdicts:", json.dumps(report["verdicts"], indent=1)[:1500])
    return report


if __name__ == "__main__":
    run()
