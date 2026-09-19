"""G3 / G4 on BSDS500 (Arbelaez, Maire, Fowlkes & Malik 2011) -- see PREREG.md.

    python bench_bsds.py --bsds path/to/BSDS500/data --phase val    # tune (ODS)
    python bench_bsds.py --bsds path/to/BSDS500/data --phase test   # score + decide

Every method is tuned on the VAL split (one setting for the whole split, chosen
by mean covering), then scored once on the TEST split.  Images and human
segmentations are reduced to 120 px on the long side for everyone.

G3  OBJECTS   does the self-EQ segment objects better than colour clustering
              (k-means on Lab + position) and than Felzenszwalb-Huttenlocher?
G4  WHERE DO THE OBJECTS GO?  object-alignment of the core and of the late
              residues, self-EQ vs the same diffusion on the blank lattice.
              alignment(X) = f(X, human segments rotated 180 deg) - f(X, human
              segments), f = within-segment energy fraction.  The rotated
              segmentation has the same region sizes in the wrong place, so a
              merely smooth field scores ~0 and only object-shaped fields score high.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from PIL import Image
import scipy.io as sio
from scipy.stats import binomtest, wilcoxon

from sihti import (resize_to, srgb_to_lab, self_affinity, lattice_affinity, GraphDiffusion,
                   FourierEQ, SIGH_PRESETS, sift, octave_depths, spectral_modes, within_fraction)
from sihti import modes as M
from sihti.pipeline import lattice_modes
from sihti.baselines import kmeans_lab_xy, felzenszwalb
from sihti.metrics import covering, score

LONG = 120
J = 9
N_MODES = 17
N_INIT = 3
TAU = 0.9

# ---- pre-registered grids (frozen with PREREG.md) -------------------------
RADII = [1.5, 3.0]
SIGMAS = [3.0, 5.0, 8.0]
K_NJW = [2, 3, 4, 6, 8, 12, "auto"]
K_KM = [2, 3, 4, 6, 8, 12, 16]
POS_CORE = [0.0, 20.0]
POS_KM = [0.0, 10.0, 20.0, 40.0]
FH_SCALE = [100.0, 300.0, 1000.0, 3000.0]
FH_MIN = [10, 30, 60]
FH_SIGMA = [0.5, 0.8]
LATE_START_DEPTH = 16          # G4: residues whose start depth >= 16 are "late"
LEAKY_SIGMA = 12.0             # G4 exploratory arm (report only)
# test images looked at during a 2-image debugging smoke run -> excluded from all test statistics
EXCLUDED_TEST = {"100007", "100039"}


def load_split(root, split, limit=None):
    items = []
    for f in sorted(glob.glob(os.path.join(root, "images", split, "*.jpg")))[:limit]:
        name = os.path.splitext(os.path.basename(f))[0]
        rgb = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64) / 255.0
        x = resize_to(rgb, LONG)
        h, w = x.shape[:2]
        m = sio.loadmat(os.path.join(root, "groundTruth", split, name + ".mat"))["groundTruth"]
        gts = []
        for i in range(m.shape[1]):
            s = m[0, i]["Segmentation"][0, 0].astype(np.int32)
            gts.append(np.asarray(Image.fromarray(s).resize((w, h), Image.Resampling.NEAREST)))
        items.append((name, x, gts))
    return items


def auto_k(md, use=None):
    mu = md.mu if use is None else md.mu[list(use)]
    return M.eigengap_k(mu, 2, min(12, len(mu) - 1))


def sihti_segmentations(x, lab, r, s):
    """All Sihti segmentations for one graph setting: {key: labels}."""
    H, W = x.shape[:2]
    Wm = self_affinity(lab, radius=r, sigma=s)
    md = spectral_modes(Wm, H, W, k=N_MODES)
    basis = M.lattice_basis(lattice_modes(H, W, r), 12)
    shares = M.substrate_share(md, basis, which=range(1, N_MODES))
    scene = M.scene_modes(shares, TAU)
    out = {}
    for k in K_NJW:
        kk = auto_k(md) if k == "auto" else k
        out[("njw", r, s, k)] = M.segments(md, kk, n_init=N_INIT)
        use = scene if k != "auto" else scene
        kk2 = auto_k(md, scene) if k == "auto" else k
        kk2 = max(2, min(kk2, len(scene)))
        out[("njw_scene", r, s, k)] = M.segments(md, kk2, n_init=N_INIT, use=use)
    core = sift(x, GraphDiffusion(Wm, H, W), octave_depths(J)).core
    core_lab = srgb_to_lab(np.clip(core, 0, 1))
    for k in K_KM:
        for p in POS_CORE:
            out[("core", r, s, k, p)] = kmeans_lab_xy(core_lab, k, p, n_init=N_INIT)
    return out


def baseline_segmentations(x, lab):
    out = {}
    for k in K_KM:
        for p in POS_KM:
            out[("kmeans", k, p)] = kmeans_lab_xy(lab, k, p, n_init=N_INIT)
    for sc in FH_SCALE:
        for mn in FH_MIN:
            for sg in FH_SIGMA:
                out[("fh", sc, mn, sg)] = felzenszwalb(x, sc, sg, mn)
    return out


def family(key):
    return key[0]


def keystr(key):
    return "|".join(str(v) for v in key)


def _load_cache(path):
    done = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                d = json.loads(line)
                done[d["image"]] = d
    return done


def phase_val(root, limit, out_path):
    items = load_split(root, "val", limit)
    cache_path = out_path.replace(".json", "_cache.jsonl")
    done = _load_cache(cache_path)
    t0 = time.time()
    with open(cache_path, "a") as fc:
        for n, (name, x, gts) in enumerate(items):
            if name in done:
                continue
            lab = srgb_to_lab(x)
            segs = baseline_segmentations(x, lab)
            for r in RADII:
                for s in SIGMAS:
                    segs.update(sihti_segmentations(x, lab, r, s))
            rec = {"image": name,
                   "cov": [[list(k), float(np.mean([covering(seg, g) for g in gts]))]
                           for k, seg in segs.items()]}
            fc.write(json.dumps(rec) + "\n")
            fc.flush()
            done[name] = rec
            print(f"val {len(done)}/{len(items)} {name}  {time.time() - t0:.0f}s", flush=True)
    names = [it[0] for it in items]
    totals = {}
    for nm in names:
        for k, v in done[nm]["cov"]:
            totals[tuple(k)] = totals.get(tuple(k), 0.0) + v
    count = len(names)
    means = {k: v / count for k, v in totals.items()}
    best = {}
    for key, v in means.items():
        f = family(key)
        if f not in best or v > best[f][1]:
            best[f] = (key, v)
    sihti_fams = ["njw", "njw_scene", "core"]
    primary = max(sihti_fams, key=lambda f: best[f][1])
    rep = {"images": count, "long_side": LONG,
           "best": {f: {"key": list(k), "val_covering": v} for f, (k, v) in best.items()},
           "primary_sihti_family": primary,
           "all": {keystr(k): v for k, v in sorted(means.items(), key=lambda kv: -kv[1])}}
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1)
    for f_, (k, v) in sorted(best.items(), key=lambda kv: -kv[1][1]):
        print(f"best {f_:10s} {v:.4f}  {k}")
    print("primary Sihti family:", primary)


def segment_with(key, x, lab, cache):
    """Compute exactly one segmentation for a tuned key (graph work cached per (r, sigma))."""
    f = key[0]
    if f == "kmeans":
        return kmeans_lab_xy(lab, int(key[1]), float(key[2]), n_init=N_INIT)
    if f == "fh":
        return felzenszwalb(x, float(key[1]), float(key[3]), int(key[2]))
    r, s = float(key[1]), float(key[2])
    H, W = x.shape[:2]
    if (r, s) not in cache:
        Wm = self_affinity(lab, radius=r, sigma=s)
        md = spectral_modes(Wm, H, W, k=N_MODES)
        basis = M.lattice_basis(lattice_modes(H, W, r), 12)
        shares = M.substrate_share(md, basis, which=range(1, N_MODES))
        cache[(r, s)] = {"W": Wm, "md": md, "scene": M.scene_modes(shares, TAU)}
    c = cache[(r, s)]
    if f == "njw":
        kk = auto_k(c["md"]) if key[3] == "auto" else int(key[3])
        return M.segments(c["md"], kk, n_init=N_INIT)
    if f == "njw_scene":
        kk = auto_k(c["md"], c["scene"]) if key[3] == "auto" else int(key[3])
        kk = max(2, min(kk, len(c["scene"])))
        return M.segments(c["md"], kk, n_init=N_INIT, use=c["scene"])
    if f == "core":
        if "core_lab" not in c:
            core = sift(x, GraphDiffusion(c["W"], H, W), octave_depths(J)).core
            c["core_lab"] = srgb_to_lab(np.clip(core, 0, 1))
        return kmeans_lab_xy(c["core_lab"], int(key[3]), float(key[4]), n_init=N_INIT)
    raise ValueError(key)


def alignment(X, gts):
    return float(np.mean([within_fraction(X, np.rot90(g, 2)) - within_fraction(X, g) for g in gts]))


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    wins, losses = int((a > b).sum()), int((a < b).sum())
    p_sign = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
    try:
        p_w = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p_w = 1.0
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "wins": wins,
            "losses": losses, "ties": int(len(a) - wins - losses),
            "p_sign": float(p_sign), "p_wilcoxon": p_w}


def phase_test(root, limit, val_path, out_path, per_image_path):
    val = json.load(open(val_path))
    best = {f: tuple(v["key"]) for f, v in val["best"].items()}
    primary = val["primary_sihti_family"]
    items = [it for it in load_split(root, "test", limit) if it[0] not in EXCLUDED_TEST]
    cache_path = out_path.replace(".json", "_cache.jsonl")
    done = _load_cache(cache_path)
    fc = open(cache_path, "a")
    t0 = time.time()
    pr = best[primary]
    rP, sP = float(pr[1]), float(pr[2])
    for n, (name, x, gts) in enumerate(items):
        if name in done:
            continue
        lab = srgb_to_lab(x)
        cache = {}
        row = {"image": name}
        for f, key in best.items():
            seg = segment_with(key, x, lab, cache)
            row[f] = score(seg, gts)
        # G4: where does object structure end up -- core or residues?
        H, W = x.shape[:2]
        arms = {"self": GraphDiffusion(self_affinity(lab, radius=rP, sigma=sP), H, W),
                "lattice": GraphDiffusion(lattice_affinity(H, W, rP), H, W),
                "sigh_low_pass": FourierEQ(H, W, SIGH_PRESETS["low pass"]),
                "self_leaky": GraphDiffusion(self_affinity(lab, radius=rP, sigma=LEAKY_SIGMA), H, W)}
        g4 = {"image": name}
        for arm, step in arms.items():
            res = sift(x, step, octave_depths(J))
            per = [alignment(R, gts) for R in res.residues]
            late = [per[j] for j in range(len(per)) if res.depths[j] >= LATE_START_DEPTH]
            g4[arm] = {"per_octave": per, "late": float(np.mean(late)), "core": alignment(res.core, gts)}
        done[name] = {"image": name, "row": row, "g4": g4}
        fc.write(json.dumps(done[name]) + "\n")
        fc.flush()
        print(f"test {len(done)}/{len(items)} {name}  {time.time() - t0:.0f}s", flush=True)
    fc.close()
    rows = [done[it[0]]["row"] for it in items]
    g4rows = [done[it[0]]["g4"] for it in items]

    fams = list(best)
    summary = {f: {m: float(np.mean([r[f][m] for r in rows])) for m in ("covering", "ari", "vi", "segments")}
               for f in fams}
    cov = {f: [r[f]["covering"] for r in rows] for f in fams}
    comparisons = {}
    for f in ("njw", "njw_scene", "core"):
        for b in ("kmeans", "fh"):
            comparisons[f"{f}_vs_{b}"] = paired(cov[f], cov[b])
    g3a = comparisons[f"{primary}_vs_kmeans"]
    g3b = comparisons[f"{primary}_vs_fh"]
    decide = lambda c: bool(c["mean_a"] > c["mean_b"] and c["wins"] > c["losses"] and c["p_sign"] < 0.01)

    def arr(arm, field):
        return np.array([g[arm][field] for g in g4rows])
    core_s, core_l = arr("self", "core"), arr("lattice", "core")
    late_s, late_l = arr("self", "late"), arr("lattice", "late")
    c_core = paired(core_s, core_l)                 # win = self core MORE object-aligned
    c_late = paired(-late_s, -late_l)               # win = self late residues LESS object-aligned
    frac_core = float((core_s > core_l).mean())
    frac_late = float((late_s < late_l).mean())
    g4a = {"pass": bool(core_s.mean() > core_l.mean() and frac_core >= 0.70 and c_core["p_sign"] < 0.01),
           "mean_self": float(core_s.mean()), "mean_lattice": float(core_l.mean()),
           "mean_sigh_low_pass": float(arr("sigh_low_pass", "core").mean()),
           "mean_self_leaky": float(arr("self_leaky", "core").mean()),
           "fraction_self_higher": frac_core,
           **{k: v for k, v in c_core.items() if k.startswith("p_") or k in ("wins", "losses", "ties")}}
    g4b = {"pass": bool(late_s.mean() < late_l.mean() and frac_late >= 0.70 and c_late["p_sign"] < 0.01),
           "mean_self": float(late_s.mean()), "mean_lattice": float(late_l.mean()),
           "mean_sigh_low_pass": float(arr("sigh_low_pass", "late").mean()),
           "mean_self_leaky": float(arr("self_leaky", "late").mean()),
           "fraction_self_lower": frac_late,
           **{k: v for k, v in c_late.items() if k.startswith("p_") or k in ("wins", "losses", "ties")}}
    octaves = len(g4rows[0]["self"]["per_octave"])
    curves = {arm: [float(np.mean([g[arm]["per_octave"][j] for g in g4rows])) for j in range(octaves)]
              for arm in ("self", "lattice", "sigh_low_pass", "self_leaky")}
    rep = {"images": len(rows), "excluded": sorted(EXCLUDED_TEST), "long_side": LONG,
           "primary_sihti_family": primary,
           "settings": {f: list(k) for f, k in best.items()},
           "summary": summary, "comparisons": comparisons,
           "G3a_objects_vs_kmeans": {"pass": decide(g3a), **g3a},
           "G3b_objects_vs_felzenszwalb": {"pass": decide(g3b), **g3b},
           "G4a_core_holds_objects": g4a,
           "G4b_residues_hold_insides": g4b,
           "G4_per_octave_alignment": curves,
           "octave_start_depths": octave_depths(J)[:-1],
           "g4_graph": {"radius": rP, "sigma": sP, "leaky_sigma": LEAKY_SIGMA}}
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1)
    with open(per_image_path, "w") as f:
        json.dump({"segmentation": rows, "residues": g4rows}, f)
    print(json.dumps({k: rep[k] for k in ("summary", "G3a_objects_vs_kmeans",
                                          "G3b_objects_vs_felzenszwalb")}, indent=1))
    print("G4a", rep["G4a_core_holds_objects"])
    print("G4b", rep["G4b_residues_hold_insides"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsds", required=True)
    ap.add_argument("--phase", choices=["val", "test"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    val_path = os.path.join(a.outdir, "BSDS_VAL.json")
    if a.phase == "val":
        phase_val(a.bsds, a.limit, val_path)
    else:
        phase_test(a.bsds, a.limit, val_path, os.path.join(a.outdir, "BSDS_TEST.json"),
                   os.path.join(a.outdir, "BSDS_TEST_PER_IMAGE.json"))


if __name__ == "__main__":
    main()
