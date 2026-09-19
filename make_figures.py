"""Figures for the README.   python make_figures.py --bsds path/to/BSDS500/data

Uses VAL images only (test images are reserved for the gates).
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from sihti import (analyse, resize_to, srgb_to_lab, self_affinity, lattice_affinity, GraphDiffusion,
                   FourierEQ, SIGH_PRESETS, sift, octave_depths, log_depths)
from sihti import scenes as SC, render as R, metrics
from sihti.baselines import kmeans_lab_xy, felzenszwalb

SIGMA, RADIUS = 5.0, 1.5


def load(bsds, name, long_side=96):
    p = os.path.join(bsds, "images", "val", name + ".jpg")
    return resize_to(np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0, long_side)


def hero(x, path):
    out = analyse(x, radius=RADIUS, sigma=SIGMA, J=9, k="auto")
    res = out["sieve"]
    seg = out["segments"]
    row1 = [x, res.frame(8)[0], res.frame(64)[0], res.core, out["painted"],
            R.overlay_boundaries(R.label_colors(seg, x), seg, (0.93, 0.9, 0.8))]
    picks = [0, 3, 5, 7, 9]
    row2 = [R.signed(res.residues[j])[0] for j in picks] + [res.core]
    names = [f"{res.depths[j]}\u2192{res.depths[j + 1]}" for j in picks]
    R.grid([row1, row2], titles=["input", "loop, depth 8", "depth 64", "depth 512 (core)",
                                 "slow modes as colour", f"{out['k']} segments"], scale=3,
           caption="bottom: what bled out between depths " + ", ".join(names) + ", and the core").save(path)


def where(x, path):
    H, W = x.shape[:2]
    lab = srgb_to_lab(x)
    arms = [("own EQ", GraphDiffusion(self_affinity(lab, RADIUS, SIGMA), H, W)),
            ("blank lattice", GraphDiffusion(lattice_affinity(H, W, RADIUS), H, W)),
            ("Sigh low pass", FourierEQ(H, W, SIGH_PRESETS["low pass"]))]
    rows, picks = [], [0, 3, 5, 7, 9]
    for _, step in arms:
        res = sift(x, step, octave_depths(9))
        rows.append([R.signed(res.residues[j])[0] for j in picks] + [np.clip(res.core, 0, 1)])
    R.grid(rows, titles=["0\u21921", "4\u21928", "16\u219232", "64\u2192128", "256\u2192512", "core"],
           row_labels=[a for a, _ in arms], label_w=118, scale=3,
           caption="same image, three purifiers: the own EQ keeps the statues in the core; the lattice bleeds them out").save(path)


def scene_table(path):
    rows, lines = [], []
    for name, fn in SC.SCENES.items():
        x, gt = fn()
        k = int(gt.max() + 1)
        out = analyse(x, radius=RADIUS, sigma=SIGMA, J=6, k=k)
        km = kmeans_lab_xy(out["lab"], k, 20.0)
        fh = felzenszwalb(x, 300.0, 0.5, 30)
        segs = {"Sihti": out["segments"], "k-means Lab+xy": km, "Felzenszwalb": fh}
        row = [x, R.label_colors(gt), out["painted"]] + [
            R.overlay_boundaries(R.label_colors(s, x), s, (1, 1, 1)) for s in segs.values()]
        rows.append(row)
        lines.append({"scene": name, **{m: round(metrics.ari(s, gt), 3) for m, s in segs.items()}})
    R.grid(rows, titles=["scene", "truth", "slow modes", "Sihti", "k-means Lab+xy", "Felzenszwalb"],
           row_labels=list(SC.SCENES), label_w=78, scale=2).save(path)
    return lines


def movie(x, path):
    H, W = x.shape[:2]
    own = sift(x, GraphDiffusion(self_affinity(srgb_to_lab(x), RADIUS, SIGMA), H, W),
               octave_depths(9), frame_depths=log_depths(512, 40))
    sigh = sift(x, FourierEQ(H, W, SIGH_PRESETS["high pass"]), octave_depths(9),
                frame_depths=log_depths(512, 40))
    frames = []
    for d in sorted(own.frames):
        b = sigh.frames[d]
        b = (b - b.min()) / (b.max() - b.min() + 1e-12)
        tile = np.concatenate([own.frames[d], np.full((H, 4, 3), 0.23), b], axis=1)
        im = R.upscale(R.to_u8(tile), 3)
        frames.append(im)
    frames += [frames[-1]] * 12
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=110, loop=0)


def bench_plot(test_json, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = json.load(open(test_json))
    curves = rep["G4_per_octave_alignment"]
    depths = rep["octave_start_depths"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), dpi=110)
    fig.patch.set_facecolor("#3A3C3F")
    for a in ax:
        a.set_facecolor("#323437")
        a.tick_params(colors="#E6E4DF")
        for sp in a.spines.values():
            sp.set_color("#6A6D71")
    style = {"self": ("#D9C9A0", "own EQ"), "lattice": ("#8FB8CC", "blank lattice"),
             "sigh_low_pass": ("#C98F8F", "Sigh low pass"), "self_leaky": ("#A7C79A", "own EQ, leaky (\u03c3=12)")}
    for arm, (c, lab) in style.items():
        ax[0].plot(range(len(depths)), curves[arm], "-o", color=c, label=lab, ms=4)
    ax[0].set_xticks(range(len(depths)))
    ax[0].set_xticklabels([str(d) for d in depths], color="#E6E4DF")
    ax[0].set_xlabel("residue starting at depth", color="#E6E4DF")
    ax[0].set_ylabel("object alignment", color="#E6E4DF")
    ax[0].set_title("what bleeds out: residues", color="#E6E4DF")
    ax[0].legend(facecolor="#323437", labelcolor="#E6E4DF", fontsize=8)
    g4a = rep["G4a_core_holds_objects"]
    vals = [g4a["mean_self"], g4a["mean_self_leaky"], g4a["mean_sigh_low_pass"], g4a["mean_lattice"]]
    ax[1].bar(range(4), vals, color=["#D9C9A0", "#A7C79A", "#C98F8F", "#8FB8CC"])
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(["own EQ", "leaky", "Sigh LP", "lattice"], color="#E6E4DF")
    ax[1].set_title("what survives: the core", color="#E6E4DF")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsds", required=True)
    ap.add_argument("--only", default="all")
    a = ap.parse_args()
    os.makedirs("figures", exist_ok=True)
    if a.only in ("all", "images"):
        statues = load(a.bsds, "101085")
        hero(statues, "figures/hero.png")
        where(statues, "figures/where_objects_go.png")
        movie(statues, "figures/loop.gif")
        lines = scene_table("figures/scenes.png")
        json.dump(lines, open("results/SCENES.json", "w"), indent=1)
        print(json.dumps(lines))
    if a.only in ("all", "bench") and os.path.exists("results/BSDS_TEST.json"):
        bench_plot("results/BSDS_TEST.json", "figures/bench.png")
