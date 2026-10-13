"""figures/eyes.png from results/eye/eye_results.json (+ one test face re-sieved)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eye_persistence as E

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
S = {"sihti_learned": "#2a78d6", "sihti_natural": "#eb6834", "jpeg": "#1baf7a",
     "laplacian_pyramid": "#eda100", "sihti_lattice": "#e87ba4"}
NAMES = {"sihti_learned": "Sihti, learned order", "sihti_natural": "Sihti, deep-first",
         "jpeg": "JPEG", "laplacian_pyramid": "Laplacian pyramid",
         "sihti_lattice": "Sihti, plain diffusion"}
SHOW = [0, 8, 16, 32, 64, 256]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    r = json.loads(Path("results/eye/eye_results.json").read_text())
    test = E.load_split("testing_with_face_landmarks.xml")
    face = max(test, key=lambda f: f["box_px"])
    sieves = {p: E.sieve(face["rgb"], p) for p in ("self", "lattice")}

    fig = plt.figure(figsize=(12, 7.6), facecolor=SURF)
    gs = fig.add_gridspec(3, len(SHOW), height_ratios=[1, 1, 2.3], hspace=0.35,
                          wspace=0.08, left=0.06, right=0.98, top=0.86, bottom=0.15)
    fig.text(0.06, 0.955, "When do the eyes stop being these eyes?", fontsize=14,
             color=INK, weight="bold")
    fig.text(0.06, 0.925, "Core only, no residues added back. Box = the eye patch the "
             "readers score.", fontsize=9, color=INK2)
    lm = face["landmarks"][36:48]
    for row, (pur, label) in enumerate((("self", "own EQ"), ("lattice", "plain diffusion"))):
        for c, d in enumerate(SHOW):
            ax = fig.add_subplot(gs[row, c])
            ax.imshow(np.clip(sieves[pur].frames[d], 0, 1), interpolation="nearest")
            x0, x1 = lm[:, 0].min(), lm[:, 0].max()
            w = (x1 - x0) * 1.4
            cx, cy = (x0 + x1) / 2, lm[:, 1].mean()
            ax.add_patch(plt.Rectangle((cx - w / 2, cy - w / 4), w, w / 2, fill=False,
                                       ec="#ffffff", lw=0.8))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if row == 0:
                ax.set_title(f"depth {d}", fontsize=8, color=INK2)
            if c == 0:
                ax.set_ylabel(label, fontsize=9, color=INK)

    bottom = gs[2, :].subgridspec(1, 3, wspace=0.32)
    ax = fig.add_subplot(bottom[0])
    style(ax)
    for pur, col, lab in (("self", S["sihti_learned"], "own EQ"),
                          ("lattice", S["sihti_lattice"], "plain diffusion")):
        cur = r["persistence"][pur]["curve"]
        x = [max(c["depth"], 0.5) for c in cur]
        y = [c["accuracy"] for c in cur]
        ax.plot(x, y, color=col, lw=2, marker="o", ms=4)
        ax.annotate(lab, (x[-1], y[-1]), xytext=(5, 2 if pur == "self" else -4),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.4, 2 ** 11.5)
    ax.set_title("Eye identity (25-way)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("purification depth (0 plotted at ½)", fontsize=8, color=INK2)
    ax.set_ylim(-0.03, 1.05)

    ax = fig.add_subplot(bottom[1])
    style(ax)
    for pur, col, lab in (("self", S["sihti_learned"], "own EQ"),
                          ("lattice", S["sihti_lattice"], "plain diffusion")):
        cur = r["persistence"][pur]["curve"]
        x = [max(c["depth"], 0.5) for c in cur]
        y = [c["mean_generic_rel"] for c in cur]
        ax.plot(x, y, color=col, lw=2, marker="o", ms=4)
        ax.annotate(lab, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set_xlim(0.4, 2 ** 11.5)
    ax.set_xscale("log", base=2)
    ax.set_title("Generic eye-likeness (1 = original)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("purification depth", fontsize=8, color=INK2)

    ax = fig.add_subplot(bottom[2])
    style(ax)
    for m in ("jpeg", "laplacian_pyramid", "sihti_learned", "sihti_natural",
              "sihti_lattice"):
        env = np.array(r["envelopes"][m])
        ax.step(env[:, 0], env[:, 1], where="post", color=S[m], lw=2)
        b = r["B92_mean_bytes_per_face"][m]
        ax.plot([b], [0.92], marker="o", ms=6, color=S[m], mec=SURF, mew=1.5,
                label=f"{NAMES[m]} · {b:,.0f} B", ls="none")
    ax.axhline(0.92, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(40, 6e4)
    ax.set_ylim(-0.03, 1.05)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2,
                    fontsize=7, frameon=False, title="first reaches 23/25 at",
                    title_fontsize=7, columnspacing=1.0, handletextpad=0.2)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title("Identity vs stored bytes per face", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("mean bytes per face (log)", fontsize=8,
                  color=INK2)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/eyes.png", dpi=130, facecolor=SURF)
    print("wrote figures/eyes.png")


if __name__ == "__main__":
    main()
