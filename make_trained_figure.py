"""figures/trained.png: equal-storage purification at depth 32 for three test faces."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eye_persistence as E
import posthoc_sigma as P
import trained_purifier as T
import trained_purifier_v2 as V

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def from_json(block):
    t = np.log(np.maximum(np.array(list(block["theta"].values())), 1e-30))
    return t, np.array(list(block["gate_u"].values()))


def main():
    r = json.loads(open("results/trained/trained_v2_results.json").read())
    tr_raw = E.load_split("training_with_face_landmarks.xml")
    te_raw = E.load_split("testing_with_face_landmarks.xml")
    allphi = np.concatenate([T.features(f["rgb"]) for f in tr_raw])
    m, s = allphi.mean(0), allphi.std(0) + 1e-9
    te = [T.Face(f, m, s) for f in te_raw]
    V.prepare(te)
    faces = sorted(te, key=lambda f: -f.f["box_px"])[:3]
    cols = [("original", None), ("fixed own EQ, σ 8", (T.BASE_T, T.BASE_U)),
            ("one knob: σ 2 (post-hoc)", P.params(2.0)),
            ("trained on mouth", from_json(r["training"]["mouth"])),
            ("trained on eyes*", from_json(r["training"]["eye"]))]
    fig, axes = plt.subplots(3, 5, figsize=(10, 6.6), facecolor=SURF)
    for i, fc in enumerate(faces):
        for j, (label, p) in enumerate(cols):
            ax = axes[i, j]
            if p is None:
                img = fc.f["rgb"]
            else:
                x, n, ok = V.run_bytes(fc, p[0], p[1], 32)
                img = x.reshape(T.N, T.N, 3)
            ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(label, fontsize=8.5, color=INK2)
    fig.text(0.02, 0.965, "Same storage budget (depth-32 bytes of the fixed purifier), three test faces",
             fontsize=12, color=INK, weight="bold")
    fig.text(0.02, 0.025, "* the eye-trained purifier never reached the byte budget on 20 of 25 test faces "
             "(the 512-step cap was exploited); it is not credited.", fontsize=8, color=INK2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.07, wspace=0.05, hspace=0.06)
    fig.savefig("figures/trained.png", dpi=120, facecolor=SURF)
    print("wrote figures/trained.png")


if __name__ == "__main__":
    main()
