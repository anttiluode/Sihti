"""POST-HOC diagnostic (not pre-registered): does tuning ONE knob of the fixed own EQ
(the Lab range weight, i.e. sigma) under the v2 equal-bytes budget explain the v2 gain?
Chosen on train faces only; scored once on test.  python posthoc_sigma.py"""
import json
from pathlib import Path

import numpy as np

import eye_persistence as E
import trained_purifier as T
import trained_purifier_v2 as V

GRID_SIGMA = (8.0, 4.0, 2.0, 1.4, 1.0, 0.7, 0.5)


def params(sigma):
    t = T.BASE_T.copy()
    t[:3] = np.log(1 / (2 * sigma ** 2))
    return t, T.BASE_U


def main():
    tr_raw = E.load_split("training_with_face_landmarks.xml")
    te_raw = E.load_split("testing_with_face_landmarks.xml")
    allphi = np.concatenate([T.features(f["rgb"]) for f in tr_raw])
    m, s = allphi.mean(0), allphi.std(0) + 1e-9
    tr = [T.Face(f, m, s) for f in tr_raw]
    te = [T.Face(f, m, s) for f in te_raw]
    V.prepare(tr)
    V.prepare(te)
    train_rows = {}
    for sg in GRID_SIGMA:
        t, u = params(sg)
        ss = [V.run_bytes(fc, t, u, 32) for fc in tr]
        c = T.score(tr, [x[0] for x in ss], "eye")[1].mean()
        train_rows[sg] = {"train_eye_c_own": float(c),
                          "unreached": int(sum(not x[2] for x in ss))}
        print("sigma", sg, train_rows[sg], flush=True)
    ok = {k: v for k, v in train_rows.items() if v["unreached"] == 0}
    best = max(ok, key=lambda k: ok[k]["train_eye_c_own"])
    t, u = params(best)
    test = {}
    for d in T.DEPTHS:
        ss = [V.run_bytes(fc, t, u, d) for fc in te]
        xs = [x[0] for x in ss]
        row = {"unreached": int(sum(not x[2] for x in ss)),
               "steps_mean": float(np.mean([x[1] for x in ss]))}
        for which in ("eye", "mouth"):
            acc, c = T.score(te, xs, which)
            row[which] = {"correct": int(round(acc * len(te))), "mean_c_own": float(c.mean())}
        row["raw_eye_rmse_mean"] = float(np.mean([
            np.sqrt(np.mean((V.raw_eye_patch(x.reshape(T.N, T.N, 3), fc.f) - fc.raw_eye) ** 2))
            for fc, x in zip(te, xs)]))
        test[str(d)] = row
        print("test d", d, row, flush=True)
    out = {"posthoc": True, "chosen_sigma_on_train": best, "train_grid": train_rows,
           "test": test}
    Path("results/trained/posthoc_sigma.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
