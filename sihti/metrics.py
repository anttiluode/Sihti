"""Segmentation metrics (numpy only).

covering(seg, gt): the BSDS-style covering of a ground truth by a segmentation,
    (1/N) * sum over gt regions R of |R| * max over seg regions S of IoU(R, S).
ari(a, b): adjusted Rand index.
vi(a, b): variation of information, natural log (lower is better).

With several human ground truths per image we report the mean over them.
"""
import numpy as np


def _relabel(x):
    _, inv = np.unique(np.asarray(x).ravel(), return_inverse=True)
    return inv


def contingency(a, b):
    a, b = _relabel(a), _relabel(b)
    na, nb = a.max() + 1, b.max() + 1
    return np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(np.float64)


def covering(seg, gt):
    C = contingency(gt, seg)            # rows: gt regions, cols: seg regions
    rg, rs = C.sum(1), C.sum(0)
    iou = C / (rg[:, None] + rs[None, :] - C)
    return float((rg * iou.max(1)).sum() / C.sum())


def ari(a, b):
    C = contingency(a, b)
    n = C.sum()
    comb = lambda x: x * (x - 1) / 2.0
    sum_ij = comb(C).sum()
    sa, sb = comb(C.sum(1)).sum(), comb(C.sum(0)).sum()
    expected = sa * sb / comb(n)
    maxi = 0.5 * (sa + sb)
    if maxi == expected:
        return 1.0
    return float((sum_ij - expected) / (maxi - expected))


def vi(a, b):
    C = contingency(a, b)
    n = C.sum()
    P = C / n
    pa, pb = P.sum(1), P.sum(0)
    nz = P > 0
    Ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    Hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    I = (P[nz] * np.log(P[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    return float(Ha + Hb - 2.0 * I)


def score(seg, gts):
    """Mean covering / ARI / VI of one segmentation against a list of ground truths."""
    return {
        "covering": float(np.mean([covering(seg, g) for g in gts])),
        "ari": float(np.mean([ari(seg, g) for g in gts])),
        "vi": float(np.mean([vi(seg, g) for g in gts])),
        "segments": int(len(np.unique(seg))),
    }
