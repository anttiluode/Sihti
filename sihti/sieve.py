"""The sieve (Finnish: sihti).

Run any purifier and keep what bleeds out between recorded depths:

    x_0 --> x_1 --> x_2 --> x_4 --> ... --> x_{2^J}
       R_0     R_1     R_2     ...                core

    R_j  = x_{d_j} - x_{d_{j+1}}           (what died between two depths)
    x_0  = R_0 + R_1 + ... + R_{J} + core   (telescoping -- nothing is lost)

The identity holds for ANY sequence of states: linear or nonlinear purifier,
changing purifiers, even an approximate or noisy one.  An imperfect purifier
changes what the channels *mean*; it can never make the sum lossy.  So the
inverse at the soma is plain addition, and it does not need to know which
operator produced the residues.

Depths are recorded in octaves (0, 1, 2, 4, 8, ...) because a single mode with
per-pass gain g spreads thinly over ~1/(1-g) consecutive single-pass residues;
octave spacing gives roughly constant resolution in lifetime.
"""
import numpy as np


def octave_depths(J):
    """[0, 1, 2, 4, ..., 2^J]"""
    return [0] + [2 ** j for j in range(int(J) + 1)]


def log_depths(dmax, count=48):
    """Roughly log-spaced integer depths in [0, dmax], always including 0 and dmax."""
    if dmax <= 0:
        return [0]
    d = np.unique(np.round(np.geomspace(1, dmax, count)).astype(int))
    return [0] + [int(v) for v in d]


class SieveResult:
    def __init__(self, depths, residues, core, frames):
        self.depths = list(depths)          # recorded octave depths
        self.residues = residues            # list of (H, W, C), len = len(depths) - 1
        self.core = core                    # state at the deepest depth
        self.frames = frames                # {depth: state} for scrubbing the loop

    @property
    def channels(self):
        return len(self.residues)

    def reconstruct(self, gains=None, core_gain=1.0):
        """Persistence EQ:  sum_j g_j R_j + g_core * core.  All ones -> the input."""
        if gains is None:
            gains = np.ones(len(self.residues))
        acc = core_gain * self.core.copy()
        for g, R in zip(reversed(list(gains)), reversed(self.residues)):
            acc = acc + g * R
        return acc

    def frame(self, depth):
        """Nearest recorded state at or below `depth` (for the loop movie)."""
        keys = sorted(self.frames)
        k = max([d for d in keys if d <= depth] or [keys[0]])
        return self.frames[k], k


def sift(x0, step, depths, frame_depths=None):
    """Run `step` from x0, record states at `depths` (must start at 0), keep residues."""
    depths = sorted(set(int(d) for d in depths))
    if depths[0] != 0:
        raise ValueError("depths must start at 0")
    frame_depths = sorted(set(int(d) for d in (frame_depths or [])) | set(depths))
    x = np.array(x0, dtype=np.float64, copy=True)
    frames = {0: x.copy()}
    cur = 0
    for d in frame_depths[1:]:
        for _ in range(d - cur):
            x = step(x)
        cur = d
        frames[d] = x.copy()
    states = [frames[d] for d in depths]
    residues = [states[i] - states[i + 1] for i in range(len(states) - 1)]
    return SieveResult(depths, residues, states[-1], frames)


def within_fraction(R, labels):
    """How much of a residue's energy varies *inside* the segments of `labels`.

    0 = the residue is flat on every segment (it moves whole regions),
    1 = the segments explain none of it.  Scale-free, so a small residue is not
    rewarded for being small.
    """
    C = R.shape[-1] if R.ndim == 3 else 1
    X = R.reshape(-1, C)
    lab = np.asarray(labels).ravel()
    _, lab = np.unique(lab, return_inverse=True)
    total = ((X - X.mean(axis=0)) ** 2).sum()
    if total <= 0:
        return 0.0
    counts = np.bincount(lab).astype(np.float64)
    between = 0.0
    for c in range(C):
        s = np.bincount(lab, weights=X[:, c])
        between += (s * s / np.maximum(counts, 1)).sum() - X[:, c].sum() ** 2 / X.shape[0]
    return float(max(0.0, total - between) / total)
