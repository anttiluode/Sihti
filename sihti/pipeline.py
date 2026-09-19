"""analyse(): the whole v0 instrument in one call.

    image --Lab--> self-EQ (the image's own graph)
          --sieve--> octave residues + core      (lossless by addition)
          --modes--> slowest non-trivial modes  (objects, painted as RGB)
          --NJW----> segments
          --blank--> substrate share of the painted modes
"""
import time

import numpy as np
from PIL import Image

from .color import srgb_to_lab
from .graph import self_affinity, lattice_affinity
from .purifiers import GraphDiffusion, FourierEQ, SIGH_PRESETS, RebuiltSelfEQ
from .sieve import sift, octave_depths, log_depths
from . import modes as M

_LATTICE = {}


def resize_to(rgb, long_side):
    """Area-average resize so the longer side is `long_side` pixels."""
    H, W = rgb.shape[:2]
    s = float(long_side) / max(H, W)
    if abs(s - 1.0) < 1e-9:
        return np.asarray(rgb, dtype=np.float64)
    h, w = max(2, int(round(H * s))), max(2, int(round(W * s)))
    im = Image.fromarray(np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8))
    return np.asarray(im.resize((w, h), Image.Resampling.BOX), dtype=np.float64) / 255.0


def lattice_modes(H, W, radius, k=13):
    key = (H, W, float(radius), k)
    if key not in _LATTICE:
        if len(_LATTICE) > 6:
            _LATTICE.clear()
        _LATTICE[key] = M.spectral_modes(lattice_affinity(H, W, radius), H, W, k=k)
    return _LATTICE[key]


def purifier_for(name, rgb, Wm=None, radius=2.0, sigma=8.0, laziness=0.5, gains=None):
    H, W = rgb.shape[:2]
    if name == "self":
        return GraphDiffusion(Wm, H, W, laziness)
    if name == "lattice":
        return GraphDiffusion(lattice_affinity(H, W, radius), H, W, laziness)
    if name == "sigh":
        return FourierEQ(H, W, gains if gains is not None else SIGH_PRESETS["low pass"])
    if name == "rebuilt":
        return RebuiltSelfEQ(H, W, radius, sigma, laziness)
    raise ValueError(f"unknown purifier {name!r}")


def analyse(rgb, radius=2.0, sigma=8.0, J=9, k=6, n_modes=16, purifier="self",
            gains=None, laziness=0.5, frames=48, want_modes=True, want_substrate=True,
            seed=0):
    t0 = time.perf_counter()
    rgb = np.asarray(rgb, dtype=np.float64)
    H, W = rgb.shape[:2]
    lab = srgb_to_lab(rgb)
    Wm = self_affinity(lab, radius=radius, sigma=sigma)
    t_graph = time.perf_counter()

    step = purifier_for(purifier, rgb, Wm, radius, sigma, laziness, gains)
    depths = octave_depths(J)
    res = sift(rgb, step, depths, frame_depths=log_depths(depths[-1], frames))
    t_sieve = time.perf_counter()

    out = {"rgb": rgb, "lab": lab, "W": Wm, "sieve": res, "H": H, "W_px": W,
           "params": dict(radius=radius, sigma=sigma, J=J, k=k, purifier=purifier,
                          laziness=laziness)}
    recon = res.reconstruct()
    out["lossless_error"] = float(np.abs(recon - rgb).max())

    if want_modes:
        md = M.spectral_modes(Wm, H, W, k=n_modes, seed=seed)
        kk = M.eigengap_k(md.mu, 2, min(12, n_modes - 1)) if k in (None, "auto", 0) else int(k)
        out["modes"] = md
        out["k"] = kk
        out["segments"] = M.segments(md, kk, seed=seed)
        which = (1, 2, 3)
        if want_substrate:
            basis = M.lattice_basis(lattice_modes(H, W, radius), 12)
            shares = M.substrate_share(md, basis, which=range(1, md.U.shape[1]))
            out["shares"] = shares
            out["substrate"] = shares[:3]
            scene = M.scene_modes(shares, tau=0.9)
            out["scene_modes"] = scene
            if len(scene) >= 4:
                which = scene[1:4]
        out["painted_which"] = list(which)
        out["painted"] = M.paint(md, which)
    t_modes = time.perf_counter()
    out["timing"] = {"graph": t_graph - t0, "sieve": t_sieve - t_graph,
                     "modes": t_modes - t_sieve, "total": t_modes - t0}
    return out
