"""Small synthetic scenes with known ownership maps.

twins    two identical red disks: colour alone cannot tell them apart, cohesion can
shade    one object lit from the side: colour clustering splits it along the ramp
overlap  three occluding shapes: ownership, not sums
stripes  a striped disk: colour affinity cuts along every stripe -- a designed FAILURE
leaves   random occluding disks under a lighting gradient
"""
import numpy as np


def _grid(H, W):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    return yy, xx


def _disk(yy, xx, cy, cx, r):
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _finish(img, gt, noise, seed):
    rng = np.random.default_rng(seed)
    img = np.clip(img + noise * rng.standard_normal(img.shape), 0.0, 1.0)
    return img, gt.astype(np.int64)


def twins(H=72, W=96, noise=0.02, seed=0):
    yy, xx = _grid(H, W)
    img = np.empty((H, W, 3))
    img[:] = (0.35, 0.50, 0.35)
    gt = np.zeros((H, W), int)
    for lab, cx in ((1, 0.29 * W), (2, 0.71 * W)):
        m = _disk(yy, xx, H / 2, cx, 0.19 * H)
        img[m] = (0.80, 0.20, 0.20)
        gt[m] = lab
    return _finish(img, gt, noise, seed)


def shade(H=72, W=96, noise=0.02, seed=1):
    yy, xx = _grid(H, W)
    img = np.empty((H, W, 3))
    img[:] = (0.50, 0.35, 0.25)       # a brown that matches the object's mid-tone
    m = ((np.abs(yy - H / 2) < 0.30 * H) & (np.abs(xx - W / 2) < 0.38 * W))
    t = np.clip((xx - 0.12 * W) / (0.76 * W), 0, 1)            # light from the right
    ramp = (0.20 + 0.80 * t)[..., None] * np.array([1.0, 0.55, 0.20])
    img[m] = ramp[m]
    gt = m.astype(int)
    return _finish(img, gt, noise, seed)


def overlap(H=72, W=96, noise=0.02, seed=2):
    yy, xx = _grid(H, W)
    img = np.empty((H, W, 3))
    img[:] = (0.85, 0.85, 0.80)
    gt = np.zeros((H, W), int)
    rect = (np.abs(yy - 0.45 * H) < 0.25 * H) & (np.abs(xx - 0.35 * W) < 0.22 * W)
    img[rect], gt[rect] = (0.10, 0.55, 0.55), 1
    disk = _disk(yy, xx, 0.55 * H, 0.55 * W, 0.27 * H)
    img[disk], gt[disk] = (0.95, 0.80, 0.15), 2
    ell = ((yy - 0.40 * H) / (0.18 * H)) ** 2 + ((xx - 0.75 * W) / (0.20 * W)) ** 2 <= 1
    img[ell], gt[ell] = (0.75, 0.15, 0.55), 3
    return _finish(img, gt, noise, seed)


def stripes(H=72, W=96, noise=0.02, seed=3):
    yy, xx = _grid(H, W)
    img = np.empty((H, W, 3))
    img[:] = (0.5, 0.5, 0.5)
    m = _disk(yy, xx, H / 2, W / 2, 0.36 * H)
    band = ((xx // 3) % 2 == 0)
    img[m & band] = (0.05, 0.05, 0.05)
    img[m & ~band] = (0.95, 0.95, 0.95)
    return _finish(img, m.astype(int), noise, seed)


def leaves(H=72, W=96, count=9, noise=0.02, seed=4):
    rng = np.random.default_rng(seed)
    yy, xx = _grid(H, W)
    img = np.empty((H, W, 3))
    img[:] = rng.uniform(0.2, 0.8, 3)
    gt = np.zeros((H, W), int)
    for lab in range(1, count + 1):
        m = _disk(yy, xx, rng.uniform(0, H), rng.uniform(0, W), rng.uniform(0.10, 0.28) * H)
        img[m], gt[m] = rng.uniform(0.05, 0.95, 3), lab
    light = (0.75 + 0.5 * xx / W)[..., None]
    img = np.clip(img * light, 0, 1)
    _, gt = np.unique(gt, return_inverse=True)
    return _finish(img, gt.reshape(H, W), noise, seed)


SCENES = {"twins": twins, "shade": shade, "overlap": overlap, "stripes": stripes, "leaves": leaves}
