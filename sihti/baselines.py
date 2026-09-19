"""Attackers for the object claim.

kmeans_lab_xy   k-means on (L, a, b, w*x, w*y): colour clustering with a position
                pull.  If the self-EQ cannot beat this, it is colour clustering
                with extra steps.
felzenszwalb    the classic efficient graph segmentation (Felzenszwalb &
                Huttenlocher 2004), scikit-image's reference implementation.
                Optional dependency -- only the benchmark needs it.
"""
import numpy as np

from .cluster import kmeans


def kmeans_lab_xy(lab, k, pos_weight=20.0, seed=0, n_init=4):
    H, W = lab.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    s = float(max(H, W))
    F = np.concatenate([lab.reshape(-1, 3),
                        pos_weight * (xx.reshape(-1, 1) / s),
                        pos_weight * (yy.reshape(-1, 1) / s)], axis=1)
    lab_, _, _ = kmeans(F, k, n_init=n_init, seed=seed)
    return lab_.reshape(H, W)


def felzenszwalb(rgb, scale=1.0, sigma=0.5, min_size=20):
    from skimage.segmentation import felzenszwalb as _fh
    return _fh(np.clip(rgb, 0, 1), scale=scale, sigma=sigma, min_size=int(min_size),
               channel_axis=-1)
