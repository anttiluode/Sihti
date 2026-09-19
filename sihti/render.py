"""Rendering helpers (PIL + numpy only)."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (58, 60, 63)          # neutral photographic mid-grey: does not bias how images read
INK = (230, 228, 223)
DIM = (160, 162, 165)


def to_u8(x):
    return (np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def signed(R, scale=None):
    """Signed residue as an image: grey = nothing died here."""
    R = np.asarray(R, dtype=np.float64)
    if scale is None:
        m = np.percentile(np.abs(R), 99.5)
        scale = 0.45 / m if m > 1e-12 else 1.0
    return np.clip(0.5 + scale * R, 0.0, 1.0), scale


def label_colors(labels, rgb=None, seed=7):
    """Fill each segment with its mean colour (if rgb given) or a fixed palette."""
    lab = np.asarray(labels)
    _, inv = np.unique(lab.ravel(), return_inverse=True)
    K = inv.max() + 1
    if rgb is not None:
        X = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
        cnt = np.bincount(inv, minlength=K)[:, None]
        mean = np.stack([np.bincount(inv, weights=X[:, c], minlength=K) for c in range(3)], 1)
        cols = mean / np.maximum(cnt, 1)
    else:
        rng = np.random.default_rng(seed)
        cols = rng.uniform(0.15, 0.95, (K, 3))
    return cols[inv].reshape(lab.shape + (3,))


def boundaries(labels):
    lab = np.asarray(labels)
    b = np.zeros(lab.shape, bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    return b


def overlay_boundaries(rgb, labels, color=(1.0, 1.0, 1.0)):
    out = np.array(rgb, dtype=np.float64, copy=True)
    out[boundaries(labels)] = color
    return out


def upscale(img_u8, factor):
    im = Image.fromarray(img_u8)
    return im.resize((im.width * factor, im.height * factor), Image.Resampling.NEAREST)


def font(size=14):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def grid(rows, titles=None, row_labels=None, scale=3, pad=8, title_h=22, label_w=0,
         caption=None):
    """rows: list of lists of float images (H, W, 3) in [0, 1] (same H, W per row)."""
    f = font(14)
    tiles = [[upscale(to_u8(img), scale) for img in row] for row in rows]
    cols = max(len(r) for r in tiles)
    tw = max(t.width for r in tiles for t in r)
    th = [max(t.height for t in r) for r in tiles]
    lw = label_w if row_labels else 0
    top = title_h if titles else 0
    cap_h = 26 if caption else 0
    Wt = lw + pad + cols * (tw + pad)
    Ht = top + pad + sum(h + pad for h in th) + cap_h
    canvas = Image.new("RGB", (Wt, Ht), BG)
    d = ImageDraw.Draw(canvas)
    if titles:
        for c, t in enumerate(titles):
            d.text((lw + pad + c * (tw + pad), 4), t, fill=INK, font=f)
    y = top + pad
    for r, row in enumerate(tiles):
        if row_labels:
            d.text((6, y + th[r] // 2 - 8), row_labels[r], fill=INK, font=f)
        for c, t in enumerate(row):
            canvas.paste(t, (lw + pad + c * (tw + pad), y))
        y += th[r] + pad
    if caption:
        d.text((pad, Ht - cap_h + 4), caption, fill=DIM, font=font(13))
    return canvas
