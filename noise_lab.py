#!/usr/bin/env python3
"""Interactive null-input laboratory for Sihti.

This is an exploratory visualizer, not a scientific gate. It asks what each
purifier makes persistent when the input contains no intended objects.

Run:
    python noise_lab.py
    python noise_lab.py photo.jpg
    python noise_lab.py --seed 7 --sigma 8 --J 8
    python noise_lab.py photo.jpg --save figures/noise_lab.png --no-show

Processed panels use auto-contrast where needed. The energy panel is therefore
the important absolute-amplitude check: a visually crisp late pattern may have
almost no remaining energy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

from sihti import analyse, resize_to, SIGH_PRESETS
from sihti import scenes


SOURCE_NAMES = ("IID", "Blurred", "1/f power", "Phase scramble")
PURIFIER_NAMES = ("Own EQ", "Lattice", "Sigh high-pass")


def _unit_range(x: np.ndarray) -> np.ndarray:
    """Scale each channel independently to [0, 1] for a valid image."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[..., None]
    y = np.empty_like(x)
    for c in range(x.shape[-1]):
        a = float(np.min(x[..., c]))
        b = float(np.max(x[..., c]))
        if b - a < 1e-12:
            y[..., c] = 0.5
        else:
            y[..., c] = (x[..., c] - a) / (b - a)
    return y


def iid_noise(h: int, w: int, seed: int) -> np.ndarray:
    """Independent uniform RGB noise."""
    return np.random.default_rng(seed).random((h, w, 3))


def blurred_noise(h: int, w: int, seed: int, blur_sigma: float = 2.0) -> np.ndarray:
    """IID noise with short-range spatial correlation."""
    x = iid_noise(h, w, seed)
    x = ndimage.gaussian_filter(x, sigma=(blur_sigma, blur_sigma, 0), mode="reflect")
    return _unit_range(x)


def one_over_f_power_noise(h: int, w: int, seed: int) -> np.ndarray:
    """2-D RGB noise with approximately 1/f radial power spectral density."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    shape = np.zeros_like(f)
    nz = f > 0
    shape[nz] = 1.0 / np.sqrt(f[nz])

    chans = []
    for _ in range(3):
        z = rng.standard_normal((h, w))
        Z = np.fft.fft2(z)
        y = np.fft.ifft2(Z * shape).real
        chans.append(y)
    return _unit_range(np.stack(chans, axis=-1))


def phase_scramble(rgb: np.ndarray, seed: int) -> np.ndarray:
    """Destroy phase while keeping each channel's non-DC spectrum shape.

    The random unit-phase field comes from the FFT of a real noise image, so it
    has the Hermitian symmetry needed for a real inverse FFT. The final
    per-channel affine rescale changes absolute gain/DC but preserves relative
    non-DC Fourier-magnitude shape.
    """
    x = np.asarray(rgb, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty_like(x)
    for c in range(x.shape[-1]):
        X = np.fft.fft2(x[..., c])
        n = rng.standard_normal(x.shape[:2])
        N = np.fft.fft2(n)
        unit = N / np.maximum(np.abs(N), 1e-12)
        y = np.fft.ifft2(np.abs(X) * unit).real
        out[..., c] = y
    return _unit_range(out)


def load_reference(path: str | None, size: int) -> tuple[np.ndarray, str]:
    """Load a user image, or use Sihti's built-in shade scene."""
    if path:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
        return resize_to(rgb, size), Path(path).name
    rgb = scenes.shade()[0]
    return resize_to(rgb, size), "built-in shade scene"


def make_source(name: str, h: int, w: int, seed: int, reference: np.ndarray) -> np.ndarray:
    if name == "IID":
        return iid_noise(h, w, seed)
    if name == "Blurred":
        return blurred_noise(h, w, seed)
    if name == "1/f power":
        return one_over_f_power_noise(h, w, seed)
    if name == "Phase scramble":
        return phase_scramble(resize_to(reference, max(h, w)), seed)[:h, :w, :]
    raise ValueError(f"unknown source {name!r}")


def centered_rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean((x - x.mean(axis=(0, 1), keepdims=True)) ** 2)))


def signed_view(x: np.ndarray) -> np.ndarray:
    """Robust signed auto-contrast, zero mapped to mid-grey."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x, axis=(0, 1), keepdims=True)
    q = float(np.percentile(np.abs(x), 99.0))
    if q < 1e-12:
        return np.full_like(x, 0.5)
    return np.clip(0.5 + 0.5 * x / q, 0.0, 1.0)


def ordinary_view(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


class NoiseLab:
    def __init__(self, reference, reference_name, size=64, seed=0, sigma=5.0, J=8):
        self.reference = reference
        self.reference_name = reference_name
        self.size = int(size)
        self.seed = int(seed)
        self.sigma = float(sigma)
        self.J = int(J)
        self.source_name = "IID"
        self.purifier_name = "Own EQ"
        self.out = None

        self.fig = plt.figure(figsize=(13.5, 8.4))
        gs = self.fig.add_gridspec(2, 3)
        self.ax_input = self.fig.add_subplot(gs[0, 0])
        self.ax_state = self.fig.add_subplot(gs[0, 1])
        self.ax_residue = self.fig.add_subplot(gs[0, 2])
        self.ax_core = self.fig.add_subplot(gs[1, 0])
        self.ax_modes = self.fig.add_subplot(gs[1, 1])
        self.ax_energy = self.fig.add_subplot(gs[1, 2])
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.27, wspace=0.18, hspace=0.28)

        self.source_ax = self.fig.add_axes([0.06, 0.035, 0.13, 0.16])
        self.purifier_ax = self.fig.add_axes([0.21, 0.035, 0.15, 0.16])
        self.depth_ax = self.fig.add_axes([0.41, 0.145, 0.48, 0.035])
        self.sigma_ax = self.fig.add_axes([0.41, 0.085, 0.48, 0.035])
        self.prev_ax = self.fig.add_axes([0.41, 0.025, 0.08, 0.04])
        self.next_ax = self.fig.add_axes([0.50, 0.025, 0.08, 0.04])
        self.reset_ax = self.fig.add_axes([0.60, 0.025, 0.09, 0.04])

        self.source_radio = RadioButtons(self.source_ax, SOURCE_NAMES, active=0)
        self.purifier_radio = RadioButtons(self.purifier_ax, PURIFIER_NAMES, active=0)
        self.depth_slider = Slider(self.depth_ax, "depth index", 0, self.J + 1, valinit=0, valstep=1)
        self.sigma_slider = Slider(self.sigma_ax, "self-EQ sigma", 2.0, 20.0, valinit=self.sigma)
        self.prev_button = Button(self.prev_ax, "seed -")
        self.next_button = Button(self.next_ax, "seed +")
        self.reset_button = Button(self.reset_ax, "reset")

        self.source_radio.on_clicked(self._source_changed)
        self.purifier_radio.on_clicked(self._purifier_changed)
        self.depth_slider.on_changed(lambda _v: self.draw())
        self.sigma_slider.on_changed(self._sigma_changed)
        self.prev_button.on_clicked(lambda _e: self._change_seed(-1))
        self.next_button.on_clicked(lambda _e: self._change_seed(+1))
        self.reset_button.on_clicked(self._reset)
        self.recompute()

    def _source_changed(self, label):
        self.source_name = str(label)
        self.recompute()

    def _purifier_changed(self, label):
        self.purifier_name = str(label)
        self.recompute()

    def _sigma_changed(self, value):
        self.sigma = float(value)
        self.recompute()

    def _change_seed(self, delta):
        self.seed = max(0, self.seed + int(delta))
        self.recompute()

    def _reset(self, _event):
        self.seed = 0
        self.source_name = "IID"
        self.purifier_name = "Own EQ"
        self.sigma = 5.0
        self.source_radio.set_active(0)
        self.purifier_radio.set_active(0)
        self.sigma_slider.set_val(5.0)
        self.depth_slider.set_val(0)
        self.recompute()

    def recompute(self):
        source = make_source(self.source_name, self.size, self.size, self.seed, self.reference)
        purifier = {"Own EQ": "self", "Lattice": "lattice", "Sigh high-pass": "sigh"}[self.purifier_name]
        gains = SIGH_PRESETS["high pass"] if purifier == "sigh" else None
        self.out = analyse(
            source, radius=1.5, sigma=self.sigma, J=self.J, k="auto",
            n_modes=10, purifier=purifier, gains=gains, frames=32, seed=0,
        )
        self.depth_slider.valmax = len(self.out["sieve"].depths) - 1
        self.depth_slider.ax.set_xlim(self.depth_slider.valmin, self.depth_slider.valmax)
        if self.depth_slider.val > self.depth_slider.valmax:
            self.depth_slider.set_val(self.depth_slider.valmax)
        self.draw()

    def draw(self):
        if self.out is None:
            return
        res = self.out["sieve"]
        depths = res.depths
        idx = int(np.clip(round(self.depth_slider.val), 0, len(depths) - 1))
        depth = depths[idx]
        state = res.frames[depth]
        ridx = min(idx, len(res.residues) - 1)
        residue = res.residues[ridx]
        rlo, rhi = depths[ridx], depths[ridx + 1]

        for ax in (self.ax_input, self.ax_state, self.ax_residue, self.ax_core, self.ax_modes):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        self.ax_input.imshow(ordinary_view(self.out["rgb"]))
        self.ax_input.set_title(f"{self.source_name} input")

        if self.purifier_name == "Sigh high-pass":
            self.ax_state.imshow(signed_view(state))
            self.ax_core.imshow(signed_view(res.core))
        else:
            self.ax_state.imshow(ordinary_view(state))
            self.ax_core.imshow(ordinary_view(res.core))
        self.ax_state.set_title(f"state at depth {depth} | RMS {centered_rms(state):.4g}")

        self.ax_residue.imshow(signed_view(residue))
        self.ax_residue.set_title(f"residue {rlo}->{rhi} | RMS {centered_rms(residue):.4g}")

        self.ax_core.set_title(f"deep core @ {depths[-1]} | RMS {centered_rms(res.core):.4g}")

        self.ax_modes.imshow(self.out["painted"])
        scene_count = max(0, len(self.out.get("scene_modes", [])) - 1)
        self.ax_modes.set_title(f"input-graph slow modes | non-lattice candidates {scene_count}")

        self.ax_energy.clear()
        state_energy = [centered_rms(res.frames[d]) for d in depths]
        residue_energy = [centered_rms(r) for r in res.residues]
        x_state = np.arange(len(depths))
        x_res = np.arange(len(residue_energy)) + 0.5
        self.ax_energy.plot(x_state, state_energy, marker="o", label="state RMS")
        self.ax_energy.plot(x_res, residue_energy, marker=".", label="residue RMS")
        self.ax_energy.axvline(idx, linewidth=1, alpha=0.45)
        self.ax_energy.set_xticks(x_state)
        self.ax_energy.set_xticklabels([str(d) for d in depths], rotation=45, ha="right")
        self.ax_energy.set_xlabel("depth")
        self.ax_energy.set_ylabel("absolute centered RMS")
        self.ax_energy.set_yscale("log")
        self.ax_energy.grid(alpha=0.2)
        self.ax_energy.legend(loc="best", fontsize=8)
        self.ax_energy.set_title("Don't trust auto-contrast: watch absolute energy")

        title = f"Sihti noise null lab - {self.purifier_name} - seed {self.seed} - sigma={self.sigma:.2f}"
        if self.source_name == "Phase scramble":
            title += f" - reference: {self.reference_name}"
        self.fig.suptitle(title, fontsize=13)
        self.fig.canvas.draw_idle()

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("photo", nargs="?", help="optional photo used by the phase-scrambled null")
    p.add_argument("--size", type=int, default=64, help="square working size (default: 64)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=5.0, help="self-EQ colour scale")
    p.add_argument("--J", type=int, default=8, help="deepest octave exponent; 8 -> depth 256")
    p.add_argument("--save", help="save the current view to this path")
    p.add_argument("--no-show", action="store_true", help="do not open the interactive window")
    return p.parse_args()


def main():
    a = parse_args()
    size = max(24, min(int(a.size), 160))
    J = max(4, min(int(a.J), 10))
    ref, ref_name = load_reference(a.photo, size)
    lab = NoiseLab(ref, ref_name, size=size, seed=max(0, a.seed), sigma=float(a.sigma), J=J)
    if a.save:
        lab.save(a.save)
        print(f"saved {a.save}")
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
