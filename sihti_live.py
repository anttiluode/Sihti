#!/usr/bin/env python3
"""Sihti live -- a sieve for images.

    python sihti_live.py                 # starts on a built-in scene
    python sihti_live.py photo.jpg       # starts on your image

Panels
    Input     the working image (small on purpose: this is an instrument, not a filter)
    Loop      the image purified by the chosen EQ, at the depth on the slider
              (space plays the movie -- Sigh's loop, but on the image's own graph)
    Objects   the slowest modes of the image's own EQ painted as colour, or the
              segments they give.  Modes a blank frame also has are skipped.
    Sum       every residue times its slider, plus the core.  All sliders at 1
              gives back the input exactly -- the error is printed.

Residue strip: what bled out between depths 0-1, 1-2, 2-4, ... and the core.
With the image's own EQ the core is the object layout and the residues are
what is inside the objects (fine texture first, broad shading last).

Keys: space play/stop, o open, w webcam, b blank frame, s save snapshot,
1/2/3 purifier (own EQ / lattice / Sigh EQ).
"""
import json
import os
import queue
import sys
import threading
import time
import traceback

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

from sihti import analyse, resize_to, SIGH_PRESETS
from sihti import scenes as SC
from sihti import render as R
from sihti import modes as M

COL = {"bg": "#3A3C3F", "panel": "#323437", "ink": "#E6E4DF", "dim": "#A3A5A8",
       "accent": "#D9C9A0", "line": "#4B4E52", "trough": "#2A2C2F"}
PANEL = 288          # target panel width in screen pixels
DEFAULTS = {"sigma": 5.0, "radius": 1.5, "J": 9, "k": "auto", "long_side": 96,
            "purifier": "self", "sigh_preset": "low pass"}
LIVE_LONG_SIDE = 64


# ---------------------------------------------------------------------------
# background worker
# ---------------------------------------------------------------------------
class Engine(threading.Thread):
    """Runs the analysis off the UI thread.  Latest request wins."""

    def __init__(self, post):
        super().__init__(daemon=True)
        self.post = post
        self.cv = threading.Condition()
        self.job = None
        self.cam = None
        self.cam_params = None
        self.alive = True
        self.prev = None           # previous frame's painted-mode block, for stable colours

    def submit(self, image, params):
        with self.cv:
            self.job = (image, dict(params))
            self.cv.notify()

    def live(self, cam, params):
        with self.cv:
            self.cam, self.cam_params, self.prev = cam, dict(params), None
            self.cv.notify()

    def update_live_params(self, params):
        with self.cv:
            self.cam_params = dict(params)

    def stop_live(self):
        with self.cv:
            cam, self.cam = self.cam, None
        if cam is not None:
            try:
                cam.release()
            except Exception:
                pass

    def run(self):
        while self.alive:
            with self.cv:
                while self.alive and self.job is None and self.cam is None:
                    self.cv.wait(0.25)
                job, self.job = self.job, None
                cam, params = self.cam, self.cam_params
            try:
                if job is not None:
                    image, params = job
                    live = False
                elif cam is not None:
                    ok, frame = cam.read()
                    if not ok:
                        self.post(("error", "The webcam returned no frame."))
                        time.sleep(0.5)
                        continue
                    rgb = frame[:, ::-1, ::-1].astype(np.float64) / 255.0   # BGR->RGB, mirrored
                    image = resize_to(rgb, params.get("live_long_side", LIVE_LONG_SIDE))
                    live = True
                else:
                    continue
                out = compute(image, params)
                if live:
                    self._stabilise(out)
                self.post(("result", out, params, live))
            except Exception:
                self.post(("error", traceback.format_exc(limit=2)))
                time.sleep(0.2)

    def _stabilise(self, out):
        """Rotate the painted modes to match the previous frame (colours stop flickering)."""
        md = out.get("modes")
        if md is None:
            return
        use = out.get("scene_modes", [0, 1, 2, 3])[1:5]
        if len(use) < 3:
            use = [1, 2, 3]
        block = md.V[:, use]
        block = block / (np.linalg.norm(block, axis=0, keepdims=True) + 1e-12)
        if self.prev is not None and self.prev.shape == block.shape:
            block = M.align(block, self.prev)
        self.prev = block
        out["painted"] = paint_block(block, md.H, md.W)


def paint_block(block, H, W):
    out = np.zeros((block.shape[0], 3))
    for c in range(min(3, block.shape[1])):
        v = block[:, c]
        a, b = np.percentile(v, [1, 99])
        out[:, c] = np.clip((v - a) / (b - a + 1e-12), 0, 1)
    return out.reshape(H, W, 3)


def compute(image, p):
    k = p["k"]
    k = "auto" if k == "auto" else int(k)
    gains = SIGH_PRESETS.get(p.get("sigh_preset", "low pass"))
    J = int(p["J"])
    return analyse(image, radius=float(p["radius"]), sigma=float(p["sigma"]), J=J, k=k,
                   n_modes=14, purifier=p["purifier"], gains=gains, frames=40)


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------
class App:
    def __init__(self, root, first_image=None):
        self.root = root
        self.q = queue.Queue()
        self.engine = Engine(self.q.put)
        self.engine.start()
        self.source = None
        self.source_name = ""
        self.out = None
        self.params_used = None
        self.photos = {}
        self.gain_vars = []
        self.playing = False
        self.after_recompute = None
        self.cam_on = False
        self.frame_depths = [0]
        self._style()
        self._build()
        if first_image:
            self.open_path(first_image)
        else:
            self.load_scene("shade")
        root.after(30, self._poll)

    # ------------------------------------------------------------------ look
    def _style(self):
        r = self.root
        r.title("Sihti \u2014 a sieve for images")
        r.configure(bg=COL["bg"])
        st = ttk.Style(r)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=COL["bg"], foreground=COL["ink"], fieldbackground=COL["panel"],
                     bordercolor=COL["line"], lightcolor=COL["bg"], darkcolor=COL["bg"],
                     troughcolor=COL["trough"], focuscolor=COL["accent"])
        st.configure("TButton", background=COL["panel"], padding=(10, 4))
        st.map("TButton", background=[("active", COL["line"])])
        st.configure("Accent.TButton", background=COL["accent"], foreground="#23221F")
        st.map("Accent.TButton", background=[("active", "#E8DBB8")])
        st.configure("TRadiobutton", background=COL["bg"], foreground=COL["ink"])
        st.map("TRadiobutton", background=[("active", COL["bg"])],
               indicatorcolor=[("selected", COL["accent"]), ("!selected", COL["panel"])])
        st.configure("TLabel", background=COL["bg"], foreground=COL["ink"])
        st.configure("Dim.TLabel", foreground=COL["dim"])
        st.configure("Title.TLabel", foreground=COL["ink"], font=("TkDefaultFont", 11, "bold"))
        st.configure("TCombobox", fieldbackground=COL["panel"], background=COL["panel"],
                     foreground=COL["ink"], arrowcolor=COL["ink"])
        st.map("TCombobox", fieldbackground=[("readonly", COL["panel"])],
               foreground=[("readonly", COL["ink"])], background=[("readonly", COL["panel"])],
               selectbackground=[("readonly", COL["panel"])], selectforeground=[("readonly", COL["ink"])])
        r.option_add("*TCombobox*Listbox.background", COL["panel"])
        r.option_add("*TCombobox*Listbox.foreground", COL["ink"])
        r.option_add("*TCombobox*Listbox.selectBackground", COL["accent"])
        r.option_add("*TCombobox*Listbox.selectForeground", "#23221F")
        st.configure("Horizontal.TScale", background=COL["bg"])

    def _build(self):
        r = self.root
        top = ttk.Frame(r, padding=(10, 8, 10, 2))
        top.pack(fill=tk.X)
        ttk.Button(top, text="Open image", command=self.open_dialog).pack(side=tk.LEFT)
        self.cam_btn = ttk.Button(top, text="Use webcam", command=self.toggle_webcam)
        self.cam_btn.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Blank frame", command=self.load_blank).pack(side=tk.LEFT, padx=(6, 0))
        self.scene_var = tk.StringVar(value="shade")
        cb = ttk.Combobox(top, textvariable=self.scene_var, values=list(SC.SCENES), width=9,
                          state="readonly")
        cb.pack(side=tk.LEFT, padx=(6, 0))
        cb.bind("<<ComboboxSelected>>", lambda e: self.load_scene(self.scene_var.get()))
        ttk.Button(top, text="Save snapshot", command=self.save_snapshot).pack(side=tk.RIGHT)

        opts = ttk.Frame(r, padding=(10, 2, 10, 6))
        opts.pack(fill=tk.X)
        ttk.Label(opts, text="Purify with").pack(side=tk.LEFT)
        self.purifier = tk.StringVar(value=DEFAULTS["purifier"])
        for txt, val in (("the image's own EQ", "self"), ("the blank lattice", "lattice"),
                         ("Sigh's EQ", "sigh")):
            ttk.Radiobutton(opts, text=txt, value=val, variable=self.purifier,
                            command=self.params_changed).pack(side=tk.LEFT, padx=(8, 0))
        self.sigh_preset = tk.StringVar(value=DEFAULTS["sigh_preset"])
        self.sigh_box = ttk.Combobox(opts, textvariable=self.sigh_preset, width=9, state="readonly",
                                     values=[p for p in SIGH_PRESETS if p != "all pass"])
        self.sigh_box.pack(side=tk.LEFT, padx=(4, 0))
        self.sigh_box.bind("<<ComboboxSelected>>", lambda e: self.params_changed())

        knobs = ttk.Frame(r, padding=(10, 0, 10, 6))
        knobs.pack(fill=tk.X)
        self.sigma = tk.DoubleVar(value=DEFAULTS["sigma"])
        self.sigma_lbl = ttk.Label(knobs, text="", width=12)
        ttk.Label(knobs, text="Edge sharpness").pack(side=tk.LEFT)
        ttk.Scale(knobs, from_=2.0, to=20.0, variable=self.sigma, length=160,
                  command=lambda v: self._knob_moved()).pack(side=tk.LEFT, padx=(6, 0))
        self.sigma_lbl.pack(side=tk.LEFT, padx=(6, 12))
        self.radius = tk.StringVar(value=str(DEFAULTS["radius"]))
        self.J = tk.StringVar(value=str(DEFAULTS["J"]))
        self.k = tk.StringVar(value=DEFAULTS["k"])
        self.long_side = tk.StringVar(value=str(DEFAULTS["long_side"]))
        for label, var, vals, w in (("Reach", self.radius, ["1.5", "2.0", "3.0"], 4),
                                    ("Depth 2^", self.J, [str(v) for v in range(5, 12)], 3),
                                    ("Objects", self.k, ["auto"] + [str(v) for v in range(2, 13)], 5),
                                    ("Working size", self.long_side, ["64", "96", "128", "160"], 4)):
            ttk.Label(knobs, text=label).pack(side=tk.LEFT, padx=(8, 0))
            c = ttk.Combobox(knobs, textvariable=var, values=vals, width=w, state="readonly")
            c.pack(side=tk.LEFT, padx=(4, 0))
            c.bind("<<ComboboxSelected>>", lambda e: self.params_changed())
        self._update_sigma_label()

        # four panels
        mid = ttk.Frame(r, padding=(10, 4))
        mid.pack(fill=tk.X)
        self.img_labels, self.captions = {}, {}
        for col, (key, title) in enumerate((("input", "Input"), ("loop", "Loop"),
                                            ("objects", "Objects"), ("sum", "Sum"))):
            f = ttk.Frame(mid)
            f.grid(row=0, column=col, padx=6, sticky="n")
            ttk.Label(f, text=title, style="Title.TLabel").pack(anchor="w")
            blank = ImageTk.PhotoImage(Image.new("RGB", (PANEL, int(PANEL * 0.75)), COL["panel"]))
            self.photos[f"blank_{key}"] = blank
            lab = tk.Label(f, bg=COL["panel"], image=blank, bd=0)
            lab.pack()
            self.img_labels[key] = lab
            cap = ttk.Label(f, text="", style="Dim.TLabel", wraplength=PANEL, justify=tk.LEFT)
            cap.pack(anchor="w", pady=(2, 0))
            self.captions[key] = cap
            if key == "loop":
                row = ttk.Frame(f)
                row.pack(fill=tk.X)
                self.depth_idx = tk.DoubleVar(value=0)
                self.depth_scale = ttk.Scale(row, from_=0, to=1, variable=self.depth_idx,
                                             length=PANEL - 70, command=lambda v: self.show_loop())
                self.depth_scale.pack(side=tk.LEFT)
                self.play_btn = ttk.Button(row, text="Play", width=6, style="Accent.TButton",
                                           command=self.toggle_play)
                self.play_btn.pack(side=tk.LEFT, padx=(6, 0))
            if key == "objects":
                row = ttk.Frame(f)
                row.pack(fill=tk.X)
                self.view = tk.StringVar(value="modes")
                for txt, val in (("Modes", "modes"), ("Segments", "segments")):
                    ttk.Radiobutton(row, text=txt, value=val, variable=self.view,
                                    command=self.show_objects).pack(side=tk.LEFT, padx=(0, 8))

        # residue strip
        self.strip_title = ttk.Label(r, text="What bled out between depths (grey = nothing)",
                                     style="Title.TLabel", padding=(16, 8, 10, 0))
        self.strip_title.pack(anchor="w")
        self.strip = ttk.Frame(r, padding=(10, 2))
        self.strip.pack(fill=tk.X)
        pres = ttk.Frame(r, padding=(10, 4))
        pres.pack(fill=tk.X)
        for txt, fn in (("Everything (lossless)", self.preset_all), ("Objects only", self.preset_core),
                        ("Insides only", self.preset_insides), ("Smooth the insides", self.preset_smooth),
                        ("Boost the insides", self.preset_boost)):
            ttk.Button(pres, text=txt, command=fn).pack(side=tk.LEFT, padx=(0, 6))

        self.status = ttk.Label(r, text="", style="Dim.TLabel", padding=(12, 6))
        self.status.pack(fill=tk.X)

        r.bind("<space>", lambda e: self.toggle_play())
        r.bind("o", lambda e: self.open_dialog())
        r.bind("w", lambda e: self.toggle_webcam())
        r.bind("b", lambda e: self.load_blank())
        r.bind("s", lambda e: self.save_snapshot())
        for key, val in (("1", "self"), ("2", "lattice"), ("3", "sigh")):
            r.bind(key, lambda e, v=val: (self.purifier.set(v), self.params_changed()))
        r.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------ inputs
    def params(self):
        return {"sigma": round(float(self.sigma.get()), 2), "radius": float(self.radius.get()),
                "J": int(self.J.get()), "k": self.k.get(), "purifier": self.purifier.get(),
                "sigh_preset": self.sigh_preset.get(), "live_long_side": LIVE_LONG_SIDE}

    def set_source(self, rgb, name):
        self.source = resize_to(rgb, int(self.long_side.get()))
        self.source_name = name
        self.recompute(now=True)

    def open_dialog(self):
        path = filedialog.askopenfilename(title="Open an image", filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")])
        if path:
            self.open_path(path)

    def open_path(self, path):
        try:
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
        except Exception as exc:
            self.status.config(text=f"Could not open {os.path.basename(path)}: {exc}")
            return
        self.stop_webcam()
        self._raw = rgb
        self.set_source(rgb, os.path.basename(path))

    def load_scene(self, name):
        self.stop_webcam()
        rgb, _ = SC.SCENES[name]()
        self._raw = rgb
        self.set_source(rgb, f"scene: {name}")

    def load_blank(self):
        self.stop_webcam()
        rgb = np.full((72, 96, 3), 0.55)
        self._raw = rgb
        self.set_source(rgb, "blank frame (only the substrate is left)")

    def toggle_webcam(self):
        if self.cam_on:
            self.stop_webcam()
            return
        try:
            import cv2
        except ImportError:
            self.status.config(text="The webcam needs OpenCV: pip install opencv-python")
            return
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            self.status.config(text="No webcam found at index 0.")
            return
        self.cam_on = True
        self.cam_btn.config(text="Stop webcam")
        self.source_name = "webcam"
        self.engine.live(cam, self.params())

    def stop_webcam(self):
        if self.cam_on:
            self.engine.stop_live()
            self.cam_on = False
            self.cam_btn.config(text="Use webcam")

    # ------------------------------------------------------------ compute
    def _knob_moved(self):
        self._update_sigma_label()
        self.params_changed()

    def _update_sigma_label(self):
        s = float(self.sigma.get())
        self.sigma_lbl.config(text=f"\u03c3 = {s:4.1f} Lab")

    def params_changed(self):
        if self.cam_on:
            self.engine.update_live_params(self.params())
            return
        if self.after_recompute is not None:
            self.root.after_cancel(self.after_recompute)
        self.after_recompute = self.root.after(180, self.recompute)

    def recompute(self, now=False):
        self.after_recompute = None
        if getattr(self, "_raw", None) is not None and not self.cam_on:
            want = int(self.long_side.get())
            if self.source is None or max(self.source.shape[:2]) != want:
                self.source = resize_to(self._raw, want)
        if self.source is None:
            return
        self.status.config(text="Sifting\u2026")
        self.engine.submit(self.source, self.params())

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "error":
                    self.status.config(text="Error: " + msg[1].strip().splitlines()[-1])
                else:
                    _, out, params, live = msg
                    self.show(out, params, live)
        except queue.Empty:
            pass
        self.root.after(30, self._poll)

    # ------------------------------------------------------------ display
    def _photo(self, key, img, stretch=False):
        img = np.asarray(img, dtype=np.float64)
        if stretch:
            lo, hi = float(img.min()), float(img.max())
            img = (img - lo) / (hi - lo + 1e-12)
        H, W = img.shape[:2]
        f = max(1, PANEL // W)
        ph = ImageTk.PhotoImage(R.upscale(R.to_u8(img), f))
        self.photos[key] = ph
        return ph

    def show(self, out, params, live):
        first = self.out is None or self.out["sieve"].channels != out["sieve"].channels
        self.out, self.params_used = out, params
        res = out["sieve"]
        self.frame_depths = sorted(res.frames)
        self.depth_scale.config(to=max(1, len(self.frame_depths) - 1))
        if first or not live:
            self.depth_idx.set(len(self.frame_depths) - 1 if not self.playing else 0)
        self.img_labels["input"].config(image=self._photo("input", out["rgb"]))
        H, W = out["rgb"].shape[:2]
        self.captions["input"].config(text=f"{self.source_name}  \u00b7  {W}\u00d7{H} px")
        self.show_loop()
        self.show_objects()
        if first or len(self.gain_vars) != res.channels + 1:
            self._build_strip()
        else:
            self._fill_strip()
        self.show_sum()
        t = out["timing"]
        purifier = {"self": "the image's own EQ", "lattice": "the blank lattice",
                    "sigh": f"Sigh's EQ ({params.get('sigh_preset')})"}[params["purifier"]]
        self.status.config(text=(
            f"Purified with {purifier}  \u00b7  graph {1000 * t['graph']:.0f} ms  \u00b7  "
            f"sieve {1000 * t['sieve']:.0f} ms  \u00b7  modes {1000 * t['modes']:.0f} ms"
            + ("  \u00b7  live" if live else "")))

    def show_loop(self):
        if self.out is None:
            return
        i = int(round(float(self.depth_idx.get())))
        i = max(0, min(i, len(self.frame_depths) - 1))
        d = self.frame_depths[i]
        x = self.out["sieve"].frames[d]
        sigh = (self.params_used or {}).get("purifier") == "sigh"
        # Sigh's EQ does not average: it removes DC and leaves tiny high-frequency fields,
        # so (like the original Sigh app) the display is normalised.
        stretch = bool(sigh or x.min() < -0.05 or x.max() > 1.05)
        self.img_labels["loop"].config(image=self._photo("loop", x, stretch))
        self.captions["loop"].config(text=f"depth {d} of {self.frame_depths[-1]} passes"
                                     + ("  (display stretched)" if stretch else ""))

    def show_objects(self):
        if self.out is None or "modes" not in self.out:
            return
        if self.view.get() == "modes":
            img = self.out["painted"]
            which = self.out.get("painted_which", [1, 2, 3])
            sh = self.out.get("shares", [])
            shares = ", ".join(f"{sh[m - 1]:.2f}" for m in which if 0 < m <= len(sh))
            skipped = [m for m in range(1, (which[-1] if which else 3) + 1) if m not in which]
            txt = (f"modes {', '.join(map(str, which))} as R, G, B  \u00b7  blank-frame share {shares}"
                   + (f"  \u00b7  skipped substrate modes {', '.join(map(str, skipped))}" if skipped else ""))
        else:
            seg = self.out["segments"]
            img = R.overlay_boundaries(R.label_colors(seg, self.out["rgb"]), seg, (0.93, 0.90, 0.80))
            txt = f"{self.out['k']} segments" + ("  (eigengap)" if self.params_used["k"] == "auto" else "")
        self.img_labels["objects"].config(image=self._photo("objects", img))
        self.captions["objects"].config(text=txt)

    def _build_strip(self):
        for w in self.strip.winfo_children():
            w.destroy()
        res = self.out["sieve"]
        self.gain_vars, self.thumb_labels = [], []
        names = [f"{a}\u2192{b}" for a, b in zip(res.depths[:-1], res.depths[1:])] + ["core"]
        for c, name in enumerate(names):
            f = ttk.Frame(self.strip)
            f.grid(row=0, column=c, padx=3, sticky="n")
            lab = tk.Label(f, bg=COL["panel"], bd=0)
            lab.pack()
            self.thumb_labels.append(lab)
            ttk.Label(f, text=name, style="Dim.TLabel").pack()
            var = tk.DoubleVar(value=1.0)
            s = tk.Scale(f, from_=2.0, to=-1.0, resolution=0.05, orient=tk.VERTICAL, length=86,
                         variable=var, showvalue=True, command=lambda v: self.show_sum(),
                         bg=COL["bg"], fg=COL["ink"], troughcolor=COL["trough"],
                         activebackground=COL["accent"], highlightthickness=0, bd=0,
                         font=("TkDefaultFont", 8), sliderlength=14, width=12)
            s.pack()
            self.gain_vars.append(var)
        self._fill_strip()

    def _fill_strip(self):
        res = self.out["sieve"]
        H, W = self.out["rgb"].shape[:2]
        tw = 104 if len(res.residues) <= 10 else 90
        f = max(1, tw // W)
        for c, R_ in enumerate(res.residues + [res.core]):
            img = R_ if c == len(res.residues) else R.signed(R_)[0]
            ph = ImageTk.PhotoImage(R.upscale(R.to_u8(np.clip(img, 0, 1)), f))
            self.photos[f"strip{c}"] = ph
            self.thumb_labels[c].config(image=ph)

    def show_sum(self):
        if self.out is None or not self.gain_vars:
            return
        res = self.out["sieve"]
        g = [v.get() for v in self.gain_vars[:-1]]
        gc = self.gain_vars[-1].get()
        y = res.reconstruct(g, gc) + (1.0 - gc) * res.core.mean(axis=(0, 1))
        self.img_labels["sum"].config(image=self._photo("sum", np.clip(y, 0, 1)))
        if all(abs(v - 1.0) < 1e-9 for v in g + [gc]):
            err = float(np.abs(res.reconstruct() - self.out["rgb"]).max())
            self.captions["sum"].config(text=f"all sliders at 1: lossless, max error {err:.1e}")
        else:
            self.captions["sum"].config(text="edited: residues times their sliders, plus the core")

    # ------------------------------------------------------------ presets
    def _set_gains(self, fn):
        n = len(self.gain_vars) - 1
        for j, v in enumerate(self.gain_vars[:-1]):
            v.set(fn(j, n))
        self.show_sum()

    def preset_all(self):
        self._set_gains(lambda j, n: 1.0)
        self.gain_vars[-1].set(1.0)
        self.show_sum()

    def preset_core(self):
        self._set_gains(lambda j, n: 0.0)
        self.gain_vars[-1].set(1.0)
        self.show_sum()

    def preset_insides(self):
        self._set_gains(lambda j, n: 1.0)
        self.gain_vars[-1].set(0.0)
        self.show_sum()

    def preset_smooth(self):
        self._set_gains(lambda j, n: 0.0 if j < n // 2 else 1.0)
        self.gain_vars[-1].set(1.0)
        self.show_sum()

    def preset_boost(self):
        self._set_gains(lambda j, n: 2.0 if j < (2 * n) // 3 else 1.0)
        self.gain_vars[-1].set(1.0)
        self.show_sum()

    # ------------------------------------------------------------ play / save / close
    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="Stop" if self.playing else "Play")
        if self.playing:
            if int(round(float(self.depth_idx.get()))) >= len(self.frame_depths) - 1:
                self.depth_idx.set(0)
            self._tick()

    def _tick(self):
        if not self.playing:
            return
        i = int(round(float(self.depth_idx.get()))) + 1
        if i >= len(self.frame_depths):
            self.playing = False
            self.play_btn.config(text="Play")
            return
        self.depth_idx.set(i)
        self.show_loop()
        self.root.after(90, self._tick)

    def save_snapshot(self):
        if self.out is None:
            return
        os.makedirs("snapshots", exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        res = self.out["sieve"]
        i = int(round(float(self.depth_idx.get())))
        d = self.frame_depths[max(0, min(i, len(self.frame_depths) - 1))]
        g = [v.get() for v in self.gain_vars[:-1]]
        gc = self.gain_vars[-1].get()
        y = np.clip(res.reconstruct(g, gc) + (1 - gc) * res.core.mean(axis=(0, 1)), 0, 1)
        seg = self.out["segments"]
        row1 = [self.out["rgb"], np.clip(res.frames[d], 0, 1), self.out["painted"],
                R.overlay_boundaries(R.label_colors(seg, self.out["rgb"]), seg), y]
        row2 = [R.signed(Rr)[0] for Rr in res.residues][-5:]
        R.grid([row1, row2], titles=["input", f"loop depth {d}", "modes", "segments", "sum"],
               scale=3).save(f"snapshots/sihti-{stamp}.png")
        meta = {"source": self.source_name, "params": self.params_used, "gains": g, "core_gain": gc,
                "depth_shown": d, "k": self.out["k"], "shares": self.out.get("shares", [])[:6],
                "lossless_error": self.out["lossless_error"]}
        with open(f"snapshots/sihti-{stamp}.json", "w") as f:
            json.dump(meta, f, indent=1)
        self.status.config(text=f"Saved snapshots/sihti-{stamp}.png and .json")

    def close(self):
        self.engine.alive = False
        self.stop_webcam()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root, sys.argv[1] if len(sys.argv) > 1 else None)
    root.mainloop()


if __name__ == "__main__":
    main()
