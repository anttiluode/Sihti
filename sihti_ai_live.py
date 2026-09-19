#!/usr/bin/env python3
"""Sihti AI Live -- causal SDXL-Turbo sieve instrument.

The important control is Residue Back alpha:

    alpha = 0.0   -> refine the purified Sihti core only
    alpha = 0.5   -> restore half of the removed detail
    alpha = 1.0   -> exact identity: core + residue == draft

The old prototype always used alpha=1, so changing the purifier mostly changed
the diagnostic panels while the diffusion model still received the original
draft. This version makes the decomposition causally affect refinement.

Run:
    python sihti_ai_live.py
"""
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

MODEL_ID = "stabilityai/sdxl-turbo"
PANEL_SIZE = 256
COL = {
    "bg": "#2B2D30",
    "panel": "#1E1F22",
    "ink": "#DFE1E5",
    "dim": "#8C9099",
    "accent": "#5E81AC",
}


class GPUSieve(torch.nn.Module):
    def __init__(self, steps=6, sigma=0.12):
        super().__init__()
        self.steps = int(steps)
        self.sigma = float(sigma)
        self._refresh()

    def _refresh(self):
        self.inv_2s2 = 1.0 / (2.0 * self.sigma ** 2)

    def update_params(self, steps, sigma):
        self.steps = int(steps)
        self.sigma = float(sigma)
        self._refresh()

    @torch.no_grad()
    def forward(self, img):
        curr = img
        for _ in range(self.steps):
            pc = F.pad(curr, (1, 1, 1, 1), mode="replicate")
            po = F.pad(img, (1, 1, 1, 1), mode="replicate")

            uo = po[:, :, :-2, 1:-1]
            do = po[:, :, 2:, 1:-1]
            lo = po[:, :, 1:-1, :-2]
            ro = po[:, :, 1:-1, 2:]

            wu = torch.exp(-torch.sum((img - uo) ** 2, dim=1, keepdim=True) * self.inv_2s2)
            wd = torch.exp(-torch.sum((img - do) ** 2, dim=1, keepdim=True) * self.inv_2s2)
            wl = torch.exp(-torch.sum((img - lo) ** 2, dim=1, keepdim=True) * self.inv_2s2)
            wr = torch.exp(-torch.sum((img - ro) ** 2, dim=1, keepdim=True) * self.inv_2s2)

            uv = pc[:, :, :-2, 1:-1]
            dv = pc[:, :, 2:, 1:-1]
            lv = pc[:, :, 1:-1, :-2]
            rv = pc[:, :, 1:-1, 2:]

            wsum = 1.0 + 0.25 * (wu + wd + wl + wr)
            curr = (
                img
                + 0.25 * (wu * uv + wd * dv + wl * lv + wr * rv)
            ) / wsum

        return curr, img - curr


def pil_to_tensor(image, device="cuda"):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_pil(tensor):
    arr = (
        torch.clamp(tensor[0].permute(1, 2, 0), 0, 1)
        .mul(255).byte().cpu().numpy()
    )
    return Image.fromarray(arr, mode="RGB")


class AIEngine(threading.Thread):
    def __init__(self, post):
        super().__init__(daemon=True)
        self.post = post
        self.cv = threading.Condition()
        self.job = None
        self.alive = True
        self.pipe_t2i = None
        self.pipe_i2i = None
        self.sieve = GPUSieve().to("cuda")

    def load_model(self):
        self.post(("status", "Loading SDXL-Turbo into CUDA VRAM..."))
        self.pipe_t2i = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        self.pipe_i2i = AutoPipelineForImage2Image.from_pipe(self.pipe_t2i).to("cuda")
        self.pipe_t2i.set_progress_bar_config(disable=True)
        self.pipe_i2i.set_progress_bar_config(disable=True)
        self.post(("status", "Ready."))

    def submit(self, prompt, params, seed):
        with self.cv:
            self.job = (prompt, dict(params), int(seed))
            self.cv.notify()

    def run(self):
        try:
            self.load_model()
        except Exception as exc:
            self.post(("error", f"Model load failed: {exc}"))
            return

        while self.alive:
            with self.cv:
                while self.alive and self.job is None:
                    self.cv.wait(0.25)
                if not self.alive:
                    return
                job, self.job = self.job, None

            prompt, params, seed = job
            try:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                gen = torch.Generator(device="cuda").manual_seed(seed)

                self.post(("status", "Generating 256x256 draft..."))
                draft = self.pipe_t2i(
                    prompt=prompt,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    width=256,
                    height=256,
                    generator=gen,
                ).images[0]

                self.sieve.update_params(params["steps"], params["sigma"])
                draft_t = pil_to_tensor(draft)
                core_t, residue_t = self.sieve(draft_t)
                identity_error = float((core_t + residue_t - draft_t).abs().max().item())

                alpha = float(params["alpha"])
                condition_t = torch.clamp(core_t + alpha * residue_t, 0, 1)

                self.post(("status", f"Refining with residue alpha={alpha:.2f}..."))
                condition_512 = F.interpolate(
                    condition_t, size=(512, 512), mode="bilinear", align_corners=False
                )
                refine_input = tensor_to_pil(condition_512)

                gen = torch.Generator(device="cuda").manual_seed(seed + 100000)
                refined = self.pipe_i2i(
                    prompt=prompt,
                    image=refine_input,
                    num_inference_steps=int(params["refine_steps"]),
                    strength=float(params["strength"]),
                    guidance_scale=0.0,
                    generator=gen,
                ).images[0]

                torch.cuda.synchronize()
                dt = time.perf_counter() - t0

                self.post(("result", {
                    "draft": draft,
                    "core": tensor_to_pil(core_t),
                    "residue": tensor_to_pil(torch.clamp(residue_t * 2.0 + 0.5, 0, 1)),
                    "refined": refined,
                    "time": dt,
                    "alpha": alpha,
                    "identity_error": identity_error,
                }))
            except Exception as exc:
                self.post(("error", str(exc)))


class App:
    def __init__(self, root):
        self.root = root
        root.title("Sihti AI Live — causal SDXL-Turbo sieve")
        root.configure(bg=COL["bg"])

        self.q = queue.Queue()
        self.engine = AIEngine(self.q.put)
        self.engine.start()
        self.photos = {}
        self._build()
        root.after(40, self._poll)

    def _build(self):
        bar = ttk.Frame(self.root, padding=(12, 10))
        bar.pack(fill=tk.X)
        self.prompt = tk.StringVar(
            value="Oil painting of an ancient stone castle surrounded by stormy ocean waves"
        )
        e = ttk.Entry(bar, textvariable=self.prompt)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        e.bind("<Return>", lambda _e: self.generate())
        ttk.Button(bar, text="Generate", command=self.generate).pack(side=tk.RIGHT)

        ctrl = ttk.Frame(self.root, padding=(12, 2))
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Purifier steps").pack(side=tk.LEFT)
        self.steps = tk.IntVar(value=8)
        ttk.Combobox(
            ctrl, textvariable=self.steps, values=[1, 2, 3, 5, 8, 12],
            width=3, state="readonly"
        ).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(ctrl, text="Edge sigma").pack(side=tk.LEFT)
        self.sigma = tk.DoubleVar(value=0.12)
        ttk.Scale(ctrl, from_=0.02, to=0.40, variable=self.sigma, length=100).pack(
            side=tk.LEFT, padx=(4, 12)
        )

        ttk.Label(ctrl, text="Residue back α").pack(side=tk.LEFT)
        self.alpha = tk.DoubleVar(value=0.0)
        ttk.Scale(ctrl, from_=0.0, to=1.0, variable=self.alpha, length=110).pack(
            side=tk.LEFT, padx=(4, 5)
        )
        ttk.Button(ctrl, text="Core", command=lambda: self.alpha.set(0.0)).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Half", command=lambda: self.alpha.set(0.5)).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Identity", command=lambda: self.alpha.set(1.0)).pack(
            side=tk.LEFT, padx=(0, 12)
        )

        ttk.Label(ctrl, text="Refine strength").pack(side=tk.LEFT)
        self.strength = tk.DoubleVar(value=0.60)
        ttk.Scale(ctrl, from_=0.40, to=0.90, variable=self.strength, length=90).pack(
            side=tk.LEFT, padx=(4, 10)
        )

        ttk.Label(ctrl, text="Steps").pack(side=tk.LEFT)
        self.refine_steps = tk.IntVar(value=2)
        ttk.Combobox(
            ctrl, textvariable=self.refine_steps, values=[2, 3, 4],
            width=3, state="readonly"
        ).pack(side=tk.LEFT, padx=(4, 10))

        ttk.Label(ctrl, text="Seed").pack(side=tk.LEFT)
        self.seed = tk.IntVar(value=100)
        ttk.Entry(ctrl, textvariable=self.seed, width=6).pack(side=tk.LEFT, padx=(4, 0))

        panels = ttk.Frame(self.root, padding=(12, 8))
        panels.pack(fill=tk.X)
        self.labels = {}
        titles = (
            ("draft", "1. Draft"),
            ("core", "2. Sieve Core"),
            ("residue", "3. Residue"),
            ("refined", "4. Refined Output"),
        )
        for col, (key, title) in enumerate(titles):
            f = ttk.Frame(panels)
            f.grid(row=0, column=col, padx=6)
            ttk.Label(f, text=title).pack(anchor="w")
            blank = ImageTk.PhotoImage(Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), COL["panel"]))
            self.photos[key] = blank
            lab = tk.Label(f, image=blank, bg=COL["panel"], bd=0)
            lab.pack(pady=(4, 0))
            self.labels[key] = lab

        self.status = ttk.Label(self.root, text="Initializing...", padding=(12, 6))
        self.status.pack(fill=tk.X)

    def generate(self):
        self.engine.submit(
            self.prompt.get(),
            {
                "steps": self.steps.get(),
                "sigma": self.sigma.get(),
                "alpha": self.alpha.get(),
                "strength": self.strength.get(),
                "refine_steps": self.refine_steps.get(),
            },
            self.seed.get(),
        )

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "status":
                    self.status.config(text=msg[1])
                elif msg[0] == "error":
                    self.status.config(text=f"Error: {msg[1]}")
                elif msg[0] == "result":
                    data = msg[1]
                    for key in ("draft", "core", "residue", "refined"):
                        ph = ImageTk.PhotoImage(
                            data[key].resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
                        )
                        self.photos[key] = ph
                        self.labels[key].config(image=ph)
                    self.status.config(
                        text=(
                            f"Done in {data['time']:.2f}s · residue α={data['alpha']:.2f} · "
                            f"core+residue identity error {data['identity_error']:.2e}"
                        )
                    )
        except queue.Empty:
            pass
        self.root.after(40, self._poll)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Sihti AI Live requires CUDA.")
    root = tk.Tk()
    app = App(root)
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: (setattr(app.engine, "alive", False), root.destroy()),
    )
    root.mainloop()


if __name__ == "__main__":
    main()
