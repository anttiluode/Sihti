#!/usr/bin/env python3
"""Sihti AI-G0: manual GPU gate for SDXL-Turbo.

Question:
    Does the purified Sihti core let SDXL-Turbo use a cheaper refinement budget
    without losing the structure/detail that a more expensive refinement creates?

Controls:
    raw            original 256x256 draft
    half_residue   core + 0.5 * residue
    sihti_core     core only
    gaussian_match ordinary Gaussian blur matched to the same draft->core MSE

The same prompt, draft and refinement noise seed are used.  The script records:
    - wall-clock time
    - exact core+residue identity error
    - PSNR to a more-expensive raw-draft teacher
    - 64x64 layout PSNR to the teacher
    - edge-map correlation to the teacher
    - direct 512x512 SDXL-Turbo latency

A speed claim is earned only when a quality-preserving route is faster after
draft + sieve + refinement overhead is counted.

Run:
    pip install -r requirements-ai.txt
    python ai_gate0_sdxl_turbo.py --local-only

Heavier:
    python ai_gate0_sdxl_turbo.py --local-only --seeds 100 101 102
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

MODEL_ID = "stabilityai/sdxl-turbo"
PROMPTS = (
    "Oil painting of an ancient stone castle surrounded by stormy ocean waves",
    "A futuristic solar bicycle parked on a Finnish forest gravel road, morning mist",
    "A cyberpunk cat wearing reflective sunglasses in a neon alley, cinematic lighting",
)


class GPUSieve(torch.nn.Module):
    def __init__(self, steps=8, sigma=0.12):
        super().__init__()
        self.steps = int(steps)
        self.sigma = float(sigma)
        self.inv_2s2 = 1.0 / (2.0 * self.sigma ** 2)

    @torch.no_grad()
    def forward(self, img):
        curr = img
        for _ in range(self.steps):
            pc = F.pad(curr, (1, 1, 1, 1), mode="replicate")
            po = F.pad(img, (1, 1, 1, 1), mode="replicate")
            uo, do = po[:, :, :-2, 1:-1], po[:, :, 2:, 1:-1]
            lo, ro = po[:, :, 1:-1, :-2], po[:, :, 1:-1, 2:]
            wu = torch.exp(-torch.sum((img - uo) ** 2, 1, keepdim=True) * self.inv_2s2)
            wd = torch.exp(-torch.sum((img - do) ** 2, 1, keepdim=True) * self.inv_2s2)
            wl = torch.exp(-torch.sum((img - lo) ** 2, 1, keepdim=True) * self.inv_2s2)
            wr = torch.exp(-torch.sum((img - ro) ** 2, 1, keepdim=True) * self.inv_2s2)
            uv, dv = pc[:, :, :-2, 1:-1], pc[:, :, 2:, 1:-1]
            lv, rv = pc[:, :, 1:-1, :-2], pc[:, :, 1:-1, 2:]
            wsum = 1.0 + 0.25 * (wu + wd + wl + wr)
            curr = (img + 0.25 * (wu * uv + wd * dv + wl * lv + wr * rv)) / wsum
        return curr, img - curr


def sync():
    torch.cuda.synchronize()


def timed(fn):
    sync()
    t0 = time.perf_counter()
    value = fn()
    sync()
    return value, time.perf_counter() - t0


def gen(seed):
    return torch.Generator(device="cuda").manual_seed(int(seed))


def to_tensor(image):
    x = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to("cuda")


def to_pil(t):
    x = torch.clamp(t[0].permute(1, 2, 0), 0, 1).mul(255).byte().cpu().numpy()
    return Image.fromarray(x, mode="RGB")


def gaussian_blur(img, sigma):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    c = img.shape[1]
    kh = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    out = F.conv2d(F.pad(img, (radius, radius, 0, 0), mode="reflect"), kh, groups=c)
    return F.conv2d(F.pad(out, (0, 0, radius, radius), mode="reflect"), kv, groups=c)


def matched_gaussian(draft, core):
    target = float(torch.mean((draft - core) ** 2).item())
    best = None
    for sigma in (0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        blurred = gaussian_blur(draft, sigma)
        mse = float(torch.mean((draft - blurred) ** 2).item())
        item = (abs(mse - target), float(sigma), blurred)
        if best is None or item[0] < best[0]:
            best = item
    return best[2], best[1]


def arr(image):
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def psnr(a, b):
    mse = float(np.mean((arr(a) - arr(b)) ** 2))
    return 120.0 if mse <= 1e-14 else float(10.0 * np.log10(1.0 / mse))


def layout_psnr(a, b):
    aa = a.resize((64, 64), Image.Resampling.BILINEAR)
    bb = b.resize((64, 64), Image.Resampling.BILINEAR)
    return psnr(aa, bb)


def edges(image):
    x = arr(image)
    gray = 0.299*x[:, :, 0] + 0.587*x[:, :, 1] + 0.114*x[:, :, 2]
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    return np.sqrt(gx*gx + gy*gy).ravel()


def edge_corr(a, b):
    x, y = edges(a), edges(b)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float(np.allclose(x, y))
    return float(np.corrcoef(x, y)[0, 1])


def sheet(items, path):
    thumb, label_h = 256, 28
    cols = min(4, len(items))
    rows = math.ceil(len(items) / cols)
    canvas = Image.new("RGB", (cols*thumb, rows*(thumb+label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate(items):
        r, c = divmod(i, cols)
        x0, y0 = c*thumb, r*(thumb+label_h)
        canvas.paste(image.resize((thumb, thumb), Image.Resampling.LANCZOS), (x0, y0))
        draw.text((x0+4, y0+thumb+5), label, fill="black")
    canvas.save(path)


def refine(pipe, prompt, image, steps, strength, seed):
    return timed(lambda: pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=int(steps),
        strength=float(strength),
        guidance_scale=0.0,
        generator=gen(seed),
    ).images[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[100])
    ap.add_argument("--teacher-steps", type=int, default=4)
    ap.add_argument("--cheap-steps", type=int, default=2)
    ap.add_argument("--direct-steps", type=int, default=2)
    ap.add_argument("--strength", type=float, default=0.60)
    ap.add_argument("--sieve-steps", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--skip-direct", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("results/ai_gate0"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("AI-G0 requires CUDA.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = tuple(args.prompt) if args.prompt else PROMPTS

    print("GPU:", torch.cuda.get_device_name(0))
    pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=args.local_only,
    ).to("cuda")
    pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i).to("cuda")
    pipe_t2i.set_progress_bar_config(disable=True)
    pipe_i2i.set_progress_bar_config(disable=True)

    sieve = GPUSieve(args.sieve_steps, args.sigma).to("cuda")

    print("Warmup...")
    _ = pipe_t2i(
        prompt="warmup", num_inference_steps=1, guidance_scale=0.0,
        width=256, height=256, generator=gen(1)
    ).images[0]
    sync()

    cases = []

    for pidx, prompt in enumerate(prompts):
        for seed in args.seeds:
            stem = f"p{pidx+1:02d}_s{seed}"
            print("\n", stem, prompt)

            draft, draft_s = timed(lambda: pipe_t2i(
                prompt=prompt, num_inference_steps=1, guidance_scale=0.0,
                width=256, height=256, generator=gen(seed)
            ).images[0])
            draft.save(args.output_dir / f"{stem}_draft.png")

            draft_t = to_tensor(draft)
            (core_t, residue_t), sieve_s = timed(lambda: sieve(draft_t))
            identity_error = float((core_t + residue_t - draft_t).abs().max().item())
            if identity_error > 2e-6:
                raise RuntimeError(f"core+residue identity failed: {identity_error}")

            gaussian_t, gaussian_sigma = matched_gaussian(draft_t, core_t)
            conditions = {
                "raw": draft_t,
                "half_residue": torch.clamp(core_t + 0.5*residue_t, 0, 1),
                "sihti_core": core_t,
                "gaussian_match": gaussian_t,
            }

            direct = None
            direct_s = None
            if not args.skip_direct:
                direct, direct_s = timed(lambda: pipe_t2i(
                    prompt=prompt, num_inference_steps=args.direct_steps,
                    guidance_scale=0.0, width=512, height=512,
                    generator=gen(seed)
                ).images[0])
                direct.save(args.output_dir / f"{stem}_direct512.png")

            raw_512 = to_pil(F.interpolate(
                draft_t, size=(512, 512), mode="bilinear", align_corners=False
            ))
            teacher, teacher_s = refine(
                pipe_i2i, prompt, raw_512, args.teacher_steps,
                args.strength, seed + 100000
            )
            teacher.save(args.output_dir / f"{stem}_teacher.png")

            methods = []
            items = [
                ("draft", draft),
                ("core", to_pil(core_t)),
                ("residue x2+0.5", to_pil(torch.clamp(residue_t*2.0+0.5, 0, 1))),
                (f"teacher/{args.teacher_steps}", teacher),
            ]
            if direct is not None:
                items.append((f"direct/{args.direct_steps}", direct))

            for name, cond_t in conditions.items():
                cond_512 = to_pil(F.interpolate(
                    cond_t, size=(512, 512), mode="bilinear", align_corners=False
                ))
                out, refine_s = refine(
                    pipe_i2i, prompt, cond_512, args.cheap_steps,
                    args.strength, seed + 100000
                )
                path = args.output_dir / f"{stem}_{name}.png"
                out.save(path)

                prep = 0.0 if name == "raw" else sieve_s
                receipt = {
                    "method": name,
                    "refine_seconds": float(refine_s),
                    "estimated_end_to_end_seconds": float(draft_s + prep + refine_s),
                    "psnr_to_teacher_db": psnr(out, teacher),
                    "layout_psnr_to_teacher_db": layout_psnr(out, teacher),
                    "edge_correlation_to_teacher": edge_corr(out, teacher),
                    "path": str(path),
                }
                methods.append(receipt)
                items.append((name, out))

            contact = args.output_dir / f"{stem}_contact.png"
            sheet(items, contact)

            case = {
                "prompt": prompt,
                "seed": seed,
                "draft_seconds": float(draft_s),
                "sieve_seconds": float(sieve_s),
                "identity_error": identity_error,
                "gaussian_sigma": gaussian_sigma,
                "direct512_seconds": None if direct_s is None else float(direct_s),
                "teacher_seconds": float(teacher_s),
                "methods": methods,
                "contact_sheet": str(contact),
            }
            cases.append(case)

            print(f"draft {draft_s:.3f}s  sieve {sieve_s:.4f}s  identity {identity_error:.2e}")
            if direct_s is not None:
                print(f"direct512/{args.direct_steps}: {direct_s:.3f}s")
            print(f"teacher/{args.teacher_steps}: {teacher_s:.3f}s")
            for row in methods:
                print(
                    f"{row['method']:14s} refine {row['refine_seconds']:.3f}s  "
                    f"layout {row['layout_psnr_to_teacher_db']:.2f}dB  "
                    f"edge {row['edge_correlation_to_teacher']:.3f}"
                )

    names = sorted({m["method"] for c in cases for m in c["methods"]})
    aggregate = {}
    for name in names:
        rows = [m for c in cases for m in c["methods"] if m["method"] == name]
        aggregate[name] = {
            "cases": len(rows),
            "mean_refine_seconds": float(np.mean([r["refine_seconds"] for r in rows])),
            "mean_end_to_end_seconds": float(np.mean([r["estimated_end_to_end_seconds"] for r in rows])),
            "mean_psnr_to_teacher_db": float(np.mean([r["psnr_to_teacher_db"] for r in rows])),
            "mean_layout_psnr_to_teacher_db": float(np.mean([r["layout_psnr_to_teacher_db"] for r in rows])),
            "mean_edge_correlation_to_teacher": float(np.mean([r["edge_correlation_to_teacher"] for r in rows])),
        }

    directs = [c["direct512_seconds"] for c in cases if c["direct512_seconds"] is not None]
    summary = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "teacher_steps": args.teacher_steps,
        "cheap_steps": args.cheap_steps,
        "strength": args.strength,
        "identity_max_error": float(max(c["identity_error"] for c in cases)),
        "mean_direct512_seconds": None if not directs else float(np.mean(directs)),
        "aggregate": aggregate,
        "cases": cases,
        "claim_rules": {
            "representation": (
                "Sihti earns a representation claim only if cheap sihti_core "
                "beats cheap raw and gaussian_match on layout/edge metrics across cases."
            ),
            "speed": (
                "Sihti earns a speed claim only if the quality-preserving route "
                "is faster after draft+sieve+refine overhead is counted."
            ),
        },
    }
    out = args.output_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nWrote", out)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
