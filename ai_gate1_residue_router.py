#!/usr/bin/env python3
"""Sihti AI-G1: does residue localize where extra diffusion work is needed?

This gate is CPU-only and consumes the already-uploaded AI-G0 images.

Target:
    per-tile correction energy between the cheap RAW two-step refinement and
    the four-step teacher.

Predictors computed before the teacher is consulted:
    - sihti_residue: RMS(draft - Sihti core)
    - gaussian_residue: RMS(draft - matched Gaussian blur)
    - edge_energy: draft gradient energy
    - local_variance: draft RGB variance

The image is divided into an 8x8 grid. The draft/core predictors are measured
on matching 32x32 draft tiles; the correction target is measured on 64x64
teacher/output tiles.

Primary metrics:
    - within-image Spearman correlation to correction energy
    - fraction of total correction energy captured by the top 25% predicted
      tiles

The predeclared router claim in AI_GATE1.md requires Sihti residue to beat every
simple baseline by >=0.05 on both mean primary metrics and to win top-25%
capture in at least 6/9 cases.

This does NOT claim that tile-wise diffusion will reproduce the teacher. It only
asks whether residue is a useful compute-allocation signal worth taking to a
GPU routing gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

GRID = 8
TOP_FRACTION = 0.25
PREDICTORS = ("sihti_residue", "gaussian_residue", "edge_energy", "local_variance")


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def tile_reduce(field: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Mean a scalar HxW field into grid x grid spatial cells."""
    field = np.asarray(field, dtype=np.float64)
    h, w = field.shape
    if h % grid or w % grid:
        raise ValueError(f"shape {field.shape} is not divisible by grid={grid}")
    th, tw = h // grid, w // grid
    return field.reshape(grid, th, grid, tw).mean(axis=(1, 3)).ravel()


def correction_target(raw: np.ndarray, teacher: np.ndarray, grid: int = GRID) -> np.ndarray:
    err = np.mean((np.asarray(raw) - np.asarray(teacher)) ** 2, axis=2)
    return tile_reduce(err, grid)


def predictor_maps(
    draft: np.ndarray,
    core: np.ndarray,
    gaussian_sigma: float,
    grid: int = GRID,
) -> dict[str, np.ndarray]:
    draft = np.asarray(draft, dtype=np.float64)
    core = np.asarray(core, dtype=np.float64)

    sihti = np.mean((draft - core) ** 2, axis=2)

    smooth = gaussian_filter(
        draft,
        sigma=(float(gaussian_sigma), float(gaussian_sigma), 0.0),
        mode="reflect",
    )
    gauss = np.mean((draft - smooth) ** 2, axis=2)

    gray = 0.299 * draft[:, :, 0] + 0.587 * draft[:, :, 1] + 0.114 * draft[:, :, 2]
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    edge = gx * gx + gy * gy

    h, w = gray.shape
    if h % grid or w % grid:
        raise ValueError("draft shape must be divisible by grid")
    th, tw = h // grid, w // grid
    blocks = draft.reshape(grid, th, grid, tw, 3)
    # Spatial+channel variance per tile.
    variance = blocks.var(axis=(1, 3, 4)).ravel()

    return {
        "sihti_residue": tile_reduce(sihti, grid),
        "gaussian_residue": tile_reduce(gauss, grid),
        "edge_energy": tile_reduce(edge, grid),
        "local_variance": variance,
    }


def safe_spearman(score: np.ndarray, target: np.ndarray) -> float:
    if np.std(score) < 1e-15 or np.std(target) < 1e-15:
        return 0.0
    value = float(spearmanr(score, target).statistic)
    return 0.0 if not np.isfinite(value) else value


def top_capture(score: np.ndarray, target: np.ndarray, fraction: float = TOP_FRACTION) -> float:
    score = np.asarray(score)
    target = np.asarray(target)
    k = max(1, int(round(len(score) * float(fraction))))
    chosen = np.argsort(score)[-k:]
    total = float(target.sum())
    if total <= 1e-20:
        return float(fraction)
    return float(target[chosen].sum() / total)


def oracle_top_capture(target: np.ndarray, fraction: float = TOP_FRACTION) -> float:
    return top_capture(target, target, fraction)


def case_paths(root: Path, prompt_index: int, seed: int) -> dict[str, Path]:
    stem = f"p{prompt_index:02d}_s{int(seed)}"
    return {
        "draft": root / f"{stem}_draft.png",
        "core": root / f"{stem}_sihti_core.png",
        "raw": root / f"{stem}_raw.png",
        "teacher": root / f"{stem}_teacher.png",
    }


def evaluate_case(
    root: Path,
    prompt_index: int,
    case: dict,
    grid: int = GRID,
) -> dict:
    paths = case_paths(root, prompt_index, int(case["seed"]))
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing AI-G0 image(s): " + ", ".join(missing))

    draft = load_rgb(paths["draft"])
    core = load_rgb(paths["core"])
    raw = load_rgb(paths["raw"])
    teacher = load_rgb(paths["teacher"])

    target = correction_target(raw, teacher, grid)
    scores = predictor_maps(draft, core, float(case["gaussian_sigma"]), grid)

    predictors = {}
    for name, score in scores.items():
        predictors[name] = {
            "spearman": safe_spearman(score, target),
            "top25_capture": top_capture(score, target, TOP_FRACTION),
        }

    return {
        "prompt": case["prompt"],
        "seed": int(case["seed"]),
        "prompt_index": int(prompt_index),
        "mean_correction_mse": float(target.mean()),
        "oracle_top25_capture": oracle_top_capture(target, TOP_FRACTION),
        "predictors": predictors,
    }


def aggregate(cases: list[dict]) -> dict:
    result = {}
    for name in PREDICTORS:
        spearman = np.asarray([c["predictors"][name]["spearman"] for c in cases])
        capture = np.asarray([c["predictors"][name]["top25_capture"] for c in cases])
        result[name] = {
            "cases": int(len(cases)),
            "mean_spearman": float(spearman.mean()),
            "median_spearman": float(np.median(spearman)),
            "mean_top25_capture": float(capture.mean()),
            "median_top25_capture": float(np.median(capture)),
            "mean_top25_capture_over_random": float(capture.mean() - TOP_FRACTION),
        }

    baselines = ("gaussian_residue", "edge_energy", "local_variance")
    sihti_capture = np.asarray([
        c["predictors"]["sihti_residue"]["top25_capture"] for c in cases
    ])
    best_baseline_capture = np.asarray([
        max(c["predictors"][b]["top25_capture"] for b in baselines)
        for c in cases
    ])
    win_count = int(np.sum(sihti_capture > best_baseline_capture))

    best_mean_spearman = max(result[b]["mean_spearman"] for b in baselines)
    best_mean_capture = max(result[b]["mean_top25_capture"] for b in baselines)

    result["claim_receipt"] = {
        "sihti_minus_best_baseline_mean_spearman": float(
            result["sihti_residue"]["mean_spearman"] - best_mean_spearman
        ),
        "sihti_minus_best_baseline_mean_top25_capture": float(
            result["sihti_residue"]["mean_top25_capture"] - best_mean_capture
        ),
        "sihti_top25_case_wins_vs_best_baseline": win_count,
        "required_margin": 0.05,
        "required_case_wins": 6,
        "router_claim_pass": bool(
            result["sihti_residue"]["mean_spearman"] >= best_mean_spearman + 0.05
            and result["sihti_residue"]["mean_top25_capture"] >= best_mean_capture + 0.05
            and win_count >= 6
        ),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gate0-summary",
        type=Path,
        default=Path("results/ai_gate0/summary.json"),
    )
    ap.add_argument(
        "--image-root",
        type=Path,
        default=Path("results/ai_gate0"),
    )
    ap.add_argument("--grid", type=int, default=GRID)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/ai_gate1_summary.json"),
    )
    args = ap.parse_args()

    gate0 = json.loads(args.gate0_summary.read_text(encoding="utf-8"))

    prompt_order = []
    for case in gate0["cases"]:
        if case["prompt"] not in prompt_order:
            prompt_order.append(case["prompt"])
    prompt_index = {prompt: i + 1 for i, prompt in enumerate(prompt_order)}

    cases = [
        evaluate_case(args.image_root, prompt_index[case["prompt"]], case, args.grid)
        for case in gate0["cases"]
    ]
    payload = {
        "source_gate": "AI-G0",
        "grid": int(args.grid),
        "tiles_per_case": int(args.grid * args.grid),
        "top_fraction": TOP_FRACTION,
        "target": "per-tile MSE between raw cheap refinement and four-step teacher",
        "cases": cases,
        "aggregate": aggregate(cases),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload["aggregate"], indent=2))
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
