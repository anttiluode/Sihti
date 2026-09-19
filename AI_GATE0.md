# Sihti AI-G0 — causal sieve + SDXL-Turbo compute gate

Status: **manual GPU gate, ready to run; no result claimed yet.**

The first SDXL prototype had a hidden identity: the purifier produced

```text
residue = draft - core
```

and then refined

```text
core + residue
```

with both gains fixed to 1. Therefore the diffusion model received the original
draft regardless of how dramatically the Core/Residue panels changed.

AI-G0 fixes that before asking about speed.

## A. Causal sanity

For every draft:

```text
max_abs(core + residue - draft) <= 2e-6
```

The live app exposes residue-back alpha:

```text
alpha=0.0  core only
alpha=0.5  half residue
alpha=1.0  exact draft identity
```

If changing alpha does not visibly/quantitatively change the refined output,
the sieve has not earned a causal role.

## B. Representation test

Generate one 256x256 SDXL-Turbo draft and derive four 512x512 img2img
conditioning images from that same draft:

1. raw draft;
2. Sihti core;
3. core + 0.5 residue;
4. Gaussian blur whose draft-distance is matched to the Sihti core.

The expensive teacher is the raw draft with a larger refinement-step budget.
Every cheap arm gets the same prompt, cheap step budget, strength and refinement
noise seed.

Metrics against the teacher:

- full-image PSNR;
- 64x64 layout PSNR;
- edge-map correlation.

**Representation claim:** Sihti core must beat both raw cheap refinement and the
matched Gaussian control on the predeclared structure metrics across prompts /
seeds. Merely being smoother is not enough.

## C. Speed test

Record separately:

- draft latency;
- sieve latency;
- cheap refinement latency;
- direct 512x512 text-to-image latency;
- estimated route latency = draft + required preprocessing + refinement.

**Speed claim:** a route only wins if it preserves the chosen quality frontier
and has lower end-to-end wall time. Residue sparsity, fewer visible details, or
a cheaper substage do not count by themselves.

## Run

```bash
pip install -r requirements-ai.txt
python ai_gate0_sdxl_turbo.py --local-only
```

Heavier:

```bash
python ai_gate0_sdxl_turbo.py --local-only --seeds 100 101 102
```

Return `results/ai_gate0/summary.json` plus the contact sheets. The next gate
will be chosen from the result rather than assumed in advance.

Likely branches:

- If core wins quality at the same cheap budget: try reducing refinement work
  further.
- If Gaussian matches core: the useful object is ordinary smoothing, not Sihti.
- If core preserves layout but loses texture: route compute by residue energy
  instead of conditioning the whole frame on the core.
- If none beats raw: keep the live instrument as a decomposition/editing tool
  and do not claim acceleration.
