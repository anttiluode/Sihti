# Sihti AI-G1 — does residue say where more diffusion work is needed?

Status: **run in GitHub Actions on all 9 AI-G0 cases; Sihti router-signal claim FAIL.**

## Why this gate exists

AI-G0 killed the first acceleration story.

Across 9 SDXL-Turbo prompt/seed cases, the cheap Sihti-core refinement did not
beat the cheap raw-draft refinement or the matched Gaussian control on the
predeclared representation metrics. It was also slower end to end than direct
512×512 generation.

That means we should **not** keep trying to make "refine the core instead of the
draft" win by tuning it.

The remaining useful hypothesis is different:

> the Sihti residue might be a *map of where the cheap representation is
> insufficient*, even if the core is not a better global conditioning image.

That is exactly the bridge to WhatToLookAt: uncertainty/error should buy local
compute.

## Target

For every uploaded AI-G0 case, split the image into an 8×8 spatial grid.

The target for each tile is:

```text
correction energy = MSE(raw cheap 2-step refinement, 4-step teacher)
```

This does **not** call the teacher ground truth. It asks a narrower question:
where does additional diffusion refinement change the image?

## Predictors

All predictors are available before looking at the teacher:

1. **Sihti residue energy** — `RMS(draft - core)`;
2. **matched Gaussian residual** — `RMS(draft - Gaussian(draft))`;
3. **draft edge energy**;
4. **draft local RGB variance**.

The Gaussian residual is the important attacker. If it predicts the teacher
corrections just as well, then the useful routing signal is ordinary smoothing
error, not Sihti.

## Primary metrics

Per image:

- Spearman correlation between predictor score and correction energy;
- fraction of total correction energy captured by the **top 25%** predicted
  tiles.

The random-location expectation for top-25% capture is 25%. The oracle ranking
is recorded only as a ceiling.

## Claim rule fixed before running

Sihti earns a **router-signal** claim only if all three conditions hold:

1. mean Spearman exceeds the best simple baseline by at least **0.05**;
2. mean top-25% capture exceeds the best simple baseline by at least **0.05**;
3. Sihti beats the best baseline on top-25% capture in at least **6 of 9** cases.

If it fails, do not build a GPU tile router around Sihti residue.

If it passes, AI-G2 will spend extra diffusion only on high-residue crops and
compare against same-compute random tiles, edge-selected tiles, Gaussian-selected
tiles, and full-frame refinement.

## Run

CPU only:

```bash
pip install numpy scipy pillow
python ai_gate1_residue_router.py
```

Output:

```text
results/ai_gate1_summary.json
```


## Result

The gate ran from the committed AI-G0 images in GitHub Actions.

| predictor | mean Spearman | mean top-25% correction capture |
|---|---:|---:|
| Sihti residue | 0.5869 | 38.54% |
| matched Gaussian residual | 0.6191 | 42.99% |
| **draft edge energy** | **0.6805** | **45.72%** |
| local variance | 0.5978 | 44.70% |

Randomly choosing 25% of tiles would capture 25% of correction energy in
expectation. So the *general* routing idea has signal, but Sihti residue is not
the best signal.

Against the best simple baseline:

```text
Sihti mean Spearman margin       -0.0936
Sihti mean top-25% capture       -0.0719
Sihti top-25% case wins           0 / 9
required                           +0.05, +0.05, >=6/9
```

Therefore:

> **AI-G1 FAILS the Sihti-specific router claim.**

This is a useful negative result. Two separate AI gates now reject the tempting
story that Sihti itself makes SDXL-Turbo cheaper:

1. the purified core is not a better global cheap conditioning image;
2. the residue is not a better local compute-allocation signal than trivial
   image observables.

But a cross-repo result survives: edge energy concentrates **45.72%** of the
teacher-correction energy into only **25%** of the tiles. The next compute-routing
experiment therefore belongs more naturally in **WhatToLookAt**, using the
cheapest winning observable rather than forcing Sihti to win.
