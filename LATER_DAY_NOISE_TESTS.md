# Later-day null tests — what does Sihti see in noise?

Status: **idea frozen for later work, not a result and not a preregistered gate**.

The motivating question is simple:

> If Sihti is given an input with no intended objects, what persistent structure does each purifier manufacture, preserve, or expose?

This is a null test on the interpretation of the slow modes. It is especially important for the image's own EQ. A random image still defines a random weighted spatial graph. Accidental same-colour neighborhoods can create strong local couplings and accidental boundaries can create weak ones. Those bottlenecks may have long diffusion lifetimes and can therefore look "object-like" even though the source was noise.

The safe claim to protect is not **slow mode = object**, but **slow mode = region or direction that is persistent under the chosen operator**. Objecthood needs extra evidence.

## Visualizer

Run:

```bash
pip install -r requirements.txt
python noise_lab.py
```

Optionally use a photograph as the reference for the phase-scrambled null:

```bash
python noise_lab.py photo.jpg
```

The app compares four source families:

1. **IID RGB noise** — no intended spatial structure.
2. **Blurred noise** — short-range correlation without objects.
3. **1/f-power noise** — broad scale structure without designed objects.
4. **Phase-scrambled reference** — keeps each channel's Fourier-magnitude shape up to display rescaling while destroying most spatial phase organization.

and three purifiers: **Own EQ**, **Lattice**, and **Sigh high-pass**.

The processed image panels use auto-contrast where needed. The energy plot is the absolute-amplitude check. A crisp-looking late mode with vanishing RMS is not evidence of a strong persistent object.

## N0 — IID null

Generate many IID seeds at a fixed working size and sweep the own-EQ colour scale `sigma`.

Record slow-mode eigenvalues/lifetimes, blank-lattice share, eigengap-selected segment count, segment-size distribution, core and residue RMS versus depth, and graph weight/degree statistics.

Question: **Does the own-EQ routinely carve visually persuasive low-lattice-share regions out of pure noise?**

If yes, those modes are evidence for graph persistence, not objecthood.

The expected qualitative sigma sweep is worth attacking rather than assuming:

```text
large sigma       -> image graph approaches the blank lattice
intermediate      -> random weighted bottlenecks may create accidental islands
small sigma       -> unlike neighbours decouple; many distinctions can become long-lived
```

The affinity floor means the graph is never literally disconnected.

## N1 — correlation without objects

Repeat the same measurements for Gaussian-blurred IID noise. This asks whether local correlation alone is enough to make the slow-mode panel look object-like.

## N2 — 1/f-power null

Generate stationary random fields with approximately `power(f) ~ 1/f`. This is a harder null because natural images have strong scale structure. If Sihti's slow modes become more natural-looking here, test whether spectrum/local smoothness alone explains the change.

## N3 — phase-scrambled photograph

For each natural image, preserve the per-channel Fourier magnitude shape, replace phase with a random Hermitian phase field, rescale to a valid display range, and run original/scrambled images with identical Sihti parameters.

Do **not** call this exactly spectrum-identical after rescaling. The useful constraint is that non-DC magnitude *shape* is preserved while recognizable spatial organization is destroyed.

Compare affinity statistics, slow-mode lifetimes, lattice share, eigengap structure, residue/core energy curves, and segmentation complexity.

## N4 — operator fingerprint

Feed the **same IID seed** to all three purifiers:

```text
IID input
   |
   +-- Sigh high-pass -> preferred Fourier / Nyquist structure
   +-- lattice        -> slow spatial substrate modes
   +-- own EQ         -> modes of the particular random weighted graph
```

If the endpoint is strongly purifier-specific, that is a direct reminder: **structure in a recursive endpoint can be a fingerprint of the operator.**

## N5 — stability attacker

An accidental region in one noise draw should not be promoted merely because it is long-lived. Perturb the same seed slightly and measure subspace / segmentation stability, then compare that with natural images under matched perturbations.

Useful comparisons: pixel-noise perturbation, tiny colour rotation, one-pixel translation, and an independent seed.

## Kill / correction conditions

The future noise study should downgrade the object language if any of these survive broad testing:

- IID or 1/f noise produces slow-mode "objects" as strong and stable as natural images;
- phase scrambling leaves the relevant slow-mode statistics essentially unchanged;
- apparent late structure is primarily an auto-contrast artifact while absolute energy has vanished;
- the own-EQ advantage is explained by generic local correlation rather than scene organization.

A negative result is useful. It would sharpen Sihti from an object story into what the code already guarantees more safely:

> the purifier defines persistence, and the sieve records where each distinction leaves the trajectory.

## Why this belongs next to Sigh

Sigh already supplies the clean warning: a recursive endpoint can expose the maximal-gain mode of an operator even when that mode was not perceptually important in the input.

The noise null asks the complementary Sihti question:

> What bled out while the operator made that endpoint, and when did apparently coherent structure emerge?

That is why the null is worth keeping even if every "object" interpretation fails.
