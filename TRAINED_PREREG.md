# Pre-registration: can a *trained* purifier keep the eyes alive longer, at the same amount of purification?

Written and committed **before** any trained-purifier score was computed. It follows `EYE_PREREG.md` (same faces, same split, same readers) and Sol's proposal: train the purifier's coupling rules, not only the order of residues.

## Question

The fixed own EQ keeps 25/25 eye identity through depth 16 and loses it by depth 32. Can a purifier whose couplings are *learned* on the 18 train faces keep more eye identity on the 25 held-out faces?

Two constraints apply. The trained purifier must not purify less, and it must not be told where the eyes are.

## The learnable purifier

This is the same lazy random walk as Sihti (laziness 0.5, radius 2), on a graph built once from the input image. The edge weight is

    w_ij = exp(-Σ_k θ_k (φ_ik − φ_jk)²) · exp(−|p_i − p_j|² / 8) · g_i · g_j,    g_i = exp(clip(u · φ̂_i, −3, 3))

The per-pixel features φ are L, a, b, Sobel gradient magnitude of L, 3×3 local std of L, and L minus its 5×5 mean, all in Lab units. φ̂ is φ standardised with constants computed once from the train pixels.

There are 12 parameters: six θ_k (log-parameterised) and six u_k. **Initialisation = the fixed own EQ exactly**: θ_Lab = 1/128, the other θ ≈ 0 (1e-6), u = 0. There is **no position input**, so the purifier can only learn *what kind of local structure* to protect.

## Equal purification (the anti-cheat rule)

For each face, the fixed own EQ run to depth d defines the removed energy E\*(d) = ‖x₀ − x_d‖².

Every purifier, fixed or trained, is run step by step until its removed energy first reaches E\*(d). The final state is linearly interpolated between the last two steps so that the removed energy equals E\*(d) exactly. Removing less is therefore impossible; learning can only decide **where** the energy is removed.

## Training

- **Teacher:** the fixed eye-identity reader from `EYE_PREREG.md`, i.e. correlation with the face's own original eye patch. It is a hand-defined reader, **not** a neural image model; this is stated plainly in the results.
- **Objective:** maximise the mean train `c_own` at d = 32, under equal purification.
- **Optimiser:** scipy Powell from the baseline initialisation, at most 800 function evaluations, a single run, with no restarts and no selection.
- **Specificity control:** an identical second run trains on the **mouth** reader instead. The mouth patch is the bounding box of landmarks 48–67, widened 1.3×, height half the width, sampled at 32×16.

## Primary verdict (test, 25 faces, d = 32, equal purification)

**P1 PASS** iff the eye-trained purifier, compared with the fixed own EQ:
- raises 25-way eye identification by **≥ 4 faces**, and
- has a higher `c_own` on **≥ 18 of 25** faces (one-sided sign test p ≈ 0.02).

## Secondary (all reported, whatever the outcome)

- **P2 double dissociation:** the eye-trained purifier improves the eyes more than the mouth-trained one does, *and* the mouth-trained purifier improves the mouth more than the eye-trained one does. If both purifiers improve both regions about equally, they learned "keep dark high-contrast detail" in general, not anything specific to one feature.
- **P3 depth transfer:** the same comparison at equal purification for d = 16 and d = 64. The purifiers are not retrained.
- **P4 where the energy went:** the fraction of removed energy inside the eye box, for trained vs fixed.
- **P5 compressibility:** zlib bytes of the purified state (q = 1/64), trained vs fixed. Equal energy removed does not mean equally simple.
- **Lossless check:** residues plus the core still sum exactly to the input.

## Limits, stated now

There are 18 train faces for 12 parameters, so overfitting is possible; the 25 held-out faces are the check. The teacher is a correlation reader rather than a perceptual network, because no deep-learning runtime is available in this environment. A pass would show that the coupling rules can be taught to protect a feature without being told where it is. It would not show a product.
