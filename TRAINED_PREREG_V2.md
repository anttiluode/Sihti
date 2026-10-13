# Pre-registration v2: trained purifier under an equal-*storage* budget

Written and committed **before** any v2 score was computed, after v1 (`TRAINED_PREREG.md`, results in `results/trained/trained_results.json`) was found to pass through a loophole.

## What v1 showed, and why v2 exists

In v1, purifiers had to remove equal **L2 energy**. The optimiser learned a gate that pulls each pixel toward its brightest neighbours. That spends the energy quota in about two steps as a brightness shift. The zero-mean, unit-norm correlation readers cannot see such a shift, so eye identity "survived" perfectly while the image became *less* simple: 7,242 vs 4,111 compressed bytes.

Equal energy was the wrong budget. v2 uses the budget Sol originally named: **storage cost**.

## The only changes from v1

1. **Budget = equal compressed size.** For each face, the target is the zlib-9 byte size of the fixed own EQ's state at depth d, quantised at q = 1/64. Every purifier runs step by step and stops at the **first** step whose quantised state compresses to ≤ that target. Steps are integers, with no interpolation and a cap of 512 steps. A face that never reaches the target uses its 512-step state and is counted in a reported "unreached" total.
2. **An absolute-fidelity reader is added to the verdict.** It is the RMSE between the *raw* (unnormalised) luma eye patch of the purified state and that of the original. Brightness or contrast changes now cost something.

Everything else is unchanged from v1: the same purifier family and 12 parameters, the same baseline initialisation, the same features, the same faces and split, the same objective (mean train eye `c_own` at d = 32), and Powell with at most 800 evaluations in a single run. The mouth-trained control is also unchanged.

## Primary verdict (test, 25 faces, d = 32, equal bytes)

**P1 PASS** iff the eye-trained purifier, compared with the fixed own EQ:
- raises 25-way eye identification by **≥ 4 faces**,
- has a higher eye `c_own` on **≥ 18 of 25** faces, **and**
- has a lower raw eye-patch RMSE on **≥ 18 of 25** faces.

## Secondary (all reported)

- **P2 double dissociation**, as in v1, plus the size of each gap. A gap under 0.02 in mean `c_own` gain is read as "no specificity".
- **P3 depth transfer:** d = 16 and d = 64 at equal bytes, with no retraining.
- **Energy removed** by each purifier at equal bytes (the reverse of v1's check).
- **Steps used** and the number of unreached faces.

## Expectation stated in advance

Genuinely uncertain. The fixed own EQ is already a strong simplifier. A trained purifier can only win if it learns to spend the byte budget *elsewhere*, by flattening non-eye regions harder. With 12 parameters and no position input it may not manage this, and in that case P1 fails.
