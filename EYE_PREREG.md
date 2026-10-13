# Pre-registration: do the eyes have a lifetime, and can Sihti's residues bring *those* eyes back cheaply?

Written and committed **before** any eye score was computed. Before this commit, the only things looked at were the annotation files' face counts and box sizes. No crop, residue or score had been seen.

## Question

Antti's totem observation: during purification the statues' faces lose their eyes at some depth. Three questions follow:

1. **Persistence.** At what purification depth do the eyes stop being *these* eyes?
2. **Recoverability.** Which residues bring them back?
3. **Usefulness.** Can a residue order *learned* on other faces recover the specific eyes at fewer stored bytes than ordinary codecs?

We separate **remembering** (recovering the particular eyes) from **imagining** (recovering eye-like structure). A codec that produces generic eyes fails the first reader and passes the second.

## Data

- dlib's example faces, `github.com/davisking/dlib/examples/faces`. Each face has 68 hand-annotated landmarks; the eyes are points 36–47.
- dlib's own split is kept: **train** = `training_with_face_landmarks.xml` (4 photos, 18 faces), **test** = `testing_with_face_landmarks.xml` (5 photos, 25 faces).
- Each face is cropped as a square centred on its box, side = 1.6 × max(box w, h), with reflected padding at borders. It is resized with Lanczos to **64 × 64** RGB in [0, 1], and the landmarks are transformed with it.
- **Eye patch:** the bounding box of points 36–47, widened by 40%, with height set to half the width. It is bilinearly sampled to 32 × 16 luma, then normalised to zero mean and unit norm.

## Readers (fixed)

- **Own-eye identity (remembering):** take the eye patch of a reconstruction and correlate it with the *original* eye patches of all 25 test faces. The reconstruction is correct if its own original is the best match. Reported as identification accuracy and as `c_own`, the correlation with its own original (1 for the original).
- **Generic eye (imagining):** correlation with the mean of the 18 *train* original eye patches. Reported relative to the same face's original.

## Purifiers and depths

- Sihti **own EQ** (`self`, radius 2, σ 8 Lab, laziness 0.5: the repo defaults) and **lattice** (plain diffusion), for contrast.
- Octave depths 0, 1, 2, 4, …, 512, giving 10 residues plus a core.

## 1. Persistence (descriptive)

For each face and purifier, read the *core-only* state at every recorded depth. Report identification accuracy vs depth, `c_own` vs depth and the generic score vs depth. The per-face **eye half-life** is the first depth where `c_own < 0.5`.

## 2–3. Bytes-matched recovery

Every stored array is quantised with step `q ∈ {1/16, 1/32, 1/64, 1/128}`, rounded to int16 and compressed with zlib (level 9), plus a 4-byte header. Cost is total bytes. Decoding is the sum of the dequantised arrays.

| method | configurations |
|---|---|
| **Sihti natural** (own EQ) | core + the *k* deepest residues, k = 0…10, × q |
| **Sihti learned** (own EQ) | core + the first *k* residues in an order learned on **train** faces only, k = 0…10, × q |
| **Sihti lattice** | core + *k* deepest residues of plain diffusion, × q |
| **Laplacian pyramid** (Burt–Adelson, 5-tap binomial, 64→32→16→8→4) | top 4×4 + the *k* coarsest band-pass levels, k = 0…4, × q |
| **JPEG** (Pillow, optimize, default subsampling) | quality {2, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90} |

The **learned order** is greedy on train. Start from the core at q = 1/64. At each step, add the remaining residue with the largest mean gain in train `c_own` per added byte. The order is frozen, then applied to test at every q.

For each method, the **envelope** gives the best test accuracy reachable at ≤ B bytes, over all of its configurations. **B92** is the smallest B at which the envelope reaches 23/25 = 0.92.

## Verdicts, fixed now

- **V1, learning helps:** pass iff B92(Sihti learned) ≤ 0.9 × B92(Sihti natural).
- **V2, Sihti vs classical:** let B* = min(B92 JPEG, B92 pyramid). Sihti (the better of learned or natural) **wins** if its B92 ≤ 0.8 × B*, **loses** if it is > B*, and otherwise **ties**.
- **V3, remembering vs imagining:** report any method and budget whose generic score is ≥ 0.9 of the original's while identification accuracy is < 0.5.

**Expectation stated in advance:** V2 will most likely **lose**. Sihti's residues are full-resolution and overcomplete (11 arrays of 64×64×3), while the pyramid and JPEG are critically sampled or transform-coded. Only zlib's ability to exploit smooth residues can offset this. A loss here is still informative about *where* the eyes live.

## Limits, stated now

There are 43 small faces in 9 photos. Boxes are 37–109 px wide, so many crops are upsampled. Identity is a 25-way match within one test set. This is an afternoon pilot: a V2 win would justify a real face set; a loss ends the codec question for Sihti. One purifier setting, no tuning on test.
