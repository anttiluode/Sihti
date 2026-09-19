# Pre-registration: G3 and G4 on BSDS500

Written and committed **before** the full validation or test run. The grids and
decision rules below are also frozen in `bench_bsds.py` at this commit.

## Data and protocol

- BSDS500 (Arbeláez, Maire, Fowlkes & Malik 2011), mirror `github.com/BIDS/BSDS500`.
- Every image is area-resized so its long side is 120 px; every human segmentation is
  nearest-resized to the same grid. All methods see the same pixels.
- Metric: segmentation covering of each human segmentation by the machine
  segmentation, averaged over the 5–9 humans per image. ARI and VI are reported too.
- **ODS protocol**: each method family picks one setting for the whole *val* split
  (100 images) by mean covering, then is scored once on *test*.
- Test images `100007` and `100039` were looked at during a two-image debugging run
  of the harness and are **excluded from every test statistic** (n = 198).

## Methods and grids

| family | what it is | grid |
|---|---|---|
| `njw` | Sihti: slow modes of the image's own EQ, Ng–Jordan–Weiss k-means | radius {1.5, 3}, σ {3, 5, 8} Lab, k {2,3,4,6,8,12, eigengap} |
| `njw_scene` | same, but modes that a blank frame already has (substrate share ≥ 0.9) are skipped | same |
| `core` | Sihti: purify on the image's own EQ for 512 passes, then k-means on the core's (Lab, xy) | radius, σ as above; k {2,3,4,6,8,12,16}; position weight {0, 20} |
| `kmeans` | attacker: k-means on raw (Lab, xy) | k {2,3,4,6,8,12,16}; position weight {0,10,20,40} |
| `fh` | attacker: Felzenszwalb–Huttenlocher (scikit-image) | scale {100,300,1000,3000}; min size {10,30,60}; σ {0.5,0.8} |

The primary Sihti method is whichever of `njw`, `njw_scene`, `core` has the best val
covering. The other two are reported, not decided on.

## G3: does the self-EQ find objects?

- **G3a** passes iff, on test, the primary Sihti method has higher mean covering than
  the best `kmeans`, wins on more images, and the two-sided sign test gives p < 0.01.
  If it fails, v0 is colour clustering with extra steps.
- **G3b** is the same rule against the best `fh`.
- Expectation, stated in advance: G3b is likely to **fail**. In Arbeláez et al. 2011
  (full resolution, their own protocol) normalized cuts scored covering 0.45 against
  Felzenszwalb–Huttenlocher's 0.52. G3a is genuinely uncertain.

## G4: where do the objects go, the core or the residues?

The first formulation ("late residues of the self-EQ are more object-shaped than the
lattice's") was dropped **before** the confirmatory run. A discovery run on 30 *val*
images showed two things. First, the raw within-segment fraction rewards any smooth
field, so it cannot tell object-shaped from merely blurry. Second, with the
smoothness-controlled metric below, the object structure sits in the opposite place
from the one predicted in chat. The two debugging images above had shown the raw-metric
reversal first; that is why they are excluded.

Metric: `alignment(X) = f(X, rotated humans) − f(X, humans)`. Here `f` is the fraction
of X's energy that varies inside segments. The humans' segmentation rotated by 180°
keeps the same region sizes in the wrong places. A merely smooth field scores about 0;
only a field shaped like the objects scores high.

Discovery numbers (30 val images, radius 1.5, 512 passes):

| arm | core alignment | last-octave residue alignment |
|---|---|---|
| self σ=3 | 0.309 | 0.021 |
| self σ=5 | 0.308 | 0.117 |
| self σ=8 | 0.282 | 0.222 |
| self σ=12 | 0.227 | 0.275 |
| lattice | 0.117 | 0.235 |
| Sigh low pass | 0.228 | 0.038 |

Confirmatory hypotheses on test, using the primary method's (radius, σ):

- **G4a (the core holds the objects).** Core alignment is higher for the self-EQ than
  for the same diffusion on the blank lattice, on ≥ 70 % of test images, with sign test
  p < 0.01.
- **G4b (the residues hold the insides).** Mean alignment of the late residues
  (start depth ≥ 16) is *lower* for the self-EQ than for the lattice, on ≥ 70 % of test
  images, with p < 0.01. The self-EQ bleeds out what is inside objects; the lattice
  bleeds out the objects' edges.
- Reported, not decided: Sigh's own low-pass EQ, and a leaky self-EQ (σ = 12). Discovery
  suggests the leaky one is where a part–whole hierarchy starts to appear in the late
  residues.

## What would change the plan

- If G3a fails, the segmentation claim is dropped from the README headline. The
  instrument stays, because being lossless and showing where the objects go do not
  depend on it.
- If G4a fails, "the core is the object layout" is withdrawn from the README and from
  the app's labels.
