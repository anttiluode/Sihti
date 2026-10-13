import numpy as np

import eye_persistence as E


def test_pyramid_rebuilds_exactly():
    x = np.random.default_rng(0).random((64, 64, 3))
    top, bands = E.pyramid(x)
    assert top.shape == (4, 4, 3)
    assert np.abs(E.pyramid_rebuild(top, bands) - x).max() < 1e-12


def test_quantise_roundtrip_and_byte_count():
    x = np.random.default_rng(1).random((8, 8, 3))
    nb, xq = E.quantise(x, 1 / 128)
    assert np.abs(xq - x).max() <= 0.5 / 128 + 1e-12
    assert nb > 4


def test_eye_patch_is_normalised_and_identifies_itself():
    rng = np.random.default_rng(2)
    lm = np.zeros((68, 2))
    lm[36:48] = rng.uniform(20, 40, size=(12, 2))
    pats = np.stack([E.eye_patch(rng.random((64, 64, 3)), lm) for _ in range(5)])
    assert np.allclose(np.linalg.norm(pats, axis=1), 1)
    acc, c_own = E.identify(pats, pats)
    assert acc == 1.0 and np.allclose(c_own, 1)


def test_sieve_on_face_crop_is_lossless():
    x = np.random.default_rng(3).random((16, 16, 3))
    for pur in ("self", "lattice"):
        res = E.sieve(x, pur)
        assert np.abs(res.reconstruct() - x).max() < 1e-12
