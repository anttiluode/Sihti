import numpy as np

import trained_purifier as T


class _F:
    pass


def _face(seed=0):
    rng = np.random.default_rng(seed)
    rgb = rng.random((T.N, T.N, 3))
    fc = _F()
    fc.x0 = rgb.reshape(-1, 3)
    phi = T.features(rgb)
    fc.D = (phi[T.G.i] - phi[T.G.j]) ** 2
    fc.phihat = (phi - phi.mean(0)) / (phi.std(0) + 1e-9)
    fc.target = {d: float(((fc.x0 - T.run_steps(fc, T.BASE_T, T.BASE_U, d)) ** 2).sum())
                 for d in (8,)}
    return fc


def test_baseline_parameters_reproduce_the_own_eq_step_count():
    fc = _face()
    x, n = T.run_matched(fc, T.BASE_T, T.BASE_U, 8)
    assert n == 8
    assert np.abs(x - T.run_steps(fc, T.BASE_T, T.BASE_U, 8)).max() < 1e-12


def test_energy_matching_is_exact_for_large_steps():
    fc = _face(1)
    u = np.zeros(6)
    u[0] = 3.0                                  # big gate -> big steps
    x, _ = T.run_matched(fc, T.BASE_T, u, 8)
    e = float(((fc.x0 - x) ** 2).sum())
    assert abs(e / fc.target[8] - 1) < 1e-9
