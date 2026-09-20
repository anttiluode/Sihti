import numpy as np

from noise_lab import iid_noise, blurred_noise, one_over_f_power_noise, phase_scramble


def test_noise_generators_are_deterministic_and_bounded():
    for fn in (iid_noise, blurred_noise, one_over_f_power_noise):
        a = fn(24, 32, 7)
        b = fn(24, 32, 7)
        assert a.shape == (24, 32, 3)
        assert np.array_equal(a, b)
        assert np.isfinite(a).all()
        assert 0.0 <= float(a.min()) <= float(a.max()) <= 1.0


def test_phase_scramble_preserves_non_dc_magnitude_shape():
    x = iid_noise(24, 32, 3)
    y = phase_scramble(x, 11)
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    assert 0.0 <= float(y.min()) <= float(y.max()) <= 1.0

    for c in range(3):
        ax = np.abs(np.fft.fft2(x[..., c])).ravel()[1:]
        ay = np.abs(np.fft.fft2(y[..., c])).ravel()[1:]
        corr = np.corrcoef(ax, ay)[0, 1]
        assert corr > 0.999999
