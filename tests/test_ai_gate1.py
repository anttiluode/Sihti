import numpy as np

from ai_gate1_residue_router import (
    correction_target,
    predictor_maps,
    safe_spearman,
    tile_reduce,
    top_capture,
)


def test_tile_reduce_preserves_spatial_cells():
    x = np.arange(64, dtype=float).reshape(8, 8)
    y = tile_reduce(x, grid=2)
    assert y.shape == (4,)
    assert np.allclose(y, [13.5, 17.5, 45.5, 49.5])


def test_router_metrics_reward_aligned_predictor():
    target = np.arange(1, 65, dtype=float)
    aligned = target.copy()
    reversed_score = target[::-1]

    assert safe_spearman(aligned, target) > 0.99
    assert safe_spearman(reversed_score, target) < -0.99
    assert top_capture(aligned, target, 0.25) > top_capture(
        reversed_score, target, 0.25
    )


def test_predictor_maps_have_one_score_per_grid_tile():
    rng = np.random.default_rng(1)
    draft = rng.random((32, 32, 3))
    core = draft * 0.9
    scores = predictor_maps(draft, core, gaussian_sigma=0.5, grid=4)

    assert set(scores) == {
        "sihti_residue",
        "gaussian_residue",
        "edge_energy",
        "local_variance",
    }
    assert all(value.shape == (16,) for value in scores.values())


def test_correction_target_matches_grid():
    a = np.zeros((64, 64, 3))
    b = a.copy()
    b[:32, :32] = 1.0
    target = correction_target(a, b, grid=2)

    assert np.argmax(target) == 0
    assert target[0] > 0
    assert np.all(target[1:] == 0)
