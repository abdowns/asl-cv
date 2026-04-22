import numpy as np

from asl.config import NUM_POINTS
from asl.data.features import (
    FEATURE_DIM, build_features, normalize_landmarks, sample_indices,
)


def _fake_landmarks(T=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, (T, NUM_POINTS, 3)).astype(np.float32)


def test_feature_shape_and_scale_invariance():
    lm = _fake_landmarks()
    mask = np.ones(len(lm), dtype=bool)
    f1 = build_features(lm, mask)
    assert f1.shape == (20, FEATURE_DIM)
    f2 = build_features(lm * 0.5 + 0.2, mask)
    np.testing.assert_allclose(f1, f2, atol=1e-4)


def test_normalize_keeps_missing_points_zero():
    lm = _fake_landmarks()
    lm[:, :21] = 0.0
    norm = normalize_landmarks(lm, np.ones(len(lm), dtype=bool))
    assert np.abs(norm[:, :21]).sum() == 0.0


def test_sample_indices():
    idx = sample_indices(100, 32, train=False)
    assert len(idx) == 32
    assert idx[0] == 0 and idx[-1] <= 99
    idx = sample_indices(10, 32, train=False)
    assert len(idx) == 32
