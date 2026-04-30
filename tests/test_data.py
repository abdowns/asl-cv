import numpy as np
import pytest

from asl.config import Config, NUM_POINTS
from asl.data import augment as aug
from asl.data.features import (
    FEATURE_DIM, build_features, normalize_landmarks, sample_indices,
    L_SHOULDER, R_SHOULDER,
)


def _fake_clip(T=20, seed=0):
    rng = np.random.default_rng(seed)
    lm = rng.uniform(0.2, 0.8, (T, NUM_POINTS, 3)).astype(np.float32)
    return {
        "landmarks": lm,
        "hand_mask": np.ones((T, 2), dtype=bool),
        "pose_mask": np.ones(T, dtype=bool),
        "crops": rng.integers(0, 255, (T, 2, 96, 96, 3), dtype=np.uint8),
    }


def test_feature_shape_and_scale_invariance():
    d = _fake_clip()
    f1 = build_features(d["landmarks"], d["pose_mask"])
    assert f1.shape == (20, FEATURE_DIM)
    scaled = d["landmarks"] * 0.5 + 0.2
    f2 = build_features(scaled, d["pose_mask"])
    np.testing.assert_allclose(f1, f2, atol=1e-4)


def test_normalize_keeps_missing_points_zero():
    d = _fake_clip()
    d["landmarks"][:, :21] = 0.0
    norm = normalize_landmarks(d["landmarks"], d["pose_mask"])
    assert np.abs(norm[:, :21]).sum() == 0.0


def test_normalize_centers_shoulders():
    d = _fake_clip()
    norm = normalize_landmarks(d["landmarks"], d["pose_mask"])
    mid = (norm[:, L_SHOULDER] + norm[:, R_SHOULDER]) / 2
    np.testing.assert_allclose(mid[:, :2], 0.0, atol=1e-5)


def test_sample_indices():
    idx = sample_indices(100, 32, train=False)
    assert len(idx) == 32 and (np.diff(idx) >= 0).all()
    idx = sample_indices(10, 32, train=False)
    assert len(idx) == 32 and idx.max() == 9
    rng = np.random.default_rng(0)
    idx = sample_indices(100, 32, train=True, rng=rng)
    assert len(idx) == 32 and (np.diff(idx) >= 0).all()


def test_horizontal_flip_involution():
    d = _fake_clip()
    dd = aug.horizontal_flip(aug.horizontal_flip(d))
    np.testing.assert_allclose(dd["landmarks"], d["landmarks"], atol=1e-6)
    np.testing.assert_array_equal(dd["crops"], d["crops"])
    np.testing.assert_array_equal(dd["hand_mask"], d["hand_mask"])


def test_flip_swaps_hands():
    d = _fake_clip()
    d["hand_mask"][:, 0] = True
    d["hand_mask"][:, 1] = False
    out = aug.horizontal_flip(d)
    assert out["hand_mask"][:, 1].all() and not out["hand_mask"][:, 0].any()
    np.testing.assert_allclose(out["landmarks"][:, 21:42, 0],
                               1.0 - d["landmarks"][:, :21, 0], atol=1e-6)


def test_temporal_crop_bounds():
    d = _fake_clip(T=40)
    rng = np.random.default_rng(0)
    out = aug.temporal_crop(d, rng)
    assert 8 <= len(out["landmarks"]) <= 40
    assert len(out["crops"]) == len(out["landmarks"])


@pytest.mark.skipif(
    not (Config().nslt_path.exists()), reason="dataset not present")
def test_split_integrity():
    from asl.data.wlasl import load_index, split_index
    cfg = Config(subset=100)
    splits = split_index(load_index(cfg))
    ids = [i.video_id for s in splits.values() for i in s]
    assert len(ids) == len(set(ids)), "video appears in multiple splits"
    labels = {i.label for i in splits["train"]}
    assert labels == set(range(100)), "train split must cover all 100 classes"
