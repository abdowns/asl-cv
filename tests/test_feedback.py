import numpy as np

from asl.config import NUM_POINTS
from asl.feedback.coach import evaluate_attempt
from asl.feedback.dtw import dtw_align
from asl.feedback.reference import REF_LEN
from asl.data.features import normalize_landmarks


def test_dtw_identical_is_zero():
    a = np.random.default_rng(0).normal(size=(20, 4))
    cost, path = dtw_align(a, a)
    assert cost < 1e-9
    assert path[0] == (0, 0) and path[-1] == (19, 19)


def test_dtw_handles_time_warp():
    t = np.linspace(0, 2 * np.pi, 30)
    a = np.stack([np.sin(t), np.cos(t)], axis=1)
    b = a[::2]
    cost, _ = dtw_align(a, b)
    rng = np.random.default_rng(1)
    cost_rand = dtw_align(a, rng.normal(size=(15, 2)))[0]
    assert cost < 0.2 * cost_rand


def _synthetic_sign(seed: int, T: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lm = np.zeros((T, NUM_POINTS, 3), dtype=np.float32)
    pose = np.array([[.5, .2, 0], [.42, .35, 0], [.58, .35, 0],
                     [.38, .5, 0], [.62, .5, 0], [.36, .6, 0], [.64, .6, 0],
                     [.45, .75, 0], [.55, .75, 0]], dtype=np.float32)
    lm[:, 42:] = pose
    center = np.array([.6, .45]) + rng.normal(0, 0.01, 2)
    phase = np.linspace(0, np.pi, T)
    offsets = rng.normal(0, 0.02, (21, 3)).astype(np.float32)
    for t in range(T):
        wrist = np.array([center[0] + 0.1 * np.cos(phase[t]),
                          center[1] + 0.1 * np.sin(phase[t]), 0])
        lm[t, 21:42] = wrist + offsets
    return lm, np.ones(T, dtype=bool)


def test_coach_scores_matching_attempt_higher():
    ref_raw, ref_pm = _synthetic_sign(seed=0, T=REF_LEN)
    reference = normalize_landmarks(ref_raw, ref_pm)

    good_raw, good_pm = _synthetic_sign(seed=0)
    fb_good = evaluate_attempt(good_raw, good_pm, reference, "test")

    bad_raw, bad_pm = _synthetic_sign(seed=0)
    bad_raw[:, 21:42, 1] += 0.35
    fb_bad = evaluate_attempt(bad_raw, bad_pm, reference, "test")

    assert fb_good.overall > fb_bad.overall + 10
    assert fb_good.location > fb_bad.location
    assert any("higher" in t or "Position" in t for t in fb_bad.tips)


def test_coach_flags_missing_hand():
    ref_raw, ref_pm = _synthetic_sign(seed=0, T=REF_LEN)
    reference = normalize_landmarks(ref_raw, ref_pm)
    empty_raw, empty_pm = _synthetic_sign(seed=0)
    empty_raw[:, 21:42] = 0.0
    fb = evaluate_attempt(empty_raw, empty_pm, reference, "test")
    assert fb.overall < 40
    assert fb.tips
