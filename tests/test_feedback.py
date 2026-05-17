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


def _synthetic_sign(T: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    lm = np.zeros((T, NUM_POINTS, 3), dtype=np.float32)
    pose = np.array([[.5, .2, 0], [.42, .35, 0], [.58, .35, 0],
                     [.38, .5, 0], [.62, .5, 0], [.36, .6, 0], [.64, .6, 0],
                     [.45, .75, 0], [.55, .75, 0]], dtype=np.float32)
    lm[:, 42:] = pose
    phase = np.linspace(0, np.pi, T)
    offsets = rng.normal(0, 0.02, (21, 3)).astype(np.float32)
    for t in range(T):
        wrist = np.array([0.6 + 0.1 * np.cos(phase[t]),
                          0.45 + 0.1 * np.sin(phase[t]), 0])
        lm[t, 21:42] = wrist + offsets
    return lm, np.ones(T, dtype=bool)


def test_coach_scores_matching_attempt_higher():
    ref_raw, ref_pm = _synthetic_sign(T=REF_LEN)
    reference = normalize_landmarks(ref_raw, ref_pm)

    good_raw, good_pm = _synthetic_sign()
    fb_good = evaluate_attempt(good_raw, good_pm, reference, "test")

    bad_raw, bad_pm = _synthetic_sign()
    bad_raw[:, 21:42, 1] += 0.35
    fb_bad = evaluate_attempt(bad_raw, bad_pm, reference, "test")

    assert fb_good.overall > fb_bad.overall
    assert fb_good.location > fb_bad.location
