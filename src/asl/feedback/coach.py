from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asl.config import NUM_HAND_LANDMARKS
from asl.data.features import normalize_landmarks, sample_indices
from asl.feedback.dtw import dtw_align
from asl.feedback.reference import REF_LEN, _wrist_track

# finger joint chains within one hand's 21 landmarks, mediapipe topology
_FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
            (13, 14, 15, 16), (17, 18, 19, 20)]


@dataclass
class Feedback:
    gloss: str
    overall: float
    location: float
    movement: float
    handshape: float


def _joint_angles(hand: np.ndarray) -> np.ndarray:
    angles = []
    for a, b, c, d in _FINGERS:
        for p, q, r in ((a, b, c), (b, c, d)):
            v1, v2 = hand[q] - hand[p], hand[r] - hand[q]
            cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angles.append(float(np.arccos(np.clip(cos, -1, 1))))
    return np.array(angles)


def _score(err: float, tolerance: float) -> float:
    return float(100.0 * max(0.0, 1.0 - err / tolerance))


def evaluate_attempt(attempt_landmarks: np.ndarray, attempt_pose_mask: np.ndarray,
                     reference: np.ndarray, gloss: str) -> Feedback:
    sel = sample_indices(len(attempt_landmarks), REF_LEN, train=False)
    att = normalize_landmarks(attempt_landmarks[sel], attempt_pose_mask[sel])
    ref = reference

    _, path = dtw_align(_wrist_track(att), _wrist_track(ref))
    ai = np.array([p[0] for p in path])
    ri = np.array([p[1] for p in path])

    loc_errs, move_errs, shape_errs = [], [], []
    for slot in (0, 1):
        off = slot * NUM_HAND_LANDMARKS
        wrist_a = att[ai, off, :2]
        wrist_r = ref[ri, off, :2]
        loc_errs.append(float(np.linalg.norm(wrist_a - wrist_r, axis=1).mean()))
        va, vr = np.diff(wrist_a, axis=0), np.diff(wrist_r, axis=0)
        move_errs.append(float(np.linalg.norm(va - vr, axis=1).mean()))
        for i, j in zip(ai, ri):
            ang_a = _joint_angles(att[i, off:off + NUM_HAND_LANDMARKS])
            ang_r = _joint_angles(ref[j, off:off + NUM_HAND_LANDMARKS])
            shape_errs.append(float(np.abs(ang_a - ang_r).mean()))

    location = _score(float(np.mean(loc_errs)), tolerance=0.5)
    movement = _score(float(np.mean(move_errs)), tolerance=0.1)
    handshape = _score(float(np.mean(shape_errs)), tolerance=0.8)
    overall = (location + movement + handshape) / 3
    return Feedback(gloss=gloss, overall=overall, location=location,
                    movement=movement, handshape=handshape)
