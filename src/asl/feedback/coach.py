from __future__ import annotations

from dataclasses import dataclass, field

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
    tips: list[str] = field(default_factory=list)


def _joint_angles(hand: np.ndarray) -> np.ndarray:
    angles = []
    for a, b, c, d in _FINGERS:
        for p, q, r in ((a, b, c), (b, c, d)):
            v1, v2 = hand[q] - hand[p], hand[r] - hand[q]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                angles.append(0.0)
            else:
                angles.append(float(np.arccos(
                    np.clip(v1 @ v2 / (n1 * n2), -1, 1))))
    return np.array(angles)


def _hand_angle_seq(traj: np.ndarray, slot: int) -> np.ndarray:
    off = slot * NUM_HAND_LANDMARKS
    return np.stack([_joint_angles(f[off:off + NUM_HAND_LANDMARKS])
                     for f in traj])


def _present(traj: np.ndarray, slot: int) -> np.ndarray:
    off = slot * NUM_HAND_LANDMARKS
    return np.abs(traj[:, off:off + NUM_HAND_LANDMARKS]).sum(axis=(1, 2)) > 1e-6


def _score(err: float, tolerance: float) -> float:
    return float(100.0 * np.exp(-0.6931 * err / max(tolerance, 1e-6)))


def evaluate_attempt(attempt_landmarks: np.ndarray, attempt_pose_mask: np.ndarray,
                     reference: np.ndarray, gloss: str) -> Feedback:
    sel = sample_indices(len(attempt_landmarks), REF_LEN, train=False)
    att = normalize_landmarks(attempt_landmarks[sel], attempt_pose_mask[sel])
    ref = reference

    _, path = dtw_align(_wrist_track(att), _wrist_track(ref))
    ai = np.array([p[0] for p in path])
    ri = np.array([p[1] for p in path])

    tips: list[str] = []
    loc_errs, move_errs, shape_errs = [], [], []
    dy_sum = dx_sum = 0.0

    for slot in (0, 1):
        att_p, ref_p = _present(att, slot), _present(ref, slot)
        if not att_p.any() or not ref_p.any():
            continue

        off = slot * NUM_HAND_LANDMARKS
        wrist_a = att[ai, off, :2]
        wrist_r = ref[ri, off, :2]
        both = att_p[ai] & ref_p[ri]
        if both.any():
            diff = wrist_a[both] - wrist_r[both]
            loc_errs.append(float(np.linalg.norm(diff, axis=1).mean()))
            dx_sum += float(diff[:, 0].mean())
            dy_sum += float(diff[:, 1].mean())

            va = np.diff(wrist_a[both], axis=0)
            vr = np.diff(wrist_r[both], axis=0)
            move_errs.append(float(np.linalg.norm(va - vr, axis=1).mean()))

            ang_a = _hand_angle_seq(att, slot)[ai][both]
            ang_r = _hand_angle_seq(ref, slot)[ri][both]
            shape_errs.append(float(np.abs(ang_a - ang_r).mean()))

    location = _score(np.mean(loc_errs) if loc_errs else 1.0, tolerance=0.25)
    movement = _score(np.mean(move_errs) if move_errs else 1.0, tolerance=0.06)
    handshape = _score(np.mean(shape_errs) if shape_errs else 1.0,
                       tolerance=0.35)

    if location < 60:
        vert = "higher" if dy_sum > 0.08 else ("lower" if dy_sum < -0.08 else "")
        horiz = ("closer to your body's midline" if abs(dx_sum) > 0.10 else "")
        detail = " and ".join(x for x in (vert, horiz) if x)
        tips.append(f"Position your hands {detail or 'closer to where the reference holds them'}.")
    if movement < 60:
        tips.append("Focus on the motion path — try matching the reference's "
                    "rhythm and direction.")
    if handshape < 60:
        tips.append("Check your handshape: curl or extend your fingers to "
                    "match the reference more closely.")

    overall = 0.4 * location + 0.3 * movement + 0.3 * handshape
    if overall >= 80 and not tips:
        tips.append("Great job — that's very close to the reference signing!")

    return Feedback(gloss=gloss, overall=round(overall, 1),
                    location=round(location, 1), movement=round(movement, 1),
                    handshape=round(handshape, 1), tips=tips[:3])
