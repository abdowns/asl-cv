from __future__ import annotations

import numpy as np

from asl.config import (
    Config, MODELS_DIR, NUM_HAND_LANDMARKS, POSE_KEEP, NUM_POINTS,
)
from asl.preprocess.extract import _crop_hand


class LiveTracker:
    def __init__(self, cfg: Config):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.cfg = cfg
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODELS_DIR / "hand_landmarker.task")
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODELS_DIR / "pose_landmarker_full.task")
                ),
                running_mode=vision.RunningMode.VIDEO,
            )
        )

    def process(self, frame_rgb: np.ndarray, ts_ms: float):
        import mediapipe as mp

        S = self.cfg.crop_size
        landmarks = np.zeros((NUM_POINTS, 3), dtype=np.float32)
        hand_mask = np.zeros(2, dtype=bool)
        crops = np.zeros((2, S, S, 3), dtype=np.uint8)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        hand_res = self.hand.detect_for_video(mp_img, int(ts_ms))
        pose_res = self.pose.detect_for_video(mp_img, int(ts_ms))
        h, w = frame_rgb.shape[:2]

        for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
            slot = 0 if handed[0].category_name == "Left" else 1
            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            off = slot * NUM_HAND_LANDMARKS
            landmarks[off:off + NUM_HAND_LANDMARKS] = pts
            hand_mask[slot] = True
            crops[slot] = _crop_hand(frame_rgb, pts[:, :2] * [w, h], S)

        pose_ok = bool(pose_res.pose_landmarks)
        if pose_ok:
            pts = np.array(
                [[p.x, p.y, p.z] for p in pose_res.pose_landmarks[0]],
                dtype=np.float32,
            )[POSE_KEEP]
            landmarks[2 * NUM_HAND_LANDMARKS:] = pts

        return landmarks, hand_mask, pose_ok, crops

    def close(self) -> None:
        self.hand.close()
        self.pose.close()
