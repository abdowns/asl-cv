from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from asl.config import (
    Config,
    MODELS_DIR,
    PROCESSED_DIR,
    NUM_HAND_LANDMARKS,
    POSE_KEEP,
    NUM_POINTS,
)
from asl.data.wlasl import Instance

FEATURES_DIR = PROCESSED_DIR / "features"


def read_frames(inst: Instance) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(inst.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise IOError(f"no frames decoded from {inst.video_path}")
    return np.stack(frames), float(fps)


def _crop_hand(frame: np.ndarray, pts_xy: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0 = pts_xy.min(axis=0)
    x1, y1 = pts_xy.max(axis=0)
    side = max(x1 - x0, y1 - y0) * 1.5
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    x1, y1 = min(w, int(cx + side / 2)), min(h, int(cy + side / 2))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


class Extractor:
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

    def extract(self, inst: Instance) -> dict:
        import mediapipe as mp

        frames, fps = read_frames(inst)
        T = len(frames)
        S = self.cfg.crop_size
        landmarks = np.zeros((T, NUM_POINTS, 3), dtype=np.float32)
        hand_mask = np.zeros((T, 2), dtype=bool)
        pose_mask = np.zeros(T, dtype=bool)
        crops = np.zeros((T, 2, S, S, 3), dtype=np.uint8)

        for t, frame in enumerate(frames):
            ts_ms = int(t * 1000 / fps) + t
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            hand_res = self.hand.detect_for_video(mp_img, ts_ms)
            pose_res = self.pose.detect_for_video(mp_img, ts_ms)
            h, w = frame.shape[:2]

            for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
                slot = 0 if handed[0].category_name == "Left" else 1
                pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
                off = slot * NUM_HAND_LANDMARKS
                landmarks[t, off:off + NUM_HAND_LANDMARKS] = pts
                hand_mask[t, slot] = True
                pts_px = pts[:, :2] * [w, h]
                crops[t, slot] = _crop_hand(frame, pts_px, S)

            if pose_res.pose_landmarks:
                pts = np.array(
                    [[p.x, p.y, p.z] for p in pose_res.pose_landmarks[0]],
                    dtype=np.float32,
                )[POSE_KEEP]
                landmarks[t, 2 * NUM_HAND_LANDMARKS:] = pts
                pose_mask[t] = True

        return {
            "landmarks": landmarks,
            "hand_mask": hand_mask,
            "pose_mask": pose_mask,
            "crops": crops,
            "meta": json.dumps({
                "video_id": inst.video_id, "gloss": inst.gloss,
                "label": inst.label, "split": inst.split,
                "fps": fps,
            }),
        }

    def close(self) -> None:
        self.hand.close()
        self.pose.close()


def output_path(video_id: str) -> Path:
    return FEATURES_DIR / f"{video_id}.npz"


def extract_all(cfg: Config, instances: list[Instance],
                log=print) -> tuple[int, list[str]]:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    todo = [i for i in instances if not output_path(i.video_id).exists()]
    log(f"{len(instances)} instances, {len(instances) - len(todo)} cached, "
        f"{len(todo)} to extract")
    failed: list[str] = []
    for n, inst in enumerate(todo, 1):
        ex = Extractor(cfg)
        try:
            data = ex.extract(inst)
        except Exception as e:
            failed.append(inst.video_id)
            log(f"  FAIL {inst.video_id} ({inst.gloss}): {e}")
            continue
        finally:
            ex.close()
        tmp = output_path(inst.video_id).with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.rename(output_path(inst.video_id))
        if n % 25 == 0 or n == len(todo):
            log(f"  [{n}/{len(todo)}] {inst.video_id} ({inst.gloss})")
    return len(todo) - len(failed), failed
