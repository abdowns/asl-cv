import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch

from asl.config import Config, NUM_HAND_LANDMARKS
from asl.data.wlasl import load_class_list
from asl.models.hybrid import build_model
from asl.realtime.engine import StreamingRecognizer
from asl.realtime.tracker import LiveTracker

# Standard 21-point MediaPipe hand topology (thumb, index, middle, ring,
# pinky chains plus the palm base) — hardcoded since this mediapipe build
# only ships the Tasks API, not the legacy `solutions` drawing helpers.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
HAND_COLORS = [(0, 220, 255), (255, 180, 0)]  # left, right


def draw_hand_overlay(frame_bgr: np.ndarray, landmarks: np.ndarray,
                       hand_mask: np.ndarray) -> None:
    h, w = frame_bgr.shape[:2]
    for slot in range(2):
        if not hand_mask[slot]:
            continue
        off = slot * NUM_HAND_LANDMARKS
        pts = landmarks[off:off + NUM_HAND_LANDMARKS]
        px = [(int(x * w), int(y * h)) for x, y, _ in pts]
        color = HAND_COLORS[slot]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, px[a], px[b], color, 2, cv2.LINE_AA)
        for x, y in px:
            cv2.circle(frame_bgr, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (x, y), 4, color, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--video", default=None)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run)
    cfg = Config.load(run_dir / "config.json")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = build_model(cfg).to(cfg.device)
    state = torch.load(run_dir / "best.pt", map_location=cfg.device)
    model.load_state_dict(state["model"])

    tracker = LiveTracker(cfg)
    recognizer = StreamingRecognizer(cfg, model, load_class_list())

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        sys.exit("cannot open capture source")

    caption, caption_conf = "", 0.0
    top5: list[tuple[str, float]] = []
    emitted_words: list[str] = []
    t0 = time.time()
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_bgr = cv2.flip(frame_bgr, 1) if not args.video else frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts_ms = int((time.time() - t0) * 1000) if not args.video \
            else int(cap.get(cv2.CAP_PROP_POS_MSEC))

        landmarks, hand_mask, pose_ok, crops = tracker.process(rgb, ts_ms)
        pred = recognizer.add_frame(landmarks, hand_mask, pose_ok, crops)
        if pred:
            caption, caption_conf = pred.gloss, pred.confidence
            top5 = pred.top5
            if pred.emitted:
                emitted_words = (emitted_words + [pred.gloss])[-8:]

        draw_hand_overlay(frame_bgr, landmarks, hand_mask)

        # Drawn every frame from the last known distribution so it stays on
        # screen between inference steps instead of flashing.
        bar_y = 24
        for gloss, p in top5:
            cv2.rectangle(frame_bgr, (10, bar_y), (10 + int(p * 200), bar_y + 14),
                          (80, 200, 80), -1)
            cv2.putText(frame_bgr, f"{gloss} {p:.2f}", (220, bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            bar_y += 22

        cv2.putText(frame_bgr, " ".join(emitted_words), (10, frame_bgr.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame_bgr, f"{caption} ({caption_conf:.2f})",
                    (10, frame_bgr.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("ASL demo", frame_bgr)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if args.video:
        print("emitted:", emitted_words)
        if recognizer.probs is not None:
            order = np.argsort(recognizer.probs)[::-1][:5]
            classes = load_class_list()
            print("final top5:", [(classes[int(i)], round(float(recognizer.probs[i]), 3))
                                  for i in order])
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
