import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import torch

from asl.config import Config
from asl.data.wlasl import load_class_list
from asl.models.hybrid import build_model
from asl.realtime.engine import StreamingRecognizer
from asl.realtime.tracker import LiveTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run)
    cfg = Config.load(run_dir / "config.json")
    device = torch.device(cfg.device)

    model = build_model(cfg).to(device)
    state = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model"])

    tracker = LiveTracker(cfg)
    recognizer = StreamingRecognizer(cfg, model, load_class_list())

    cap = cv2.VideoCapture(args.camera)
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
        frame_bgr = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts_ms = int((time.time() - t0) * 1000)

        landmarks, hand_mask, pose_ok, crops = tracker.process(rgb, ts_ms)
        pred = recognizer.add_frame(landmarks, hand_mask, pose_ok, crops)
        if pred:
            caption, caption_conf = pred.gloss, pred.confidence
            top5 = pred.top5
            if pred.emitted:
                emitted_words = (emitted_words + [pred.gloss])[-8:]

        h, w = frame_bgr.shape[:2]
        for slot in range(2):
            if not hand_mask[slot]:
                continue
            for x, y, _ in landmarks[slot * 21:(slot + 1) * 21]:
                cv2.circle(frame_bgr, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)

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

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
