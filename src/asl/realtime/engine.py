from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from asl.config import Config
from asl.data.features import build_features, sample_indices
from asl.data.dataset import CROP_MEAN, CROP_STD


@dataclass
class Prediction:
    gloss: str
    label: int
    confidence: float
    top5: list[tuple[str, float]]
    emitted: bool


class StreamingRecognizer:
    def __init__(self, cfg: Config, model: torch.nn.Module,
                 classes: dict[int, str], window: int = 64, stride: int = 6,
                 ema: float = 0.6, emit_threshold: float = 0.55,
                 idle_reset_frames: int = 30):
        self.cfg = cfg
        self.model = model.eval()
        self.device = torch.device(cfg.device)
        self.classes = classes
        self.window = window
        self.stride = stride
        self.ema = ema
        self.emit_threshold = emit_threshold
        self.idle_reset_frames = idle_reset_frames

        self.landmarks: deque[np.ndarray] = deque(maxlen=window)
        self.hand_masks: deque[np.ndarray] = deque(maxlen=window)
        self.pose_oks: deque[bool] = deque(maxlen=window)
        self.crops: deque[np.ndarray] = deque(maxlen=window)
        self.probs: np.ndarray | None = None
        self.last_emitted: int | None = None
        self._since_infer = 0
        self._idle = 0

    def reset(self) -> None:
        self.landmarks.clear()
        self.hand_masks.clear()
        self.pose_oks.clear()
        self.crops.clear()
        self.probs = None
        self.last_emitted = None
        self._since_infer = 0
        self._idle = 0

    def add_frame(self, landmarks: np.ndarray, hand_mask: np.ndarray,
                  pose_ok: bool, crops: np.ndarray) -> Prediction | None:
        if not hand_mask.any():
            self._idle += 1
            if self._idle >= self.idle_reset_frames:
                self.probs = None
                self.last_emitted = None
        else:
            self._idle = 0

        self.landmarks.append(landmarks)
        self.hand_masks.append(hand_mask)
        self.pose_oks.append(pose_ok)
        self.crops.append(crops)

        self._since_infer += 1
        if (len(self.landmarks) < self.cfg.num_frames // 2
                or self._since_infer < self.stride
                or not any(m.any() for m in self.hand_masks)):
            return None
        self._since_infer = 0
        return self._infer()

    @torch.no_grad()
    def _infer(self) -> Prediction:
        lm = np.stack(self.landmarks)
        hm = np.stack(self.hand_masks)
        pm = np.array(self.pose_oks)
        cr = np.stack(self.crops)

        sel = sample_indices(len(lm), self.cfg.num_frames, train=False)
        feats = build_features(lm[sel], pm[sel])

        x_lm = torch.from_numpy(feats)[None].to(self.device)
        x_hm = torch.from_numpy(hm[sel].astype(np.float32))[None].to(self.device)
        if self.cfg.visual_stream:
            c = cr[sel].astype(np.float32) / 255.0
            c = (c - CROP_MEAN) / CROP_STD
            x_cr = torch.from_numpy(c).permute(0, 1, 4, 2, 3)[None].to(self.device)
        else:
            S = self.cfg.crop_size
            x_cr = torch.zeros(1, self.cfg.num_frames, 2, 3, S, S,
                               device=self.device)

        probs = self.model(x_lm, x_cr, x_hm).softmax(dim=1)[0].cpu().numpy()
        self.probs = probs if self.probs is None else \
            self.ema * self.probs + (1 - self.ema) * probs

        order = np.argsort(self.probs)[::-1]
        top = int(order[0])
        conf = float(self.probs[top])
        emitted = conf >= self.emit_threshold and top != self.last_emitted
        if emitted:
            self.last_emitted = top
        return Prediction(
            gloss=self.classes[top],
            label=top,
            confidence=conf,
            top5=[(self.classes[int(i)], float(self.probs[i])) for i in order[:5]],
            emitted=emitted,
        )
