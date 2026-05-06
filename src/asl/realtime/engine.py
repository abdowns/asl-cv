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


class StreamingRecognizer:
    def __init__(self, cfg: Config, model: torch.nn.Module,
                 classes: dict[int, str], window: int = 64, stride: int = 8):
        self.cfg = cfg
        self.model = model.eval()
        self.device = torch.device(cfg.device)
        self.classes = classes
        self.window = window
        self.stride = stride
        self.buffer: deque[tuple[np.ndarray, np.ndarray, bool, np.ndarray]] = \
            deque(maxlen=window)
        self._count = 0

    def reset(self) -> None:
        self.buffer.clear()
        self._count = 0

    def add_frame(self, landmarks: np.ndarray, hand_mask: np.ndarray,
                  pose_ok: bool, crops: np.ndarray) -> Prediction | None:
        self.buffer.append((landmarks, hand_mask, pose_ok, crops))
        self._count += 1
        if len(self.buffer) < self.cfg.num_frames or self._count % self.stride:
            return None
        return self._infer()

    @torch.no_grad()
    def _infer(self) -> Prediction:
        lm = np.stack([f[0] for f in self.buffer])
        hm = np.stack([f[1] for f in self.buffer])
        pm = np.array([f[2] for f in self.buffer])
        cr = np.stack([f[3] for f in self.buffer])

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
        order = np.argsort(probs)[::-1]
        top = int(order[0])
        return Prediction(
            gloss=self.classes[top],
            label=top,
            confidence=float(probs[top]),
            top5=[(self.classes[int(i)], float(probs[i])) for i in order[:5]],
        )
