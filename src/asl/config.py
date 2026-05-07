from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
VIDEOS_DIR = DATASET_DIR / "videos"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models" / "mediapipe"

NUM_HAND_LANDMARKS = 21
# indices into mediapipe pose's 33 landmarks: nose, shoulders, elbows, wrists, hips
POSE_KEEP = [0, 11, 12, 13, 14, 15, 16, 23, 24]
NUM_POSE_LANDMARKS = len(POSE_KEEP)
NUM_POINTS = 2 * NUM_HAND_LANDMARKS + NUM_POSE_LANDMARKS


def autodetect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    subset: int = 100
    num_frames: int = 32
    crop_size: int = 96

    visual_stream: bool = True
    vit_patch: int = 16
    vit_dim: int = 192
    vit_depth: int = 4
    vit_heads: int = 4
    landmark_dim: int = 256
    fusion_dim: int = 256
    tcn_channels: int = 256
    tcn_blocks: int = 4
    gru_hidden: int = 256
    gru_layers: int = 2
    dropout: float = 0.3

    batch_size: int = 32
    epochs: int = 120
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 4
    seed: int = 42

    device: str = field(default_factory=autodetect_device)

    @property
    def num_classes(self) -> int:
        return self.subset

    @property
    def nslt_path(self) -> Path:
        return DATASET_DIR / f"nslt_{self.subset}.json"

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls(**json.loads(path.read_text()))
