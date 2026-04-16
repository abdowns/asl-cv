from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
VIDEOS_DIR = DATASET_DIR / "videos"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class Config:
    subset: int = 100
    num_frames: int = 32
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-3
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def num_classes(self) -> int:
        return self.subset

    @property
    def nslt_path(self) -> Path:
        return DATASET_DIR / f"nslt_{self.subset}.json"
