from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
NSLT_PATH = PROJECT_ROOT / "dataset" / "nslt_100.json"


@dataclass
class Config:
    subset: int = 100
    num_frames: int = 32
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-3
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
