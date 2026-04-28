import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from asl.config import Config, RUNS_DIR
from asl.data.dataset import WLASLDataset
from asl.data.wlasl import load_index, split_index
from asl.models.hybrid import build_model
from asl.train.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=100, choices=[100, 300, 1000, 2000])
    parser.add_argument("--visual-stream", choices=["on", "off"], default="on")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    cfg = Config(subset=args.subset, visual_stream=args.visual_stream == "on")
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size

    torch.manual_seed(cfg.seed)

    splits = split_index(load_index(cfg))
    train_ds = WLASLDataset(cfg, splits["train"], train=True)
    val_ds = WLASLDataset(cfg, splits["val"], train=False)
    print(f"device={cfg.device} train={len(train_ds)} val={len(val_ds)} "
          f"visual={cfg.visual_stream}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True,
                              persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers,
                            persistent_workers=cfg.num_workers > 0)

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params: {n_params:.2f}M")

    name = args.name or (f"wlasl{cfg.subset}_"
                         f"{'hybrid' if cfg.visual_stream else 'landmark'}_"
                         f"{time.strftime('%m%d_%H%M')}")
    trainer = Trainer(cfg, model, train_loader, val_loader, RUNS_DIR / name)
    result = trainer.fit()
    print(f"done: {result} -> {RUNS_DIR / name}")


if __name__ == "__main__":
    main()
