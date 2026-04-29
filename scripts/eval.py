import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from asl.config import Config
from asl.data.dataset import WLASLDataset
from asl.data.wlasl import load_index, split_index, load_class_list
from asl.models.hybrid import build_model
from asl.train.metrics import MetricTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--ckpt", default="best.pt")
    args = parser.parse_args()

    run_dir = Path(args.run)
    cfg = Config.load(run_dir / "config.json")
    device = torch.device(cfg.device)

    splits = split_index(load_index(cfg))
    ds = WLASLDataset(cfg, splits[args.split], train=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    model = build_model(cfg).to(device)
    state = torch.load(run_dir / args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {args.ckpt} (epoch {state.get('epoch')}), "
          f"{args.split} clips: {len(ds)}")

    tracker = MetricTracker(cfg.num_classes)
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["landmarks"].to(device),
                           batch["crops"].to(device),
                           batch["hand_mask"].to(device))
            tracker.update(logits, batch["label"])

    metrics = tracker.compute()
    print({k: round(v, 4) for k, v in metrics.items()})

    cm = tracker.confusion()
    np.save(run_dir / f"confusion_{args.split}.npy", cm)
    classes = load_class_list()
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    seen = cm.sum(axis=1) > 0
    worst = np.argsort(per_class + (~seen))[:10]
    print("worst classes:", [(classes[int(i)], round(float(per_class[i]), 2))
                             for i in worst if seen[i]])


if __name__ == "__main__":
    main()
