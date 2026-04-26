import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from asl.config import Config, RUNS_DIR
from asl.data.dataset import WLASLDataset
from asl.data.wlasl import load_index, split_index
from asl.models.hybrid import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=100, choices=[100, 300, 1000, 2000])
    parser.add_argument("--visual-stream", choices=["on", "off"], default="on")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = Config(subset=args.subset, visual_stream=args.visual_stream == "on")
    if args.epochs:
        cfg.epochs = args.epochs

    torch.manual_seed(cfg.seed)

    splits = split_index(load_index(cfg))
    train_ds = WLASLDataset(cfg, splits["train"], train=True)
    val_ds = WLASLDataset(cfg, splits["val"], train=False)
    print(f"train={len(train_ds)} val={len(val_ds)} visual={cfg.visual_stream}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    model = build_model(cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            loss = loss_fn(model(batch), batch["label"])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(cfg.device) for k, v in batch.items()}
                pred = model(batch).argmax(dim=-1)
                correct += (pred == batch["label"]).sum().item()
                total += batch["label"].numel()
        print(f"epoch {epoch + 1}: loss={loss.item():.4f} val_acc={correct / max(total, 1):.4f}")

    out_dir = RUNS_DIR / f"wlasl{cfg.subset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")


if __name__ == "__main__":
    main()
