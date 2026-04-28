from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from asl.config import Config
from asl.train.metrics import MetricTracker


def make_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    warmup = cfg.warmup_epochs * steps_per_epoch
    total = cfg.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, cfg: Config, model: nn.Module,
                 train_loader: DataLoader, val_loader: DataLoader,
                 run_dir: Path, log=print):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = run_dir
        self.log = log
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(run_dir / "config.json")

        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = make_scheduler(self.optimizer, cfg, len(train_loader))
        self.best_top1 = 0.0
        self.history: list[dict] = []

    def _run_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        lm = batch["landmarks"].to(self.device, non_blocking=True)
        crops = batch["crops"].to(self.device, non_blocking=True)
        mask = batch["hand_mask"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)
        logits = self.model(lm, crops, mask)
        return logits, labels

    def train_epoch(self) -> float:
        self.model.train()
        total, count = 0.0, 0
        for batch in self.train_loader:
            logits, labels = self._run_batch(batch)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            total += loss.item() * len(labels)
            count += len(labels)
        return total / max(count, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[dict[str, float], MetricTracker]:
        self.model.eval()
        tracker = MetricTracker(self.cfg.num_classes)
        for batch in loader:
            logits, labels = self._run_batch(batch)
            tracker.update(logits, labels)
        return tracker.compute(), tracker

    def fit(self) -> dict:
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_metrics, _ = self.evaluate(self.val_loader)
            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_top1": round(val_metrics["top1"], 4),
                "val_top5": round(val_metrics["top5"], 4),
                "lr": self.scheduler.get_last_lr()[0],
                "sec": round(time.time() - t0, 1),
            }
            self.history.append(row)
            (self.run_dir / "history.json").write_text(json.dumps(self.history))

            if val_metrics["top1"] >= self.best_top1:
                self.best_top1 = val_metrics["top1"]
                torch.save(
                    {"model": self.model.state_dict(), "epoch": epoch,
                     "val_top1": self.best_top1},
                    self.run_dir / "best.pt",
                )
            torch.save({"model": self.model.state_dict(), "epoch": epoch},
                       self.run_dir / "last.pt")
            self.log(f"epoch {epoch:3d}/{self.cfg.epochs} "
                     f"loss {train_loss:.3f} "
                     f"val top1 {val_metrics['top1']:.3f} "
                     f"top5 {val_metrics['top5']:.3f} "
                     f"({row['sec']}s, best {self.best_top1:.3f})")
        return {"best_val_top1": self.best_top1}
