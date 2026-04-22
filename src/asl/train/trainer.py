from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from asl.config import Config
from asl.train.metrics import MetricTracker


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

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def _run_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        lm = batch["landmarks"].to(self.device)
        crops = batch["crops"].to(self.device)
        mask = batch["hand_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        logits = self.model(lm, crops, mask)
        return logits, labels

    def train_epoch(self) -> float:
        self.model.train()
        total, count = 0.0, 0
        for batch in self.train_loader:
            logits, labels = self._run_batch(batch)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item() * len(labels)
            count += len(labels)
        return total / max(count, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        tracker = MetricTracker(self.cfg.num_classes)
        for batch in loader:
            logits, labels = self._run_batch(batch)
            tracker.update(logits, labels)
        return tracker.compute()

    def fit(self) -> dict:
        val_metrics: dict[str, float] = {}
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            torch.save(self.model.state_dict(), self.run_dir / "model.pt")
            self.log(f"epoch {epoch} loss {train_loss:.3f} "
                     f"val top1 {val_metrics['top1']:.3f} "
                     f"top5 {val_metrics['top5']:.3f} "
                     f"({time.time() - t0:.1f}s)")
        return val_metrics
