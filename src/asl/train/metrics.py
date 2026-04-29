from __future__ import annotations

import numpy as np
import torch


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor,
                  ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(labels.unsqueeze(1))
    return {f"top{k}": correct[:, :k].any(dim=1).float().mean().item() for k in ks}


class MetricTracker:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.logits: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self.logits.append(logits.detach().float().cpu())
        self.labels.append(labels.detach().cpu())

    def compute(self) -> dict[str, float]:
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        return topk_accuracy(logits, labels)

    def confusion(self) -> np.ndarray:
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        pred = logits.argmax(dim=1)
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for t, p in zip(labels.numpy(), pred.numpy()):
            cm[t, p] += 1
        return cm
