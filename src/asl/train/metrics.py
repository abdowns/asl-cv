from __future__ import annotations

import torch


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor,
                  ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(labels.unsqueeze(1))
    return {f"top{k}": correct[:, :k].any(dim=1).float().mean().item() for k in ks}


class MetricTracker:
    def __init__(self):
        self.logits: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())

    def compute(self) -> dict[str, float]:
        return topk_accuracy(torch.cat(self.logits), torch.cat(self.labels))
