from __future__ import annotations

import numpy as np


def dtw_align(a: np.ndarray, b: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    n, m = len(a), len(b)
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = dist[i - 1, j - 1] + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(acc[n, m] / len(path)), path
