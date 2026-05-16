from __future__ import annotations

import numpy as np

from asl.config import Config, NUM_POINTS
from asl.data.features import normalize_landmarks, sample_indices
from asl.data.wlasl import load_index, split_index
from asl.feedback.dtw import dtw_align
from asl.preprocess.extract import output_path


def build_references(cfg: Config, length: int = 32) -> np.ndarray:
    splits = split_index(load_index(cfg))
    by_label: dict[int, list[np.ndarray]] = {}
    for inst in splits["train"]:
        p = output_path(inst.video_id)
        if not p.exists():
            continue
        with np.load(p) as z:
            lm, pm = z["landmarks"], z["pose_mask"]
        sel = sample_indices(len(lm), length, train=False)
        by_label.setdefault(inst.label, []).append(normalize_landmarks(lm[sel], pm[sel]))

    refs = np.zeros((cfg.num_classes, length, NUM_POINTS, 3), dtype=np.float32)
    for label, trajs in by_label.items():
        flat = [t[:, :, :2].reshape(len(t), -1) for t in trajs]
        costs = np.zeros(len(trajs))
        for i in range(len(trajs)):
            for j in range(len(trajs)):
                if i != j:
                    costs[i] += dtw_align(flat[i], flat[j])[0]
        refs[label] = trajs[int(np.argmin(costs))]
    return refs
