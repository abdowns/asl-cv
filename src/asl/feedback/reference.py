from __future__ import annotations

import numpy as np

from asl.config import Config, PROCESSED_DIR, NUM_POINTS
from asl.data.features import normalize_landmarks, sample_indices
from asl.data.wlasl import Instance, load_index, split_index
from asl.feedback.dtw import dtw_align
from asl.preprocess.extract import output_path

REF_LEN = 32
L_WRIST, R_WRIST = 0, 21


def _load_normalized(inst: Instance) -> np.ndarray | None:
    p = output_path(inst.video_id)
    if not p.exists():
        return None
    with np.load(p) as z:
        lm, pm = z["landmarks"], z["pose_mask"]
    sel = sample_indices(len(lm), REF_LEN, train=False)
    return normalize_landmarks(lm[sel], pm[sel])


def _wrist_track(traj: np.ndarray) -> np.ndarray:
    return traj[:, [L_WRIST, R_WRIST], :2].reshape(len(traj), -1)


def references_path(cfg: Config):
    return PROCESSED_DIR / f"references_{cfg.subset}.npz"


def build_references(cfg: Config, log=print) -> np.ndarray:
    path = references_path(cfg)
    if path.exists():
        return np.load(path)["refs"]

    splits = split_index(load_index(cfg))
    by_label: dict[int, list[np.ndarray]] = {}
    for inst in splits["train"]:
        traj = _load_normalized(inst)
        if traj is not None:
            by_label.setdefault(inst.label, []).append(traj)

    refs = np.zeros((cfg.num_classes, REF_LEN, NUM_POINTS, 3), dtype=np.float32)
    for label, trajs in by_label.items():
        if len(trajs) == 1:
            refs[label] = trajs[0]
            continue
        tracks = [_wrist_track(t) for t in trajs]
        costs = np.zeros(len(trajs))
        for i in range(len(trajs)):
            costs[i] = np.mean([dtw_align(tracks[i], tracks[j])[0]
                                for j in range(len(trajs)) if j != i])
        refs[label] = trajs[int(np.argmin(costs))]
        if label % 20 == 0:
            log(f"  reference {label} built from {len(trajs)} clips")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, refs=refs)
    return refs
