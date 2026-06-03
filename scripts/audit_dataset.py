import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asl.config import Config, VIDEOS_DIR
from asl.data.wlasl import load_index, split_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=100, choices=[100, 300, 1000, 2000])
    args = parser.parse_args()

    cfg = Config(subset=args.subset)
    nslt = json.loads(cfg.nslt_path.read_text())
    instances = load_index(cfg)
    splits = split_index(instances)

    print(f"WLASL{args.subset}: {len(nslt)} instances in official list, "
          f"{len(instances)} with video present "
          f"({len(nslt) - len(instances)} missing)")
    for name, insts in splits.items():
        print(f"  {name:5s}: {len(insts):5d} clips, {len({i.label for i in insts}):4d} classes")

    train_counts = Counter(i.gloss for i in splits["train"])
    all_glosses = {i.gloss for i in instances}
    empty_train = all_glosses - set(train_counts)
    if empty_train:
        print(f"  WARNING: {len(empty_train)} classes have NO train clips: {sorted(empty_train)}")

    counts = sorted(train_counts.items(), key=lambda kv: kv[1])
    print(f"  train clips/class: min={counts[0][1]} ({counts[0][0]}), "
          f"max={counts[-1][1]} ({counts[-1][0]}), "
          f"mean={sum(train_counts.values()) / len(train_counts):.1f}")
    thin = [f"{g}={c}" for g, c in counts if c < 5]
    if thin:
        print(f"  classes with <5 train clips: {', '.join(thin)}")


if __name__ == "__main__":
    main()
