import argparse
import json
import sys
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

    missing = len(nslt) - len(instances)
    print(f"WLASL{args.subset}: {len(nslt)} instances, {len(instances)} with video, {missing} missing")
    for name, insts in splits.items():
        print(f"  {name}: {len(insts)} clips, {len({i.label for i in insts})} classes")

    per_class = {}
    for inst in splits["train"]:
        per_class[inst.gloss] = per_class.get(inst.gloss, 0) + 1
    print(f"  train clips/class: min={min(per_class.values())}, max={max(per_class.values())}")


if __name__ == "__main__":
    main()
