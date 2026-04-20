import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asl.config import Config
from asl.data.wlasl import load_index
from asl.preprocess.extract import extract_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=100)
    args = parser.parse_args()

    cfg = Config(subset=args.subset)
    instances = load_index(cfg)
    done, failed = extract_all(cfg, instances)
    print(f"extracted {done}, failed {failed}")


if __name__ == "__main__":
    main()
