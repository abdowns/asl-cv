import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asl.config import Config, PROCESSED_DIR
from asl.data.wlasl import load_index
from asl.preprocess.extract import extract_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=100, choices=[100, 300, 1000, 2000])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = Config(subset=args.subset)
    instances = load_index(cfg)
    if args.limit:
        instances = instances[: args.limit]

    done, failed = extract_all(cfg, instances)
    print(f"extracted {done}, failed {len(failed)}")
    if failed:
        fail_log = PROCESSED_DIR / f"failed_{args.subset}.txt"
        fail_log.write_text("\n".join(failed))
        print(f"failed ids written to {fail_log}")


if __name__ == "__main__":
    main()
