# asl-cv

ASL recognition from live video. MediaPipe landmarks feed
SignFusionNet: a from-scratch ViT (hand crops) + temporal CNN + BiGRU.
Runs as a local desktop webcam demo.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
```

Dataset: WLASL (Word-Level American Sign Language), the 100 class subset
(WLASL100) was used during development. Place it under `dataset/`
(WLASL_v0.3.json, nslt_*.json, wlasl_class_list.txt, videos/*.mp4).

## Workflow

```bash
.venv/bin/python scripts/audit_dataset.py --subset 100 # check video coverage per class
.venv/bin/python scripts/preprocess.py --subset 100 # extract landmarks and hand crops
.venv/bin/python -m pytest tests/ # run unit tests
.venv/bin/python scripts/train.py --subset 100 --visual-stream off # train landmark only, fast
.venv/bin/python scripts/train.py --subset 100 --visual-stream on # train full hybrid, gpu recommended
.venv/bin/python scripts/eval.py --run runs/<name> # evaluate a trained run
.venv/bin/python scripts/demo_webcam.py --run runs/<name> # live webcam demo
```