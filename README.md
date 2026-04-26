# asl-cv

Word-level ASL recognition with MediaPipe landmarks and a hybrid
ViT + temporal CNN + BiGRU model.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
```

Dataset: WLASL. Place it under `dataset/`
(WLASL_v0.3.json, nslt_*.json, wlasl_class_list.txt, videos/*.mp4).

## Training

```bash
.venv/bin/python scripts/preprocess.py --subset 100
.venv/bin/python scripts/train.py --subset 100 --visual-stream off
.venv/bin/python scripts/train.py --subset 100 --visual-stream on
```
