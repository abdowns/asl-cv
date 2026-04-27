import torch

from asl.config import Config
from asl.data.features import FEATURE_DIM
from asl.models.hybrid import build_model
from asl.models.tcn import TemporalCNN
from asl.models.vit import ViT


def _inputs(cfg: Config, b: int = 2):
    t = cfg.num_frames
    lm = torch.randn(b, t, FEATURE_DIM)
    crops = torch.randn(b, t, 2, 3, cfg.crop_size, cfg.crop_size)
    mask = torch.ones(b, t, 2)
    return lm, crops, mask


def test_vit_shape():
    vit = ViT(img_size=96, patch=16, dim=192, depth=2, heads=4)
    out = vit(torch.randn(4, 3, 96, 96))
    assert out.shape == (4, 192)


def test_tcn_shape():
    tcn = TemporalCNN(in_dim=64, channels=32, num_blocks=4, dropout=0.0)
    assert tcn(torch.randn(2, 16, 64)).shape == (2, 16, 32)


def test_hybrid_forward():
    cfg = Config(subset=100, num_frames=8, vit_depth=2)
    model = build_model(cfg)
    logits = model(*_inputs(cfg))
    assert logits.shape == (2, 100)
    assert torch.isfinite(logits).all()


def test_landmark_only_forward():
    cfg = Config(subset=100, num_frames=8, visual_stream=False)
    model = build_model(cfg)
    logits = model(*_inputs(cfg))
    assert logits.shape == (2, 100)


def test_masked_hand_contributes_zero():
    cfg = Config(subset=100, num_frames=8, vit_depth=2)
    model = build_model(cfg).eval()
    lm, crops, mask = _inputs(cfg)
    mask = torch.zeros_like(mask)
    a = model(lm, crops, mask)
    b = model(lm, torch.randn_like(crops), mask)
    assert torch.allclose(a, b, atol=1e-5)
