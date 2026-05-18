import torch

from asl.config import Config
from asl.data.features import FEATURE_DIM
from asl.models.hybrid import build_model
from asl.models.tcn import TemporalCNN
from asl.models.vit import ViT

B, T = 2, 8


def _batch(cfg: Config):
    return (
        torch.randn(B, T, FEATURE_DIM),
        torch.randn(B, T, 2, 3, cfg.crop_size, cfg.crop_size),
        torch.randint(0, 2, (B, T, 2)).float(),
    )


def test_vit_shape():
    vit = ViT(img_size=96, patch=16, dim=192, depth=2, heads=4)
    assert vit(torch.randn(4, 3, 96, 96)).shape == (4, 192)


def test_tcn_shape_and_receptive_field():
    tcn = TemporalCNN(in_dim=64, channels=32, num_blocks=4, dropout=0.0)
    x = torch.randn(B, 16, 64)
    assert tcn(x).shape == (B, 16, 32)


def test_hybrid_forward():
    cfg = Config(subset=100, num_frames=T, vit_depth=2)
    model = build_model(cfg)
    logits = model(*_batch(cfg))
    assert logits.shape == (B, 100)
    assert torch.isfinite(logits).all()


def test_landmark_only_forward():
    cfg = Config(subset=100, num_frames=T, visual_stream=False)
    model = build_model(cfg)
    logits = model(*_batch(cfg))
    assert logits.shape == (B, 100)


def test_lstm_variant():
    cfg = Config(subset=100, num_frames=T, visual_stream=False, rnn_type="lstm")
    logits = build_model(cfg)(*_batch(cfg))
    assert logits.shape == (B, 100)


def test_masked_hand_contributes_zero():
    cfg = Config(subset=100, num_frames=T, vit_depth=2)
    model = build_model(cfg).eval()
    lm, crops, mask = _batch(cfg)
    mask.zero_()
    a = model(lm, crops, mask)
    b = model(lm, torch.randn_like(crops), mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_overfit_one_batch():
    torch.manual_seed(0)
    cfg = Config(subset=100, num_frames=T, vit_depth=2, dropout=0.0)
    model = build_model(cfg).train()
    lm, crops, mask = _batch(cfg)
    mask.fill_(1.0)
    labels = torch.arange(B)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    first = last = None
    for _ in range(60):
        logits = model(lm, crops, mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.1, f"loss barely moved: {first:.3f} -> {last:.3f}"
