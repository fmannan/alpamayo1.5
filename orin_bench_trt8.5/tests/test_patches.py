"""The export-time patches must not change the model's math (CPU, float64, tiny random models)."""

import copy

import pytest
import torch
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm, Qwen3VLVisionModel

from orinbench.patches import (
    ScaledLayerNorm,
    ScaledRMSNorm,
    enable_batched_vision_attention,
    enable_scaled_layernorm,
)


def tiny_vision_config(impl: str):
    cfg = Qwen3VLConfig().vision_config
    cfg.depth, cfg.deepstack_visual_indexes, cfg.hidden_size = 2, [0, 1, 1], 128
    cfg.intermediate_size, cfg.out_hidden_size, cfg.num_heads = 256, 128, 4
    cfg._attn_implementation = impl
    return cfg


@pytest.mark.parametrize("scale", [1.0, 600.0, 13000.0])  # vision activations reach ~13,000 in the last blocks
def test_scaled_layernorm_matches_layernorm(scale):
    torch.manual_seed(0)
    norm = torch.nn.LayerNorm(64, eps=1e-6).double()
    torch.nn.init.normal_(norm.weight)
    torch.nn.init.normal_(norm.bias)
    x = torch.randn(8, 64, dtype=torch.float64) * scale + 3.0
    torch.testing.assert_close(ScaledLayerNorm(norm)(x), norm(x), rtol=1e-4, atol=1e-4)


def test_scaled_layernorm_is_finite_in_fp16_where_plain_squares_overflow():
    x = torch.randn(4, 1152, dtype=torch.float32) * 13000.0
    d = (x - x.mean(-1, keepdim=True)).half()
    assert not torch.isfinite(d * d).all()  # the decomposed LayerNorm overflows here
    norm = torch.nn.LayerNorm(1152).half()
    assert torch.isfinite(ScaledLayerNorm(norm)(x.half())).all()


def test_scaled_rmsnorm_matches_qwen_rmsnorm():
    torch.manual_seed(0)
    norm = Qwen3VLTextRMSNorm(64, eps=1e-6).double()
    torch.nn.init.normal_(norm.weight)
    x = torch.randn(8, 64, dtype=torch.float64) * 500.0
    torch.testing.assert_close(ScaledRMSNorm(norm)(x), norm(x), rtol=1e-4, atol=1e-4)


def test_enable_scaled_layernorm_replaces_every_layernorm():
    model = Qwen3VLVisionModel(tiny_vision_config("sdpa"))
    n = sum(isinstance(m, torch.nn.LayerNorm) for m in model.modules())
    assert enable_scaled_layernorm(model) == n > 0
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in model.modules())


@pytest.mark.parametrize("impl", ["sdpa", "eager"])
@pytest.mark.parametrize("grid", [[[1, 4, 8]] * 3, [[1, 4, 8], [1, 4, 4], [1, 8, 8]]], ids=["equal", "unequal"])
def test_batched_vision_attention_matches_per_image_loop(impl, grid):
    """Equal grids: the batched path must match the deployed SDPA loop. Unequal grids: the patch falls back to the
    original loop, so it must match the unpatched model of the same implementation (transformers' eager loop
    computes its softmax in float32, so it differs slightly from SDPA in float64)."""
    torch.manual_seed(0)
    deployed = Qwen3VLVisionModel(tiny_vision_config("sdpa")).double().eval()
    patched = Qwen3VLVisionModel(copy.deepcopy(tiny_vision_config(impl))).double().eval()
    patched.load_state_dict(deployed.state_dict())
    if len({tuple(r) for r in grid}) > 1:
        deployed = Qwen3VLVisionModel(copy.deepcopy(tiny_vision_config(impl))).double().eval()
        deployed.load_state_dict(patched.state_dict())
    enable_batched_vision_attention(patched)
    g = torch.tensor(grid)
    x = torch.randn(int(g.prod(1).sum()), 3 * 2 * 16 * 16, dtype=torch.float64)
    with torch.no_grad():
        a, b = deployed(x, grid_thw=g), patched(x, grid_thw=g)
    for p, q in zip([a[0], *a[1]], [b[0], *b[1]]):
        torch.testing.assert_close(q, p, rtol=0, atol=1e-10)
