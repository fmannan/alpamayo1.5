"""Platform workarounds. Each one is recorded in run metadata; none changes the model's math."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PatchConvAsLinear(torch.nn.Module):
    """Same computation as the Qwen3-VL patch Conv3d, using the same parameters.

    A PyTorch built without cuDNN (none exists for CUDA 12.2 on JetPack 5.1, see scripts/setup_venv.sh) falls back to
    slow_conv3d, which effectively hangs on thousands of patches. With kernel == stride
    == input size, the convolution is one inner product per output channel, i.e. a linear layer on
    the flattened patch.
    """

    def __init__(self, conv: torch.nn.Conv3d):
        super().__init__()
        if conv.kernel_size != conv.stride:
            raise ValueError(f"expected kernel == stride, got {conv.kernel_size} {conv.stride}")
        self.kernel_size = conv.kernel_size
        self.weight = conv.weight
        self.bias = conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[2:]) != tuple(self.kernel_size):
            raise ValueError(f"expected patches of size {self.kernel_size}, got {tuple(x.shape)}")
        y = F.linear(x.flatten(1), self.weight.flatten(1), self.bias)
        return y.view(x.shape[0], -1, 1, 1, 1)


def apply_platform_patches(visual: torch.nn.Module) -> list[str]:
    """Patch a Qwen3-VL vision tower in place when cuDNN is unavailable."""
    if torch.backends.cudnn.is_available():
        return []
    visual.patch_embed.proj = PatchConvAsLinear(visual.patch_embed.proj)
    return ["vision patch_embed Conv3d -> equivalent F.linear on the same weights (no cuDNN on this build)"]


def batched_vision_attention_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
    """Qwen3-VL vision attention as one batched call when every image has the same patch count.

    transformers loops over images (16 per call x 27 blocks = 432 attention launches). Images never attend
    across each other, so one batch of equal-length images computes the same values. Falls back to the
    original loop for unequal grids.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if lengths.numel() < 2 or bool((lengths != lengths[0]).any()):
        return self._original_forward(hidden_states, cu_seqlens, rotary_pos_emb, position_embeddings, **kwargs)
    n, length = lengths.numel(), int(lengths[0])
    seq = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = (t.reshape(n, length, self.num_heads, -1).transpose(1, 2) for t in (q, k, v))
    if self.config._attn_implementation == "eager":  # explicit math for the ONNX exporter
        out = torch.matmul(F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scaling, dim=-1), v)
    else:
        out = F.scaled_dot_product_attention(q, k, v)
    return self.proj(out.transpose(1, 2).reshape(seq, -1))


def enable_batched_vision_attention(visual: torch.nn.Module) -> int:
    n = 0
    for block in visual.blocks:
        attn = block.attn
        attn._original_forward = attn.forward
        attn.forward = batched_vision_attention_forward.__get__(attn)
        n += 1
    return n


class ScaledLayerNorm(torch.nn.Module):
    """LayerNorm with the same weights, safe in FP16 for any input magnitude.

    Vision activations reach ~550 from block 10 and ~12,900 at the last block, so (x-mean)^2 overflows FP16
    (max 65,504) in TensorRT's decomposed LayerNorm. Dividing each token by its own largest |d| bounds the squares
    by 1 and leaves the ratio d / std unchanged. The epsilon becomes a 1e-7 floor because the scaled variance is
    at least 1/N, so it shifts the result by well under 0.01%; the small offset guards against an all-equal token.
    """

    def __init__(self, norm: torch.nn.LayerNorm, floor: float = 1e-7):
        super().__init__()
        self.weight, self.bias = norm.weight, norm.bias
        self.floor = floor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x - x.mean(-1, keepdim=True)
        d = d / (d.abs().amax(-1, keepdim=True) + 1e-3)
        y = d / torch.sqrt((d * d).mean(-1, keepdim=True) + self.floor)
        return y * self.weight + self.bias


def enable_scaled_layernorm(root: torch.nn.Module) -> int:
    """Replace every nn.LayerNorm below root. Returns how many were replaced."""
    targets = [(n, m) for n, m in root.named_modules() if isinstance(m, torch.nn.LayerNorm)]
    for name, norm in targets:
        parent = root.get_submodule(name.rpartition(".")[0])
        setattr(parent, name.rpartition(".")[2], ScaledLayerNorm(norm))
    return len(targets)


class ScaledRMSNorm(torch.nn.Module):
    """Qwen3 text RMSNorm with the same weight, safe in FP16: scale each token by its largest |x| before squaring."""

    def __init__(self, norm, floor: float = 1e-7):
        super().__init__()
        self.weight, self.floor = norm.weight, floor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x / (x.abs().amax(-1, keepdim=True) + 1e-3)
        return self.weight * (y / torch.sqrt((y * y).mean(-1, keepdim=True) + self.floor))


def enable_scaled_rmsnorm(root: torch.nn.Module) -> int:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

    targets = [(n, m) for n, m in root.named_modules() if isinstance(m, Qwen3VLTextRMSNorm)]
    for name, norm in targets:
        parent = root.get_submodule(name.rpartition(".")[0]) if "." in name else root
        setattr(parent, name.rpartition(".")[2], ScaledRMSNorm(norm))
    return len(targets)
