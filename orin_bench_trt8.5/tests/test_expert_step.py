"""The exported expert step (fixed KV length, -1e4 mask, external rotary angles) must reproduce the official expert call.

The hybrid runtime pads the prompt KV cache with zeros to the engine's fixed length and masks the padding. This checks
that construction on a tiny random Qwen3-VL text model in float64 on the CPU.
"""

import copy

import torch
from transformers import AutoModel, DynamicCache, Qwen3VLConfig
from transformers.models.qwen3_vl import modeling_qwen3_vl as mq

from orinbench.export import ExpertStep, _sdpa_math_eager
from orinbench.hybrid import MASKED


class InProj(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lin = torch.nn.Linear(3, hidden)

    def forward(self, x, t):
        return self.lin(torch.cat([x, t.expand(*x.shape[:-1], 1)], -1))


def tiny_expert():
    text = copy.deepcopy(Qwen3VLConfig().text_config)
    text.num_hidden_layers, text.hidden_size, text.intermediate_size = 2, 64, 128
    text.num_attention_heads, text.num_key_value_heads, text.head_dim = 4, 2, 16
    text.rope_scaling = {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}
    text._attn_implementation = "eager"
    model = AutoModel.from_config(text).double().eval()
    del model.embed_tokens
    return model, text


def test_padded_engine_step_matches_official_call(monkeypatch):
    monkeypatch.setattr(mq, "eager_attention_forward", _sdpa_math_eager)
    torch.manual_seed(0)
    expert, text = tiny_expert()
    n, prompt_len, cache_len = 8, 21, 32
    aip, aop = InProj(text.hidden_size).double(), torch.nn.Linear(text.hidden_size, 2).double()
    kv = [(torch.randn(1, text.num_key_value_heads, prompt_len, text.head_dim, dtype=torch.float64),
           torch.randn(1, text.num_key_value_heads, prompt_len, text.head_dim, dtype=torch.float64))
          for _ in range(text.num_hidden_layers)]
    x, t = torch.randn(1, n, 2, dtype=torch.float64), torch.full((1, 1, 1), 0.3, dtype=torch.float64)
    pos = torch.arange(n).view(1, 1, -1).expand(3, 1, -1).clone() + 500  # positions after rope deltas
    # official mask: the gap between <traj_future_start> and the diffusion tokens is masked (here 3 tokens)
    official = torch.zeros(1, 1, n, prompt_len + n, dtype=torch.float64)
    official[..., prompt_len - 3: prompt_len] = torch.finfo(torch.float64).min

    with torch.no_grad():
        cache = DynamicCache()
        for i, (k, v) in enumerate(kv):
            cache.update(k.clone(), v.clone(), i)
        out = expert(inputs_embeds=aip(x, t), position_ids=pos, past_key_values=cache, attention_mask=official,
                     use_cache=True, is_causal=False)
        want = aop(out.last_hidden_state[:, -n:])

        cos, sin = expert.rotary_emb(x, pos)
        step = ExpertStep(copy.deepcopy(expert), aip, aop, (n, 2), pos).eval()
        mask = torch.zeros(1, 1, n, cache_len + n, dtype=torch.float64)  # as HybridAlpamayo._load_expert_inputs builds it
        mask[..., :prompt_len][official[..., :prompt_len] < 0] = MASKED
        mask[..., prompt_len:cache_len] = MASKED
        mask[..., cache_len:] = official[..., prompt_len:]
        padded = []
        for k, v in kv:
            padded += [torch.nn.functional.pad(k, (0, 0, 0, cache_len - prompt_len)),
                       torch.nn.functional.pad(v, (0, 0, 0, cache_len - prompt_len))]
        got = step(x, t, mask, cos, sin, *padded)
    torch.testing.assert_close(got.view_as(want), want, rtol=1e-9, atol=1e-9)
