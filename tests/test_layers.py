"""Numerical parity tests: MLX layers vs the fish-speech PyTorch reference.

The reference modules run in fp32 on CPU; the MLX side is also run in fp32 so
tolerances can stay tight.  Reduced head counts keep runtime low while still
exercising GQA grouping and the GatedDeltaNet head mapping.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "research" / "fish-speech"))

from fish_speech.models.text2semantic.fish_omni.layers import (  # noqa: E402
    FullAttention as TorchFullAttention,
    GatedDeltaNet as TorchGatedDeltaNet,
    GatedRMSNorm as TorchGatedRMSNorm,
    LayerContext as TorchLayerContext,
    ZeroCenteredRMSNorm as TorchZeroCenteredRMSNorm,
)
from fish_speech.models.text2semantic.fish_omni.config import (  # noqa: E402
    FishOmniTextConfig as TorchTextConfig,
)

from drama3_mac.layers import (  # noqa: E402
    FullAttention as MlxFullAttention,
    GatedDeltaNet as MlxGatedDeltaNet,
    GatedRMSNorm as MlxGatedRMSNorm,
    LayerContext as MlxLayerContext,
    Linear as MlxLinear,
    ZeroCenteredRMSNorm as MlxZeroCenteredRMSNorm,
    apply_neox_rope as mlx_rope,
    build_cos_sin,
    rope_frequencies,
)

DTYPE = torch.float32


def to_mx(t):
    return mx.array(t.detach().numpy())


def to_torch(a):
    return torch.from_numpy(np.array(a))


def make_text_config(**overrides):
    base = dict(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
        max_position_embeddings=2048,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        norm_topk_prob=True,
    )
    base.update(overrides)
    return TorchTextConfig(**base)


def copy_norm(torch_norm, eps=1e-6):
    weight = torch_norm.weight.detach().to(DTYPE)
    return MlxZeroCenteredRMSNorm(to_mx(weight))


def test_zero_centered_rmsnorm():
    torch.manual_seed(0)
    ref = TorchZeroCenteredRMSNorm(64, 1e-6, dtype=DTYPE)
    x = torch.randn(3, 5, 64, dtype=DTYPE)
    expected = ref(x)
    got = MlxZeroCenteredRMSNorm(to_mx(ref.weight))(to_mx(x))
    assert mx.abs(got - to_mx(expected)).max().item() < 1e-5


def test_gated_rmsnorm():
    torch.manual_seed(0)
    ref = TorchGatedRMSNorm(32, 1e-6, dtype=DTYPE)
    x = torch.randn(4, 32, dtype=DTYPE)
    gate = torch.randn(4, 32, dtype=DTYPE)
    expected = ref(x, gate)
    got = MlxGatedRMSNorm(to_mx(ref.weight))(to_mx(x), to_mx(gate))
    assert mx.abs(got - to_mx(expected)).max().item() < 1e-5


def test_neox_rope():
    torch.manual_seed(0)
    freqs = rope_frequencies(10000.0, 16)
    cos, sin = build_cos_sin(freqs, 0, 7)
    x = torch.randn(2, 7, 3, 32, dtype=DTYPE)

    # reference path
    angles = torch.arange(7).float()[:, None] * (1.0 / (10000.0 ** (torch.arange(0, 16, 2).float() / 16)))
    angles = torch.cat((angles, angles), dim=-1)
    rotary_dim = 16
    rotary, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cos_t, sin_t = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
    expected = torch.cat((rotary * cos_t + rotated * sin_t, passthrough), dim=-1)

    got = mlx_rope(to_mx(x), cos, sin)
    assert mx.abs(got - to_mx(expected)).max().item() < 1e-5


def build_attention_pair(config, seed=1):
    torch.manual_seed(seed)
    ref = TorchFullAttention(config, dtype=DTYPE)
    mx_attn = MlxFullAttention(
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        q_proj=MlxLinear(to_mx(ref.q_proj.weight)),
        k_proj=MlxLinear(to_mx(ref.k_proj.weight)),
        v_proj=MlxLinear(to_mx(ref.v_proj.weight)),
        o_proj=MlxLinear(to_mx(ref.o_proj.weight)),
        q_norm=copy_norm(ref.q_norm),
        k_norm=copy_norm(ref.k_norm),
    )
    return ref, mx_attn


def attention_context(start, length, config, use_cache=True):
    freqs = rope_frequencies(config.rope_theta, int(config.head_dim * config.partial_rotary_factor))
    cos, sin = build_cos_sin(freqs, start, length)
    return MlxLayerContext(
        start=start, length=length, cos=cos, sin=sin,
        kv_len=start + length, continue_state=start > 0, use_cache=use_cache,
    )

def torch_context(start, length, config, device=torch.device("cpu"), use_cache=True):
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
    frequencies = 1.0 / (config.rope_theta ** (exponents / rotary_dim))
    positions = torch.arange(start, start + length)
    angles = positions.float()[:, None] * frequencies[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return TorchLayerContext(
        input_pos=positions, cos=angles.cos(), sin=angles.sin(),
        kv_len=start + length, continue_state=start > 0, use_cache=use_cache,
    )


def test_full_attention_prefill_and_decode():
    config = make_text_config()
    ref, mx_attn = build_attention_pair(config)
    x = torch.randn(1, 9, config.hidden_size, dtype=DTYPE)

    ref_out = ref(x, torch_context(0, 9, config, use_cache=False))
    mx_out = mx_attn(to_mx(x), attention_context(0, 9, config, use_cache=False))[0]
    assert mx.abs(mx_out - to_mx(ref_out)).max().item() < 2e-4

    # decode with caches: run prefill with cache, then one step
    ref.allocate_cache(1, 16, DTYPE, torch.device("cpu"))
    mx_attn.allocate_cache(1, 16)
    ref_out = ref(x, torch_context(0, 9, config))
    mx_out = mx_attn(to_mx(x), attention_context(0, 9, config))[0]
    assert mx.abs(mx_out - to_mx(ref_out)).max().item() < 2e-4

    x2 = torch.randn(1, 1, config.hidden_size, dtype=DTYPE)
    ref_out = ref(x2, torch_context(9, 1, config))
    mx_out = mx_attn(to_mx(x2), attention_context(9, 1, config))[0]
    assert mx.abs(mx_out - to_mx(ref_out)).max().item() < 2e-4


def build_gdn_pair(config, seed=2):
    torch.manual_seed(seed)
    ref = TorchGatedDeltaNet(config, dtype=DTYPE)
    linear = lambda m: MlxLinear(to_mx(m.weight))
    mx_gdn = MlxGatedDeltaNet(
        num_k_heads=config.linear_num_key_heads,
        num_v_heads=config.linear_num_value_heads,
        head_k_dim=config.linear_key_head_dim,
        head_v_dim=config.linear_value_head_dim,
        conv_kernel_size=config.linear_conv_kernel_dim,
        in_proj_qkv=linear(ref.in_proj_qkv),
        in_proj_z=linear(ref.in_proj_z),
        in_proj_b=linear(ref.in_proj_b),
        in_proj_a=linear(ref.in_proj_a),
        conv1d_weight=to_mx(ref.conv1d.weight),
        dt_bias=to_mx(ref.dt_bias),
        a_log=to_mx(ref.A_log),
        norm=MlxGatedRMSNorm(to_mx(ref.norm.weight)),
        out_proj=linear(ref.out_proj),
    )
    return ref, mx_gdn


def test_gated_delta_net_prefill_and_decode():
    config = make_text_config()
    ref, mx_gdn = build_gdn_pair(config)
    x = torch.randn(1, 20, config.hidden_size, dtype=DTYPE)

    ref_out = ref(x, torch_context(0, 20, config, use_cache=False))
    mx_out = mx_gdn(to_mx(x), attention_context(0, 20, config, use_cache=False))[0]
    err = mx.abs(mx_out.astype(mx.float32) - to_mx(ref_out)).max().item()
    assert err < 2e-3, f"prefill mismatch {err}"

    ref.allocate_cache(1, DTYPE, torch.device("cpu"))
    mx_gdn.allocate_cache(1)
    ref_out = ref(x, torch_context(0, 20, config))
    mx_out = mx_gdn(to_mx(x), attention_context(0, 20, config))[0]
    err = mx.abs(mx_out.astype(mx.float32) - to_mx(ref_out)).max().item()
    assert err < 2e-3, f"cached prefill mismatch {err}"

    for step in range(3):
        x2 = torch.randn(1, 1, config.hidden_size, dtype=DTYPE)
        ref_out = ref(x2, torch_context(20 + step, 1, config))
        mx_out = mx_gdn(to_mx(x2), attention_context(20 + step, 1, config))[0]
        err = mx.abs(mx_out.astype(mx.float32) - to_mx(ref_out)).max().item()
        assert err < 2e-3, f"decode step {step} mismatch {err}"


def test_gdn_long_prefill_multi_chunk():
    """Two chunks: the second continues the first's recurrent state (kernel path)."""
    config = make_text_config()
    ref, mx_gdn = build_gdn_pair(config)
    ref.allocate_cache(1, DTYPE, torch.device("cpu"))
    mx_gdn.allocate_cache(1)
    x = torch.randn(1, 70, config.hidden_size, dtype=DTYPE)
    ref_out = ref(x, torch_context(0, 70, config))
    mx_out = mx_gdn(to_mx(x), attention_context(0, 70, config))[0]
    err = mx.abs(mx_out.astype(mx.float32) - to_mx(ref_out)).max().item()
    assert err < 5e-3, f"multi-chunk mismatch {err}"
