"""Parity tests for the MLX MoE block and the codebook tower."""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "research" / "fish-speech"))

from fish_speech.models.text2semantic.fish_omni.config import (  # noqa: E402
    AudioDecoderConfig as TorchAudioConfig,
    FishOmniConfig as TorchFishConfig,
    FishOmniTextConfig as TorchTextConfig,
)
from fish_speech.models.text2semantic.fish_omni.moe import (  # noqa: E402
    SparseMoeBlock as TorchMoe,
)
from fish_speech.models.text2semantic.fish_omni.tower import (  # noqa: E402
    FishOmniAudioDecoder as TorchTower,
)

from drama3_mac.config import AudioDecoderConfig, ModelConfig, TextConfig  # noqa: E402
from drama3_mac.layers import Linear as MlxLinear, ZeroCenteredRMSNorm  # noqa: E402
from drama3_mac.moe import Experts as MlxExperts, SparseMoeBlock as MlxMoe  # noqa: E402
from drama3_mac.tower import build_tower  # noqa: E402

DTYPE = torch.float32


def to_mx(t):
    return mx.array(t.detach().numpy())


def to_torch(a):
    return torch.from_numpy(np.array(a))


def torch_text_config(**kw):
    base = dict(
        vocab_size=1000, hidden_size=64, num_hidden_layers=1,
        layer_types=["full_attention"], num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, rope_theta=10000.0, partial_rotary_factor=0.5, rms_norm_eps=1e-6,
        max_position_embeddings=2048, linear_conv_kernel_dim=4, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, norm_topk_prob=True,
    )
    base.update(kw)
    return TorchTextConfig(**base)


def test_moe_matches_reference():
    torch.manual_seed(3)
    config = torch_text_config()
    ref = TorchMoe(config, fp8=False, dtype=DTYPE)
    with torch.no_grad():  # the reference allocates torch.empty slots
        for parameter in ref.parameters():
            torch.nn.init.normal_(parameter, std=0.1)

    experts = MlxExperts(
        to_mx(ref.experts.w13_weight), to_mx(ref.experts.w2_weight)
    )
    mine = MlxMoe(
        gate=MlxLinear(to_mx(ref.gate.weight)),
        experts=experts,
        shared_gate_proj=MlxLinear(to_mx(ref.shared_expert.gate_proj.weight)),
        shared_up_proj=MlxLinear(to_mx(ref.shared_expert.up_proj.weight)),
        shared_down_proj=MlxLinear(to_mx(ref.shared_expert.down_proj.weight)),
        shared_expert_gate=MlxLinear(to_mx(ref.shared_expert_gate.weight)),
        top_k=config.num_experts_per_tok,
    )

    for tokens in (5, 200):  # gathered and grouped paths
        x = torch.randn(tokens, config.hidden_size, dtype=DTYPE)
        expected = ref(x)
        got = mine(to_mx(x))
        err = mx.abs(got.astype(mx.float32) - to_mx(expected)).max().item()
        assert err < 5e-4, f"tokens={tokens} err={err}"


def torch_fish_config():
    text = torch_text_config()
    audio = TorchAudioConfig(
        dim=64, intermediate_size=128, n_head=4, n_local_heads=2, head_dim=32,
        norm_eps=1e-6, rope_base=10000.0, rotary_percent=0.25, vocab_size=64,
        num_codebooks=4, shared_enrichment_layers=2, vq_backbone_codebooks=1,
    )
    return TorchFishConfig(
        text=text, audio_decoder=audio,
        semantic_start_token_id=100, semantic_end_token_id=163, eos_token_id=50,
    )


def mlx_model_config():
    return ModelConfig(
        text=TextConfig(
            vocab_size=1000, hidden_size=64, num_hidden_layers=1,
            layer_types=("full_attention",), num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, rope_theta=10000.0,
            partial_rotary_factor=0.5, rms_norm_eps=1e-6,
            max_position_embeddings=2048, linear_conv_kernel_dim=4,
            linear_key_head_dim=32, linear_value_head_dim=32,
            linear_num_key_heads=2, linear_num_value_heads=4,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            shared_expert_intermediate_size=32, norm_topk_prob=True,
        ),
        audio_decoder=AudioDecoderConfig(
            dim=64, intermediate_size=128, n_head=4, n_local_heads=2, head_dim=32,
            norm_eps=1e-6, rope_base=10000.0, rotary_percent=0.25, vocab_size=64,
            num_codebooks=4, shared_enrichment_layers=2, vq_backbone_codebooks=1,
        ),
        semantic_start_token_id=100, semantic_end_token_id=163, eos_token_id=50,
        audio_pad_token_id=90, pad_token_id=80, fp8_experts=False,
    )


def build_tower_pair(seed=4):
    torch.manual_seed(seed)
    config = torch_fish_config()
    ref = TorchTower(config, dtype=DTYPE)
    mlx_config = mlx_model_config()

    weights = {}
    weights["audio_decoder.codebook_embeddings.weight"] = to_mx(ref.codebook_embeddings.weight)
    for index, layer in enumerate(ref.shared_layers):
        prefix = f"audio_decoder_mtp_shared.layers.{index}"
        weights[f"{prefix}.semantic_embedding.weight"] = to_mx(layer.semantic_embedding.weight)
        _copy_layer(weights, prefix, layer)
    for index, layer in enumerate(ref.residual_layers):
        prefix = f"audio_decoder_mtp.layers.{index}"
        weights[f"{prefix}.codebook_embedding.weight"] = to_mx(layer.codebook_embedding.weight)
        _copy_layer(weights, prefix, layer)
    for index, head in enumerate(ref.codebook_heads):
        weights[f"audio_decoder_mtp.heads.{index}.weight"] = to_mx(head.weight)
    return ref, build_tower(weights, mlx_config)


def _copy_layer(weights, prefix, layer):
    t = layer.transformer_layer
    weights[f"{prefix}.transformer_layer.attention.wqkv.weight"] = to_mx(t.attention.wqkv.weight)
    weights[f"{prefix}.transformer_layer.attention.wo.weight"] = to_mx(t.attention.wo.weight)
    weights[f"{prefix}.transformer_layer.attention.q_norm.weight"] = to_mx(t.attention.q_norm.weight)
    weights[f"{prefix}.transformer_layer.attention.k_norm.weight"] = to_mx(t.attention.k_norm.weight)
    weights[f"{prefix}.transformer_layer.attention_norm.weight"] = to_mx(t.attention_norm.weight)
    weights[f"{prefix}.transformer_layer.ffn_norm.weight"] = to_mx(t.ffn_norm.weight)
    weights[f"{prefix}.transformer_layer.feed_forward.w1.weight"] = to_mx(t.feed_forward.w1.weight)
    weights[f"{prefix}.transformer_layer.feed_forward.w2.weight"] = to_mx(t.feed_forward.w2.weight)
    weights[f"{prefix}.transformer_layer.feed_forward.w3.weight"] = to_mx(t.feed_forward.w3.weight)
    weights[f"{prefix}.enorm.weight"] = to_mx(layer.enorm.weight)
    weights[f"{prefix}.hnorm.weight"] = to_mx(layer.hnorm.weight)
    weights[f"{prefix}.eh_proj.weight"] = to_mx(layer.eh_proj.weight)
    weights[f"{prefix}.final_layernorm.weight"] = to_mx(layer.final_layernorm.weight)


def test_tower_prefill_and_decode_step():
    ref, mine = build_tower_pair()
    rows = 6
    frames = torch.randint(0, 64, (rows, 5), dtype=torch.long)  # [rows, 1 + 4]
    frames[:, 0] = torch.randint(100, 164, (rows,))
    hidden = torch.randn(rows, 64, dtype=DTYPE)
    positions = torch.arange(rows)

    ref.setup_caches(16)
    mine.setup_caches(16)
    ref.prefill(hidden[:-1], positions[:-1], frames[:-1], frames[1:])
    mine.prefill(to_mx(hidden[:-1]), mx.arange(rows - 1), to_mx(frames[:-1]), to_mx(frames[1:]))

    # decode step after the prompt: same logits -> same sampled frame (tiny temperature)
    logits = torch.randn(1, 1000, dtype=DTYPE)
    ref_out = ref.decode_step(
        logits, hidden[-1:], positions[rows - 1 : rows], frames[rows - 1 : rows],
        frames[:, 0], temperature=1e-5, top_k=30, top_p=0.95,
    )
    mine_out = mine.decode_step(
        to_mx(logits), to_mx(hidden[-1:]), mx.array([rows - 1]), to_mx(frames[rows - 1 : rows]),
        to_mx(frames[:, 0]), temperature=1e-5, top_k=30, top_p=0.95,
    )
    assert np.array_equal(np.array(mine_out), ref_out.numpy()), (
        f"{np.array(mine_out)} != {ref_out.numpy()}"
    )


def test_tower_decode_matches_reference_semantics():
    """Non-greedy sampling: same distribution family; just check value ranges/shapes."""
    ref, mine = build_tower_pair(seed=5)
    rows = 3
    frames = torch.randint(0, 64, (rows, 5), dtype=torch.long)
    frames[:, 0] = torch.randint(100, 164, (rows,))
    hidden = torch.randn(rows, 64, dtype=DTYPE)
    positions = torch.arange(rows)
    ref.setup_caches(16)
    mine.setup_caches(16)
    ref.prefill(hidden[:-1], positions[:-1], frames[:-1], frames[1:])
    mine.prefill(to_mx(hidden[:-1]), mx.arange(rows - 1), to_mx(frames[:-1]), to_mx(frames[1:]))
    logits = torch.randn(1, 1000, dtype=DTYPE)
    out = mine.decode_step(
        to_mx(logits), to_mx(hidden[-1:]), mx.array([rows - 1]), to_mx(frames[rows - 1 : rows]),
        to_mx(frames[:, 0]), temperature=0.8, top_k=30, top_p=0.9,
    )
    arr = np.array(out)
    assert arr.shape == (1, 5)
    assert 50 <= arr[0, 0] <= 163  # main token from eos/band candidates
    assert (arr[0, 1:] >= 0).all() and (arr[0, 1:2] < 64).all() and (arr[0, 2:] < 1024).all()
