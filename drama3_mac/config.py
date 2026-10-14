"""Parsers for the fishaudio/drama-3 ``config.json``.

Mirrors the semantics of fish-speech's ``fish_omni.config`` for the fields the
Mac runtime needs.  The architecture flags are fixed by the released model; we
validate the ones that would silently change numerics and ignore the rest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FISH_OMNI_MODEL_TYPE = "fish_omni"
FP8_QUANT_METHOD = "bsgemm-moe"
FP8_BLOCK_SIZE = 128
SPEECH_TEMPLATE_VERSION = "reference-bracket-v1"


@dataclass(frozen=True)
class TextConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    partial_rotary_factor: float
    rms_norm_eps: float
    max_position_embeddings: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def full_attention_interval(self) -> int:
        return self.layer_types.index("full_attention") + 1


@dataclass(frozen=True)
class AudioDecoderConfig:
    dim: int
    intermediate_size: int
    n_head: int
    n_local_heads: int
    head_dim: int
    norm_eps: float
    rope_base: float
    rotary_percent: float
    vocab_size: int
    num_codebooks: int
    shared_enrichment_layers: int
    vq_backbone_codebooks: int

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_percent)


@dataclass(frozen=True)
class ModelConfig:
    text: TextConfig
    audio_decoder: AudioDecoderConfig
    semantic_start_token_id: int
    semantic_end_token_id: int
    eos_token_id: int
    audio_pad_token_id: int
    pad_token_id: int
    fp8_experts: bool

    @property
    def num_codebooks(self) -> int:
        return self.audio_decoder.num_codebooks

    @property
    def frame_width(self) -> int:
        """Columns of a frame row: the main token plus one code per codebook."""
        return 1 + self.audio_decoder.num_codebooks


def _text_config(data: dict) -> TextConfig:
    if data.get("model_type") != "fish_omni_text":
        raise ValueError(f"unexpected text model_type {data.get('model_type')!r}")
    layer_types = data["layer_types"]
    rope = data.get("rope_parameters") or data
    cfg = TextConfig(
        vocab_size=data["vocab_size"],
        hidden_size=data["hidden_size"],
        num_hidden_layers=data["num_hidden_layers"],
        layer_types=tuple(layer_types),
        num_attention_heads=data["num_attention_heads"],
        num_key_value_heads=data["num_key_value_heads"],
        head_dim=data["head_dim"],
        rope_theta=rope["rope_theta"],
        partial_rotary_factor=rope.get("partial_rotary_factor", 0.25),
        rms_norm_eps=data["rms_norm_eps"],
        max_position_embeddings=data["max_position_embeddings"],
        linear_conv_kernel_dim=data["linear_conv_kernel_dim"],
        linear_key_head_dim=data["linear_key_head_dim"],
        linear_value_head_dim=data["linear_value_head_dim"],
        linear_num_key_heads=data["linear_num_key_heads"],
        linear_num_value_heads=data["linear_num_value_heads"],
        num_experts=data["num_experts"],
        num_experts_per_tok=data["num_experts_per_tok"],
        moe_intermediate_size=data["moe_intermediate_size"],
        shared_expert_intermediate_size=data["shared_expert_intermediate_size"],
        norm_topk_prob=data.get("norm_topk_prob", True),
    )
    if len(cfg.layer_types) != cfg.num_hidden_layers:
        raise ValueError("layer_types length mismatch")
    if cfg.rotary_dim <= 0 or cfg.rotary_dim % 2:
        raise ValueError(f"bad rotary_dim {cfg.rotary_dim}")
    if data.get("tie_word_embeddings", False):
        raise ValueError("the backbone keeps a separate lm_head")
    return cfg


def _audio_decoder_config(data: dict) -> AudioDecoderConfig:
    if data.get("decoder_type") != "codebook_mtp_shared":
        raise ValueError(f"unsupported decoder_type {data.get('decoder_type')!r}")
    return AudioDecoderConfig(
        dim=data["dim"],
        intermediate_size=data["intermediate_size"],
        n_head=data["n_head"],
        n_local_heads=data["n_local_heads"],
        head_dim=data["head_dim"],
        norm_eps=data["norm_eps"],
        rope_base=data["rope_base"],
        rotary_percent=data["rotary_percent"],
        vocab_size=data["vocab_size"],
        num_codebooks=data["num_codebooks"],
        shared_enrichment_layers=data["shared_enrichment_layers"],
        vq_backbone_codebooks=data["vq_backbone_codebooks"],
    )


def model_config_from_dict(data: dict) -> ModelConfig:
    if data.get("model_type") != FISH_OMNI_MODEL_TYPE:
        raise ValueError(f"model_type={data.get('model_type')!r} is not fish_omni")
    if data.get("speech_template_version") != SPEECH_TEMPLATE_VERSION:
        raise ValueError("unsupported speech_template_version")
    quantization = data.get("quantization_config")
    fp8_experts = quantization is not None
    if fp8_experts and quantization.get("quant_method") != FP8_QUANT_METHOD:
        raise ValueError(f"unsupported quant_method {quantization.get('quant_method')!r}")
    eos = data.get("eos_token_id") or data["text_config"]["eos_token_id"]
    return ModelConfig(
        text=_text_config(data["text_config"]),
        audio_decoder=_audio_decoder_config(data["audio_decoder_config"]),
        semantic_start_token_id=data["semantic_start_token_id"],
        semantic_end_token_id=data["semantic_end_token_id"],
        eos_token_id=eos,
        audio_pad_token_id=data.get("audio_pad_token_id", 248076),
        pad_token_id=data.get("pad_token_id", 248044),
        fp8_experts=fp8_experts,
    )


def load_config(checkpoint_dir: str | Path) -> ModelConfig:
    with open(Path(checkpoint_dir) / "config.json", encoding="utf-8") as f:
        return model_config_from_dict(json.load(f))
