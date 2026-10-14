"""FishOmni text backbone in MLX, built directly from a converted checkpoint.

Weights follow the converted names with the ``model.language_model.`` prefix
stripped (``model.embed_tokens``, ``model.layers.N...``, ``model.norm``,
``lm_head``).  The residual stream runs in bf16, matching the reference for
bfloat16 checkpoints.
"""

from __future__ import annotations

import mlx.core as mx

from .config import TextConfig
from .layers import (
    FullAttention,
    GatedDeltaNet,
    GatedRMSNorm,
    LayerContext,
    Linear,
    ZeroCenteredRMSNorm,
    build_cos_sin,
    rope_frequencies,
)
from .module import Module
from .moe import Experts, SparseMoeBlock


def _linear(weights: dict, name: str, group_size: int, bits: int) -> Linear:
    weight = weights[f"{name}.weight"]
    if f"{name}.scales" in weights:
        return Linear(
            weight,
            weights[f"{name}.scales"],
            weights[f"{name}.biases"],
            group_size,
            bits,
        )
    return Linear(weight)


def _norm(weights: dict, name: str) -> ZeroCenteredRMSNorm:
    return ZeroCenteredRMSNorm(weights[f"{name}.weight"])


class DecoderLayer(Module):
    def __init__(self, layer_type: str, mixer, mlp, input_layernorm, post_attention_layernorm):
        self.layer_type = layer_type
        if layer_type == "full_attention":
            self.self_attn = mixer
        else:
            self.linear_attn = mixer
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self._gdn_fn = None
        self._attn_fn = None

    def _build_gdn_step(self):
        mixer, norm, mlp = self.linear_attn, self.input_layernorm, self.mlp
        post = self.post_attention_layernorm

        def step(x, conv, state):
            out, conv_new, state_new = mixer.step_with_caches(norm(x), conv, state)
            x = x + out
            return x + mlp._call_flat(post(x)), conv_new, state_new

        return mx.compile(step)

    def _build_attn_step(self):
        mixer, norm, mlp = self.self_attn, self.input_layernorm, self.mlp
        post = self.post_attention_layernorm

        def step(x, cos, sin, k_cache, v_cache, position):
            out, k_new, v_new = mixer.decode_with_caches(norm(x), cos, sin, k_cache, v_cache, position)
            x = x + out
            return x + mlp._call_flat(post(x)), k_new, v_new

        return mx.compile(step)

    def __call__(self, x: mx.array, context: LayerContext) -> mx.array:
        if self.layer_type == "full_attention":
            x = x + self.self_attn(self.input_layernorm(x), context)
        elif context.length == 1 and context.continue_state and context.use_cache:
            # decode fast path: the GDN step has no position-dependent slicing,
            # so one compiled graph per layer serves every step
            mixer = self.linear_attn
            if self._gdn_fn is None:
                def step(h, conv, state, _mixer=mixer, _norm=self.input_layernorm):
                    return _mixer.step_with_caches(_norm(h), conv, state)
                self._gdn_fn = mx.compile(step)
            out, conv_new, state_new = self._gdn_fn(
                x, mixer.cache("conv"), mixer.cache("state")
            )
            mixer.cache("conv", conv_new)
            mixer.cache("state", state_new)
            x = x + out
        else:
            x = x + self.linear_attn(self.input_layernorm(x), context)
        return x + self.mlp(self.post_attention_layernorm(x))


class FishOmniTextModel(Module):
    def __init__(self, embed_tokens: mx.array, layers: list, norm: ZeroCenteredRMSNorm):
        self.embed_tokens = embed_tokens
        self.layers = layers
        self.norm = norm


def is_decode_step(length: int, continue_state: bool, use_cache: bool,
                   position: mx.array | None) -> bool:
    """Whether a backbone call is a single-token decode step.

    Decode callers pass ``position``; prefill callers never do.  A 1-row
    prefill chunk (``prompt_rows % prefill_chunk_size == 1``) must NOT take
    the compiled decode path — it has no position, and at a chunk seam the
    eager path's slice-based cache write and mask are the correct semantics.
    """
    return length == 1 and continue_state and use_cache and position is not None


class Backbone(Module):
    def __init__(self, config: TextConfig, model: FishOmniTextModel, lm_head: Linear):
        self.config = config
        self.model = model
        self.lm_head = lm_head
        self._frequencies = rope_frequencies(config.rope_theta, config.rotary_dim)
        self.cache_capacity = 0

    def embed(self, ids: mx.array) -> mx.array:
        return self.model.embed_tokens[ids]

    def setup_caches(self, batch: int, max_seq_len: int) -> None:
        if max_seq_len <= self.cache_capacity:
            return
        for layer in self.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn.allocate_cache(batch, max_seq_len)
            else:
                layer.linear_attn.allocate_cache(batch)
        self.cache_capacity = max_seq_len

    def __call__(
        self,
        inputs_embeds: mx.array,
        *,
        start: int,
        kv_len: int,
        continue_state: bool,
        use_cache: bool,
        position: mx.array | None = None,
    ) -> mx.array:
        length = inputs_embeds.shape[1]
        cos, sin = build_cos_sin(self._frequencies, start, length)
        if is_decode_step(length, continue_state, use_cache, position):
            return self._decode_step(inputs_embeds, cos, sin, position)
        context = LayerContext(
            start=start,
            length=length,
            cos=cos,
            sin=sin,
            kv_len=kv_len,
            continue_state=continue_state,
            use_cache=use_cache,
        )
        hidden = inputs_embeds
        for layer in self.model.layers:
            hidden = layer(hidden, context)
        return self.model.norm(hidden)

    def _decode_step(self, hidden: mx.array, cos: mx.array, sin: mx.array,
                     position: mx.array | None) -> mx.array:
        """One compiled call per layer; caches are threaded in and out."""
        if position is None:
            raise ValueError("decode step needs the position")
        for layer in self.model.layers:
            if layer.layer_type == "full_attention":
                if layer._attn_fn is None:
                    layer._attn_fn = layer._build_attn_step()
                attn = layer.self_attn
                hidden, k_new, v_new = layer._attn_fn(
                    hidden, cos, sin, attn.cache("k"), attn.cache("v"), position
                )
                attn.cache("k", k_new)
                attn.cache("v", v_new)
            else:
                if layer._gdn_fn is None:
                    layer._gdn_fn = layer._build_gdn_step()
                gdn = layer.linear_attn
                hidden, c_new, s_new = layer._gdn_fn(
                    hidden, gdn.cache("conv"), gdn.cache("state")
                )
                gdn.cache("conv", c_new)
                gdn.cache("state", s_new)
        return self.model.norm(hidden)

    def compute_logits(self, hidden: mx.array) -> mx.array:
        return self.lm_head(hidden)


def build_backbone(
    weights: dict, config: TextConfig, *, group_size: int, bits: int
) -> Backbone:
    """Assemble the backbone from a converted weight dict (prefix stripped)."""
    layers = []
    for index in range(config.num_hidden_layers):
        base = f"model.layers.{index}"
        layer_type = config.layer_types[index]
        if layer_type == "full_attention":
            mixer = FullAttention(
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                q_proj=_linear(weights, f"{base}.self_attn.q_proj", group_size, bits),
                k_proj=_linear(weights, f"{base}.self_attn.k_proj", group_size, bits),
                v_proj=_linear(weights, f"{base}.self_attn.v_proj", group_size, bits),
                o_proj=_linear(weights, f"{base}.self_attn.o_proj", group_size, bits),
                q_norm=_norm(weights, f"{base}.self_attn.q_norm"),
                k_norm=_norm(weights, f"{base}.self_attn.k_norm"),
            )
        else:
            mixer = GatedDeltaNet(
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                in_proj_qkv=_linear(
                    weights, f"{base}.linear_attn.in_proj_qkv", group_size, bits
                ),
                in_proj_z=_linear(
                    weights, f"{base}.linear_attn.in_proj_z", group_size, bits
                ),
                in_proj_b=_linear(
                    weights, f"{base}.linear_attn.in_proj_b", group_size, bits
                ),
                in_proj_a=_linear(
                    weights, f"{base}.linear_attn.in_proj_a", group_size, bits
                ),
                conv1d_weight=weights[f"{base}.linear_attn.conv1d.weight"],
                dt_bias=weights[f"{base}.linear_attn.dt_bias"],
                a_log=weights[f"{base}.linear_attn.A_log"],
                norm=GatedRMSNorm(weights[f"{base}.linear_attn.norm.weight"]),
                out_proj=_linear(
                    weights, f"{base}.linear_attn.out_proj", group_size, bits
                ),
            )
        experts_w13 = weights[f"{base}.mlp.experts.w13.weight"]
        quantized = f"{base}.mlp.experts.w13.scales" in weights
        experts = Experts(
            experts_w13,
            weights[f"{base}.mlp.experts.w2.weight"],
            weights.get(f"{base}.mlp.experts.w13.scales"),
            weights.get(f"{base}.mlp.experts.w13.biases"),
            weights.get(f"{base}.mlp.experts.w2.scales"),
            weights.get(f"{base}.mlp.experts.w2.biases"),
            group_size,
            bits,
        )
        mlp = SparseMoeBlock(
            gate=_linear(weights, f"{base}.mlp.gate", group_size, bits),
            experts=experts,
            shared_gate_proj=_linear(
                weights, f"{base}.mlp.shared_expert.gate_proj", group_size, bits
            ),
            shared_up_proj=_linear(
                weights, f"{base}.mlp.shared_expert.up_proj", group_size, bits
            ),
            shared_down_proj=_linear(
                weights, f"{base}.mlp.shared_expert.down_proj", group_size, bits
            ),
            shared_expert_gate=_linear(
                weights, f"{base}.mlp.shared_expert_gate", group_size, bits
            ),
            top_k=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
        )
        layers.append(
            DecoderLayer(
                layer_type,
                mixer,
                mlp,
                _norm(weights, f"{base}.input_layernorm"),
                _norm(weights, f"{base}.post_attention_layernorm"),
            )
        )
    model = FishOmniTextModel(
        weights["model.embed_tokens.weight"],
        layers,
        _norm(weights, "model.norm"),
    )
    lm_head = _linear(weights, "lm_head", group_size, bits)
    return Backbone(config, model, lm_head)


def strip_prefix(weights: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
