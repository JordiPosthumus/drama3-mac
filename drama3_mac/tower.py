"""Codebook tower (audio decoder) in MLX.

Reference: ``fish-speech/fish_speech/models/text2semantic/fish_omni/tower.py``.
The tower enriches each frame's backbone hidden state with the frame's codes:
one shared stack over the semantic code, then one residual layer per remaining
codebook, each with its own head.  Sampling includes the repetition-aware
second draw (RAS) of the main token.
"""

from __future__ import annotations

import mlx.core as mx

from .config import AudioDecoderConfig, ModelConfig
from .layers import Linear, ZeroCenteredRMSNorm
from .module import Module

RAS_WINDOW_FRAMES = 16
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95
MIN_TEMPERATURE = 1e-5


def _exponential_noise(shape) -> mx.array:
    return -mx.log(mx.random.uniform(shape=shape, dtype=mx.float32))


_sampler_cache: dict[int, object] = {}


def _compiled_sampler(vocab: int):
    """mx.compile'd sampler, one variant per vocabulary size."""
    if vocab not in _sampler_cache:
        _sampler_cache[vocab] = mx.compile(sample_top_k_top_p)
    return _sampler_cache[vocab]


def sample_top_k_top_p(
    logits: mx.array, temperature: mx.array, top_k: mx.array, top_p: mx.array,
    noise: mx.array,
) -> mx.array:
    """One id per row; drops the token crossing top_p, except the highest-ranked one.

    ``temperature``/``top_k``/``top_p`` are scalar arrays so the function can be
    compiled once; ``noise`` (exponential, same shape as ``logits``) is drawn by
    the caller so compiled traces stay deterministic.
    """
    scores = logits.astype(mx.float32) / mx.maximum(temperature, MIN_TEMPERATURE)
    values = -mx.sort(-scores, axis=-1)
    ids = mx.argsort(-scores, axis=-1)
    ranks = mx.arange(values.shape[-1])
    probabilities = mx.softmax(values, axis=-1)
    cumulative = mx.cumsum(probabilities, axis=-1)
    dropped = (cumulative > top_p) | (ranks >= top_k)
    dropped = mx.where(mx.arange(values.shape[-1]) == 0, False, dropped)
    picked = mx.where(dropped, mx.array(-mx.inf, mx.float32), values)
    picked = mx.softmax(picked, axis=-1)
    choice = mx.argmax(picked / noise, axis=-1, keepdims=True)
    return mx.take_along_axis(ids, choice, axis=-1)[..., 0]


class TowerAttention(Module):
    """Gated attention over time; wqkv rows are [Q | Gate | K | V], interleaved RoPE."""

    def __init__(
        self,
        cfg: AudioDecoderConfig,
        wqkv: Linear,
        wo: Linear,
        q_norm: ZeroCenteredRMSNorm,
        k_norm: ZeroCenteredRMSNorm,
    ):
        self.cfg = cfg
        self.wqkv = wqkv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm

    def setup_cache(self, max_seq_len: int) -> None:
        shape = (1, self.cfg.n_local_heads, max_seq_len, self.cfg.head_dim)
        dtype = self.wqkv.weight.dtype
        self.cache("k", mx.zeros(shape, dtype))
        self.cache("v", mx.zeros(shape, dtype))

    def rotate(self, x: mx.array, positions: mx.array) -> mx.array:
        """Interleaved (non-neox) partial RoPE in fp32; x [T, heads, head_dim]."""
        width = self.cfg.rotary_dim
        exponents = mx.arange(0, width, 2, dtype=mx.float32)
        angles = positions.astype(mx.float32)[:, None] * self.cfg.rope_base ** (
            -exponents / width
        )
        cos, sin = mx.cos(angles)[:, None, :], mx.sin(angles)[:, None, :]
        even = x[..., :width:2].astype(mx.float32)
        odd = x[..., 1:width:2].astype(mx.float32)
        rotated = mx.stack(
            [even * cos - odd * sin, odd * cos + even * sin], axis=-1
        ).reshape(*x.shape[:-1], width)
        return mx.concatenate([rotated.astype(x.dtype), x[..., width:]], axis=-1)

    def __call__(self, x: mx.array, positions: mx.array) -> mx.array:
        cfg = self.cfg
        length = x.shape[0]
        q_width = cfg.n_head * cfg.head_dim
        kv_width = cfg.n_local_heads * cfg.head_dim
        q, gate, k, v = self.wqkv(x).split([q_width, 2 * q_width, 2 * q_width + kv_width], axis=-1)
        q = self.q_norm(q.reshape(length, cfg.n_head, cfg.head_dim))
        k = self.k_norm(k.reshape(length, cfg.n_local_heads, cfg.head_dim))
        q = self.rotate(q, positions).transpose(1, 0, 2)[None]
        k = self.rotate(k, positions).transpose(1, 0, 2)[None]
        v = v.reshape(length, cfg.n_local_heads, cfg.head_dim).transpose(1, 0, 2)[None]

        k_cache = self.cache("k")
        v_cache = self.cache("v")
        start = int(positions[0])
        end = int(positions[-1]) + 1
        k_cache[:, :, start:end, :] = k
        v_cache[:, :, start:end, :] = v
        k = k_cache[:, :, :end, :]
        v = v_cache[:, :, :end, :]
        rows = positions[:, None]
        cols = mx.arange(end)[None, :]
        mask = mx.where(
            cols <= rows, mx.zeros((), q.dtype), mx.array(-mx.inf, q.dtype)
        )
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=cfg.head_dim**-0.5, mask=mask
        )
        out = out[0].transpose(1, 0, 2).reshape(length, q_width)
        return self.wo(out * mx.sigmoid(gate))


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class TowerMLP(Module):
    def __init__(self, w1: Linear, w2: Linear, w3: Linear):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(_silu(self.w1(x)) * self.w3(x))


class TowerTransformerLayer(Module):
    def __init__(self, attention, attention_norm, ffn_norm, feed_forward):
        self.attention = attention
        self.attention_norm = attention_norm
        self.ffn_norm = ffn_norm
        self.feed_forward = feed_forward

    def __call__(self, x: mx.array, positions: mx.array) -> mx.array:
        x = x + self.attention(self.attention_norm(x), positions)
        return x + self.feed_forward(self.ffn_norm(x))


class TowerLayer(Module):
    """Mix a code embedding into the hidden state, then one pre-norm block."""

    def __init__(self, shared: bool, embedding_name: str, embedding: mx.array,
                 enorm, hnorm, eh_proj: Linear, transformer_layer, final_layernorm):
        self.shared = shared
        setattr(self, embedding_name, embedding)
        self._embedding_name = embedding_name
        self.enorm = enorm
        self.hnorm = hnorm
        self.eh_proj = eh_proj
        self.transformer_layer = transformer_layer
        self.final_layernorm = final_layernorm

    def __call__(self, hidden, codes, semantic, positions, shared_input=None):
        embedding = getattr(self, self._embedding_name)
        codes = mx.clip(codes, 0, embedding.shape[0] - 1)
        e = mx.where(semantic[:, None], embedding[codes], mx.zeros((), hidden.dtype))
        if shared_input is not None:
            e = e + shared_input
        mixed = self.eh_proj(
            mx.concatenate([self.enorm(e), self.hnorm(hidden)], axis=-1)
        )
        out = self.final_layernorm(self.transformer_layer(mixed, positions))
        return out.astype(mixed.dtype)


def _compiled_penalty(vocab: int):
    """Compiled CTRL-style repetition penalty over ``vocab`` candidates.

    Token ids seen in the recent-window buffer get positive logits divided by
    the penalty and negative logits multiplied by it; all others pass through.
    ``window`` is a fixed-shape (W,) int32 buffer (unused slots = -1) and
    ``penalty`` a scalar array, so one compiled trace serves every call.
    """

    def apply(logits: mx.array, window: mx.array, penalty: mx.array) -> mx.array:
        ids = mx.arange(logits.shape[-1])
        hits = (window[:, None] == ids[None, :]).any(axis=0)
        pushed = mx.where(logits > 0, logits / penalty, logits * penalty)
        return mx.where(hits, pushed, logits)

    return mx.compile(apply)


class Tower(Module):
    def __init__(self, config: ModelConfig, cfg: AudioDecoderConfig,
                 codebook_embeddings: mx.array, shared_layers, residual_layers,
                 codebook_heads):
        self.config = config
        self.cfg = cfg
        self.codebook_embeddings = codebook_embeddings
        self.shared_layers = shared_layers
        self.residual_layers = residual_layers
        self.codebook_heads = codebook_heads
        self._candidates: mx.array | None = None

    @property
    def semantic_begin(self) -> int:
        return self.config.semantic_start_token_id

    @property
    def semantic_end(self) -> int:
        return self.config.semantic_end_token_id

    @property
    def eos_id(self) -> int:
        return self.config.eos_token_id

    def setup_caches(self, max_seq_len: int) -> None:
        for layer in (*self.shared_layers, *self.residual_layers):
            layer.transformer_layer.attention.setup_cache(max_seq_len)

    def is_semantic(self, tokens: mx.array) -> mx.array:
        return (tokens >= self.semantic_begin) & (tokens <= self.semantic_end)

    def embedding_sum(self, frames: mx.array, first: int, end: int) -> mx.array:
        """Summed embeddings of codebooks first..end-1 per frame; frames [N, 1+N]."""
        if first == end:
            return mx.zeros((frames.shape[0], self.cfg.dim), frames.dtype)
        rows = frames[:, 1 + first : 1 + end] + mx.arange(first, end) * self.cfg.vocab_size
        flat = rows.reshape(-1)
        gathered = self.codebook_embeddings[flat].reshape(rows.shape[0], end - first, self.cfg.dim)
        return gathered.sum(axis=1).astype(self.codebook_embeddings.dtype)

    def embed_backbone(self, frames: mx.array, text_embeddings: mx.array) -> mx.array:
        """Backbone input: semantic frames add the first residual code's embedding."""
        count = self.cfg.vq_backbone_codebooks
        extra = self.embedding_sum(frames, 1, 1 + count)
        mixed = (text_embeddings + extra) / (1.0 + count) ** 0.5
        semantic = self.is_semantic(frames[:, 0])
        return mx.where(semantic[:, None], mixed, text_embeddings)

    def run_shared(self, hidden, c0, semantic, positions, frames):
        unseen = self.embedding_sum(
            frames, 1 + self.cfg.vq_backbone_codebooks, self.cfg.num_codebooks
        )
        unseen = mx.where(self.is_semantic(frames[:, 0])[:, None], unseen, mx.zeros((), hidden.dtype))
        for index, layer in enumerate(self.shared_layers):
            hidden = layer(
                hidden, c0, semantic, positions, unseen if index == 0 else None
            )
        return hidden

    def prefill(self, hidden, positions, frames, next_frames):
        """Teacher-force the tower cache; row p is labelled with frame p + 1."""
        semantic = self.is_semantic(next_frames[:, 0])
        hidden = self.run_shared(hidden, next_frames[:, 1], semantic, positions, frames)
        for index, layer in enumerate(self.residual_layers):
            hidden = layer(hidden, next_frames[:, index + 1], semantic, positions)

    def decode_step(self, logits, hidden, positions, frames, history, *,
                    temperature: float, top_k: int, top_p: float,
                    repetition_penalty: float = 1.0,
                    penalty_window: mx.array | None = None) -> mx.array:
        """Sample the frame after each row; the main token is drawn over eos + semantics."""
        if self._candidates is None:
            eos = mx.array([self.eos_id])
            band = mx.arange(self.semantic_begin, self.semantic_end + 1)
            self._candidates = mx.concatenate([eos, band])
        candidates = self._candidates
        candidate_logits = logits[:, candidates]
        if repetition_penalty != 1.0:
            if penalty_window is None:
                raise ValueError("repetition_penalty needs penalty_window")
            penalty = _compiled_penalty(candidate_logits.shape[-1])
            candidate_logits = penalty(
                candidate_logits,
                penalty_window,
                mx.array(repetition_penalty, mx.float32),
            )
        temperature_a = mx.array(temperature, mx.float32)
        top_k_a = mx.array(top_k, mx.int32)
        top_p_a = mx.array(top_p, mx.float32)
        ras_temperature_a = mx.array(RAS_TEMPERATURE, mx.float32)
        ras_top_k_a = mx.array(RAS_TOP_K, mx.int32)
        ras_top_p_a = mx.array(RAS_TOP_P, mx.float32)
        sample = _compiled_sampler(candidate_logits.shape[-1])
        first = candidates[
            sample(candidate_logits, temperature_a, top_k_a, top_p_a,
                   _exponential_noise(candidate_logits.shape))
        ]
        retry = candidates[
            sample(candidate_logits, ras_temperature_a, ras_top_k_a, ras_top_p_a,
                   _exponential_noise(candidate_logits.shape))
        ]
        window = history[-RAS_WINDOW_FRAMES:]
        repeated = (window[None, :] == first[:, None]).any(axis=-1)
        repeated &= self.is_semantic(first)
        main = mx.where(repeated, retry, first)
        c0 = mx.clip(main - self.semantic_begin, 0, self.cfg.vocab_size - 1)
        semantic = self.is_semantic(main)
        hidden = self.run_shared(hidden, c0, semantic, positions, frames)
        codes = [c0]
        head_sampler = _compiled_sampler(self.cfg.vocab_size)
        for layer, head in zip(self.residual_layers, self.codebook_heads):
            hidden = layer(hidden, codes[-1], semantic, positions)
            head_logits = head(hidden)
            codes.append(
                head_sampler(
                    head_logits, temperature_a, top_k_a, top_p_a,
                    _exponential_noise(head_logits.shape),
                )
            )
        return mx.stack([main, *codes], axis=-1)


def build_tower(weights: dict, config: ModelConfig) -> Tower:
    cfg = config.audio_decoder
    group_size, bits = 0, 0  # the tower stays bf16

    def lin(name: str) -> Linear:
        return Linear(weights[f"{name}.weight"])

    def norm(name: str) -> ZeroCenteredRMSNorm:
        return ZeroCenteredRMSNorm(weights[f"{name}.weight"])

    def make_layer(prefix: str, shared: bool) -> TowerLayer:
        embedding_name = "semantic_embedding" if shared else "codebook_embedding"
        attention = TowerAttention(
            cfg,
            wqkv=lin(f"{prefix}.transformer_layer.attention.wqkv"),
            wo=lin(f"{prefix}.transformer_layer.attention.wo"),
            q_norm=norm(f"{prefix}.transformer_layer.attention.q_norm"),
            k_norm=norm(f"{prefix}.transformer_layer.attention.k_norm"),
        )
        transformer = TowerTransformerLayer(
            attention,
            norm(f"{prefix}.transformer_layer.attention_norm"),
            norm(f"{prefix}.transformer_layer.ffn_norm"),
            TowerMLP(
                lin(f"{prefix}.transformer_layer.feed_forward.w1"),
                lin(f"{prefix}.transformer_layer.feed_forward.w2"),
                lin(f"{prefix}.transformer_layer.feed_forward.w3"),
            ),
        )
        return TowerLayer(
            shared,
            embedding_name,
            weights[f"{prefix}.{embedding_name}.weight"],
            norm(f"{prefix}.enorm"),
            norm(f"{prefix}.hnorm"),
            lin(f"{prefix}.eh_proj"),
            transformer,
            norm(f"{prefix}.final_layernorm"),
        )

    shared_layers = [
        make_layer(f"audio_decoder_mtp_shared.layers.{i}", True)
        for i in range(cfg.shared_enrichment_layers)
    ]
    residual_layers = [
        make_layer(f"audio_decoder_mtp.layers.{i}", False)
        for i in range(cfg.num_codebooks - 1)
    ]
    heads = [lin(f"audio_decoder_mtp.heads.{i}") for i in range(cfg.num_codebooks - 1)]
    return Tower(
        config,
        cfg,
        weights["audio_decoder.codebook_embeddings.weight"],
        shared_layers,
        residual_layers,
        heads,
    )
