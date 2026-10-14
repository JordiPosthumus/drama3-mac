"""MLX ports of the FishOmni text-backbone layers.

Reference: ``fish-speech/fish_speech/models/text2semantic/fish_omni/layers.py``.
Numerics follow the reference: norms run in fp32, RoPE is partial (first
``rotary_dim`` channels, half-split / neox layout), the full-attention output is
sigmoid-gated, and the linear attention is the gated delta rule driven by
Apple's Metal kernel (``gated_delta.py``, vendored from mlx-lm, MIT).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .gated_delta import gated_delta_update, normalize_qk
from .module import Module


class Linear(Module):
    """A plain or 4/8-bit quantized linear layer (weights stored [out, in])."""

    def __init__(self, weight: mx.array, scales=None, biases=None, group_size=0, bits=0):
        self.weight = weight
        self._quantized = scales is not None
        if self._quantized:
            self.scales = scales
            self.biases = biases
            self.group_size = group_size
            self.bits = bits

    def __call__(self, x: mx.array) -> mx.array:
        if self._quantized:
            return mx.quantized_matmul(
                x,
                self.weight,
                self.scales,
                self.biases,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
            )
        return x @ self.weight.T


class ZeroCenteredRMSNorm(Module):
    """RMSNorm scaling by ``1 + weight``, computed in fp32."""

    def __init__(self, weight: mx.array):
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        out = x.astype(mx.float32)
        out = out * mx.rsqrt(mx.mean(out * out, axis=-1, keepdims=True) + 1e-6)
        return (out * (1.0 + self.weight.astype(mx.float32))).astype(x.dtype)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class GatedRMSNorm(Module):
    """Gated DeltaNet output norm: ``weight * rmsnorm(x) * silu(gate)``."""

    def __init__(self, weight: mx.array):
        self.weight = weight

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        dtype = x.dtype
        out = x.astype(mx.float32)
        out = out * mx.rsqrt(mx.mean(out * out, axis=-1, keepdims=True) + 1e-6)
        out = self.weight.astype(dtype) * out.astype(dtype)
        return (out.astype(mx.float32) * _silu(gate.astype(mx.float32))).astype(dtype)


@dataclass(kw_only=True)
class LayerContext:
    """Per-forward inputs shared by all backbone layers of one pass.

    Rows are contiguous: positions ``start .. start + length - 1``.
    """

    start: int
    length: int
    cos: mx.array  # [T, rotary_dim], full-width tiled angles
    sin: mx.array
    kv_len: int  # attended cache prefix (last position + 1)
    continue_state: bool
    use_cache: bool

    @property
    def end(self) -> int:
        return self.start + self.length


def rope_frequencies(theta: float, rotary_dim: int) -> mx.array:
    exponents = mx.arange(0, rotary_dim, 2, dtype=mx.float32)
    return 1.0 / (theta ** (exponents / rotary_dim))


def build_cos_sin(frequencies: mx.array, start: int, length: int) -> tuple[mx.array, mx.array]:
    angles = mx.arange(start, start + length, dtype=mx.float32)[:, None] * frequencies[None, :]
    angles = mx.concatenate([angles, angles], axis=-1)
    return mx.cos(angles), mx.sin(angles)


def apply_neox_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the first ``rotary_dim`` channels of x [B, T, heads, head_dim]."""
    rotary_dim = cos.shape[-1]
    rotary = x[..., :rotary_dim].astype(mx.float32)
    passthrough = x[..., rotary_dim:]
    first, second = rotary[..., : rotary_dim // 2], rotary[..., rotary_dim // 2 :]
    rotated = mx.concatenate([-second, first], axis=-1)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    out = rotary * cos + rotated * sin
    return mx.concatenate([out.astype(x.dtype), passthrough], axis=-1)


class FullAttention(Module):
    """Gated softmax attention: GQA, q/k norm, partial RoPE, sigmoid output gate."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        q_norm: ZeroCenteredRMSNorm,
        k_norm: ZeroCenteredRMSNorm,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.q_norm = q_norm
        self.k_norm = k_norm

    def allocate_cache(self, batch: int, max_seq_len: int, dtype=None) -> None:
        dtype = dtype or self.q_proj.weight.dtype
        shape = (batch, self.num_kv_heads, max_seq_len, self.head_dim)
        self.cache("k", mx.zeros(shape, dtype))
        self.cache("v", mx.zeros(shape, dtype))

    def decode_with_caches(self, x, cos, sin, k_cache, v_cache, position):
        """Single-token step over the whole cache with a position mask.

        All shapes are static (cache capacity is baked), so this compiles to one
        graph per (layer, capacity); the write uses ``put_along_axis`` instead of
        position-dependent slicing.
        """
        batch, length, _ = x.shape
        query, gate = self.q_proj(x).reshape(
            batch, length, self.num_heads, 2 * self.head_dim
        ).split(2, axis=-1)
        gate = gate.reshape(batch, length, self.num_heads * self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        )
        value = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        query = apply_neox_rope(query, cos, sin).transpose(0, 2, 1, 3)
        key = apply_neox_rope(key, cos, sin).transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        idx = position.astype(mx.int32).reshape(1, 1, 1, 1)
        k_cache = mx.put_along_axis(k_cache, idx, key, axis=2)
        v_cache = mx.put_along_axis(v_cache, idx, value, axis=2)
        cols = mx.arange(k_cache.shape[2])
        mask = mx.where(
            cols <= position,
            mx.zeros((), query.dtype),
            mx.array(-mx.inf, query.dtype),
        )[None, :]
        attended = mx.fast.scaled_dot_product_attention(
            query, k_cache, v_cache, scale=self.head_dim**-0.5, mask=mask
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(attended * mx.sigmoid(gate)), k_cache, v_cache

    def __call__(self, x: mx.array, context: LayerContext) -> mx.array:
        batch, length, _ = x.shape
        query, gate = self.q_proj(x).reshape(
            batch, length, self.num_heads, 2 * self.head_dim
        ).split(2, axis=-1)
        gate = gate.reshape(batch, length, self.num_heads * self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        )
        value = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        query = apply_neox_rope(query, context.cos, context.sin).transpose(0, 2, 1, 3)
        key = apply_neox_rope(key, context.cos, context.sin).transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        if context.use_cache:
            k_cache = self.cache("k")
            v_cache = self.cache("v")
            span = slice(context.start, context.end)
            k_cache[:batch, :, span, :] = key
            v_cache[:batch, :, span, :] = value
            key = k_cache[:batch, :, : context.kv_len, :]
            value = v_cache[:batch, :, : context.kv_len, :]
            if length == 1:
                mask = mx.zeros((1, context.kv_len), dtype=query.dtype)
            else:
                rows = mx.arange(context.start, context.end)[:, None]
                cols = mx.arange(context.kv_len)[None, :]
                mask = mx.where(
                    cols <= rows, mx.zeros((), query.dtype), mx.array(-mx.inf, query.dtype)
                )
            attended = mx.fast.scaled_dot_product_attention(
                query, key, value, scale=self.head_dim**-0.5, mask=mask
            )
        else:
            mask = "causal" if length > 1 else None
            attended = mx.fast.scaled_dot_product_attention(
                query, key, value, scale=self.head_dim**-0.5, mask=mask
            )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(attended * mx.sigmoid(gate))


class GatedDeltaNet(Module):
    """Linear attention with the gated delta rule (Qwen3-Next style)."""

    def __init__(
        self,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        in_proj_qkv: Linear,
        in_proj_z: Linear,
        in_proj_b: Linear,
        in_proj_a: Linear,
        conv1d_weight: mx.array,  # checkpoint layout [C, 1, K]
        dt_bias: mx.array,
        a_log: mx.array,
        norm: GatedRMSNorm,
        out_proj: Linear,
    ):
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        self.in_proj_qkv = in_proj_qkv
        self.in_proj_z = in_proj_z
        self.in_proj_b = in_proj_b
        self.in_proj_a = in_proj_a
        # [C, K] for the manual depthwise windowed conv below
        self.conv1d_weight = conv1d_weight.reshape(self.conv_dim, -1)
        self.dt_bias = dt_bias
        self.a_log = a_log
        self.norm = norm
        self.out_proj = out_proj

    def allocate_cache(self, batch: int, dtype=None) -> None:
        dtype = dtype or self.in_proj_qkv.weight.dtype
        self.cache(
            "conv", mx.zeros((batch, self.conv_kernel_size - 1, self.conv_dim), dtype)
        )
        self.cache(
            "state",
            mx.zeros((batch, self.num_v_heads, self.head_v_dim, self.head_k_dim), mx.float32),
        )

    def step_with_caches(self, x: mx.array, conv_state: mx.array, rec_state: mx.array):
        """Single-token step with explicit caches in/out (compile-friendly:
        no position-dependent slicing happens in the GDN decode path)."""
        batch, length, _ = x.shape
        history = self.conv_kernel_size - 1
        mixed = self.in_proj_qkv(x)
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        windows = mx.stack(
            [conv_input[:, i : i + length, :] for i in range(self.conv_kernel_size)],
            axis=-1,
        )
        conv_out = _silu(mx.sum(windows * self.conv1d_weight[None, None, :, :], axis=-1))
        qkv = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        query = qkv[0].reshape(batch, length, self.num_k_heads, self.head_k_dim)
        key = qkv[1].reshape(batch, length, self.num_k_heads, self.head_k_dim)
        value = qkv[2].reshape(batch, length, self.num_v_heads, self.head_v_dim)
        z = self.in_proj_z(x).reshape(batch, length, self.num_v_heads, self.head_v_dim)
        query, key = normalize_qk(query, key, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
        core, final_state = gated_delta_update(
            query, key, value,
            self.in_proj_a(x), self.in_proj_b(x),
            self.a_log, self.dt_bias, rec_state,
        )
        core = self.norm(
            core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        ).reshape(batch, length, self.value_dim)
        return self.out_proj(core), conv_input[:, -history:, :], final_state

    def __call__(self, x: mx.array, context: LayerContext) -> mx.array:
        batch, length, _ = x.shape
        history = self.conv_kernel_size - 1
        mixed = self.in_proj_qkv(x)  # [B, T, C]
        if context.continue_state:
            conv_state = self.cache("conv")[:batch]
        else:
            conv_state = mx.zeros((batch, history, self.conv_dim), mixed.dtype)
        conv_input = mx.concatenate([conv_state, mixed], axis=1)  # [B, T+h, C]
        if context.use_cache:
            self.cache("conv", conv_input[:, -history:, :])
        windows = mx.stack(
            [conv_input[:, i : i + length, :] for i in range(self.conv_kernel_size)],
            axis=-1,
        )  # [B, T, C, K]
        conv_out = _silu(
            mx.sum(windows * self.conv1d_weight[None, None, :, :], axis=-1)
        )
        qkv = mx.split(
            conv_out, [self.key_dim, 2 * self.key_dim], axis=-1
        )
        query = qkv[0].reshape(batch, length, self.num_k_heads, self.head_k_dim)
        key = qkv[1].reshape(batch, length, self.num_k_heads, self.head_k_dim)
        value = qkv[2].reshape(batch, length, self.num_v_heads, self.head_v_dim)
        z = self.in_proj_z(x).reshape(batch, length, self.num_v_heads, self.head_v_dim)
        query, key = normalize_qk(
            query, key, inv_scale=self.head_k_dim**-0.5, eps=1e-6
        )
        state = self.cache("state")[:batch] if context.continue_state else None
        core, final_state = gated_delta_update(
            query,
            key,
            value,
            self.in_proj_a(x),
            self.in_proj_b(x),
            self.a_log,
            self.dt_bias,
            state,
        )
        if context.use_cache:
            self.cache("state", final_state)
        core = self.norm(
            core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        ).reshape(batch, length, self.value_dim)
        return self.out_proj(core)
