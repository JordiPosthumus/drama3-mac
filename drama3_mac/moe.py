"""Sparse MoE block: top-8 routed experts + sigmoid-gated shared expert.

Experts are stored stacked per layer (``w13`` [E, 2I, H] gate-rows-first,
``w2`` [E, H, I]), either bf16 or MLX-quantized.  Decode uses one gathered
batched matmul over every (token, expert) pair; prefill loops over the experts
a chunk hits, each on its own rows.
"""

from __future__ import annotations

import mlx.core as mx

from .layers import Linear, _silu
from .module import Module

GATHERED_EXPERT_MAX_ROWS = 64


class Experts(Module):
    def __init__(
        self,
        w13: mx.array,
        w2: mx.array,
        scales13=None,
        biases13=None,
        scales2=None,
        biases2=None,
        group_size: int = 0,
        bits: int = 0,
    ):
        self.w13 = w13
        self.w2 = w2
        self._quantized = scales13 is not None
        if self._quantized:
            self.scales13 = scales13
            self.biases13 = biases13
            self.scales2 = scales2
            self.biases2 = biases2
            self.group_size = group_size
            self.bits = bits

    def _gate_up(self, x: mx.array, index: mx.array | int) -> mx.array:
        """x [..., H] with the expert selected per leading row -> [..., 2I]."""
        if self._quantized:
            return mx.quantized_matmul(
                x,
                self.w13[index],
                self.scales13[index],
                self.biases13[index],
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
            )
        return x @ self.w13[index].swapaxes(-1, -2)

    def _down(self, x: mx.array, index: mx.array | int) -> mx.array:
        if self._quantized:
            return mx.quantized_matmul(
                x,
                self.w2[index],
                self.scales2[index],
                self.biases2[index],
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
            )
        return x @ self.w2[index].swapaxes(-1, -2)

    def __call__(
        self, x: mx.array, topk_ids: mx.array, topk_weights: mx.array
    ) -> mx.array:
        tokens, top_k = topk_ids.shape
        if tokens * top_k <= GATHERED_EXPERT_MAX_ROWS:
            return self._gathered(x, topk_ids, topk_weights)
        return self._grouped(x, topk_ids, topk_weights)

    def _gathered(self, x, topk_ids, topk_weights):
        tokens, top_k = topk_ids.shape
        ids = topk_ids.reshape(-1)
        rows = mx.repeat(x, top_k, axis=0).reshape(-1, 1, x.shape[-1])
        gate_up = self._gate_up(rows, ids)  # [M, 1, 2I]
        gate, up = mx.split(gate_up, 2, axis=-1)
        hidden = _silu(gate) * up
        out = self._down(hidden, ids)[:, 0, :]  # [M, H]
        out = out.astype(mx.float32) * topk_weights.reshape(-1, 1).astype(mx.float32)
        return out.reshape(tokens, top_k, -1).sum(axis=1).astype(x.dtype)

    def _grouped(self, x, topk_ids, topk_weights):
        tokens, top_k = topk_ids.shape
        flat = topk_ids.reshape(-1)
        order = mx.argsort(flat)
        sorted_ids = flat[order]
        token_rows = order // top_k
        weights = topk_weights.reshape(-1)[order]
        mx.eval(sorted_ids)
        counts = np_bincount(sorted_ids, self.w13.shape[0])
        out = mx.zeros((tokens, x.shape[-1]), mx.float32)
        start = 0
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            rows = token_rows[start : start + count]
            gate_up = self._gate_up(x[rows], expert)
            gate, up = mx.split(gate_up, 2, axis=-1)
            expert_out = self._down(_silu(gate) * up, expert)
            contrib = expert_out.astype(mx.float32) * weights[start : start + count, None].astype(mx.float32)
            out[rows] = out[rows] + contrib
            start += count
        return out.astype(x.dtype)


def np_bincount(sorted_ids: mx.array, num_classes: int) -> list[int]:
    import numpy as np

    mx.eval(sorted_ids)
    ids = np.array(sorted_ids)
    counts = np.bincount(ids, minlength=num_classes)
    return [int(c) for c in counts]


class SparseMoeBlock(Module):
    """Top-k routed experts plus a sigmoid-gated shared expert."""

    def __init__(
        self,
        gate: Linear,
        experts: Experts,
        shared_gate_proj: Linear,
        shared_up_proj: Linear,
        shared_down_proj: Linear,
        shared_expert_gate: Linear,
        top_k: int,
        norm_topk_prob: bool = True,
    ):
        self.gate = gate
        self.experts = experts
        self.shared_gate_proj = shared_gate_proj
        self.shared_up_proj = shared_up_proj
        self.shared_down_proj = shared_down_proj
        self.shared_expert_gate = shared_expert_gate
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self._decode_fn = None

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 2 and x.shape[0] == 1:
            # decode shape: reuse one compiled graph (weights are constants)
            if self._decode_fn is None:
                self._decode_fn = mx.compile(self._call_flat)
            return self._decode_fn(x)
        return self._call_flat(x)

    def _call_flat(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, x.shape[-1])
        probabilities = mx.softmax(self.gate(flat).astype(mx.float32), axis=-1)
        top_k = self.top_k
        topk_ids = mx.argpartition(-probabilities, kth=top_k, axis=-1)[..., :top_k]
        topk_weights = mx.take_along_axis(probabilities, topk_ids, axis=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
        out = self.experts(flat, topk_ids.astype(mx.int32), topk_weights.astype(flat.dtype))
        shared = self.shared_down_proj(
            _silu(self.shared_gate_proj(flat)) * self.shared_up_proj(flat)
        )
        gate = mx.sigmoid(self.shared_expert_gate(flat))
        return (out + gate * shared).reshape(x.shape)
