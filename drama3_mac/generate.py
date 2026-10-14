"""Frame generation loop (MLX port of ``fish_omni.generate``).

Prefill runs in chunks; each decode step samples one frame (main token plus ten
codes) through the tower, feeds it back through the backbone, and stops on the
eos main token.  There are no CUDA-graph equivalents here — MLX builds the step
graph lazily and evaluates once per step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .backbone import Backbone
from .config import ModelConfig
from .tower import Tower

PREFILL_CHUNK_SIZE = 128
REPETITION_WINDOW = 64  # main tokens considered by the opt-in repetition penalty
DEFAULT_MAX_NEW_TOKENS = 2048


@dataclass
class GenerationOutput:
    frames: np.ndarray  # [T, 1 + num_codebooks]
    finish_reason: str  # "eos" | "length"
    elapsed: float

    @property
    def codes(self) -> np.ndarray:
        """[num_codebooks, T] for the codec (main token column dropped)."""
        return self.frames[:, 1:].T.copy()

    @property
    def frames_per_second(self) -> float:
        return len(self.frames) / self.elapsed if self.elapsed else 0.0


def generate(
    backbone: Backbone,
    tower: Tower,
    config: ModelConfig,
    prompt: np.ndarray,
    *,
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    top_k: int = 30,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
    cache_capacity: int | None = None,
) -> GenerationOutput:
    """Sample frames after the prompt rows.

    ``cache_capacity`` pre-sizes the KV/recurrent caches (default: exactly
    ``prompt_rows + max_new_tokens``).  Long-text callers pass one fixed
    capacity for every chunk so the compiled decode graphs, which are keyed on
    cache shapes, are compiled once and reused across chunks.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1; got {top_k}")
    prompt_rows = len(prompt)
    capacity = prompt_rows + max_new_tokens
    if cache_capacity is not None:
        if cache_capacity < capacity:
            raise ValueError("cache_capacity is smaller than prompt + max_new_tokens")
        capacity = cache_capacity
    if capacity > config.text.max_position_embeddings:
        raise ValueError("prompt + max_new_tokens exceeds max_position_embeddings")
    backbone.setup_caches(1, capacity)
    tower.setup_caches(capacity)

    frames = mx.zeros((capacity, prompt.shape[1]), mx.int32)
    frames[:prompt_rows] = mx.array(prompt)

    started = time.perf_counter()
    hidden = None
    for start in range(0, prompt_rows, prefill_chunk_size):
        end = min(start + prefill_chunk_size, prompt_rows)
        chunk = frames[start:end]
        embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0])[None])
        hidden = backbone(
            embeddings,
            start=start,
            kv_len=end,
            continue_state=start > 0,
            use_cache=True,
        )
        paired_end = min(end, prompt_rows - 1)
        if paired_end > start:
            tower.prefill(
                hidden[0][: paired_end - start],
                mx.arange(start, paired_end),
                frames[start:paired_end],
                frames[start + 1 : paired_end + 1],
            )
        mx.eval(hidden)
    prefill_elapsed = time.perf_counter() - started
    hidden = hidden[0][-1:]

    produced = 0
    finish_reason = "length"
    decode_started = time.perf_counter()
    # Fish carries a repetition_penalty field but ignores it for Drama 3 (RAS
    # replaces it), so 1.0 = faithful default.  The opt-in penalty is a
    # CTRL-style push against recently drawn semantic codes (see tower.py).
    penalty_window = (
        np.full((REPETITION_WINDOW,), -1, np.int32)
        if repetition_penalty != 1.0
        else None
    )
    for step in range(max_new_tokens):
        position = prompt_rows - 1 + step
        logits = backbone.compute_logits(hidden)
        sampled = tower.decode_step(
            logits,
            hidden,
            mx.array([position]),
            frames[position : position + 1],
            frames[: position + 1, 0],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            penalty_window=(
                mx.array(penalty_window, mx.int32) if penalty_window is not None else None
            ),
        )
        embeddings = tower.embed_backbone(sampled, backbone.embed(sampled[:, 0])[None])
        hidden = backbone(
            embeddings,
            start=position + 1,
            kv_len=position + 2,
            continue_state=True,
            use_cache=True,
            position=mx.array(position + 1, mx.int32),
        )[0]
        mx.eval(sampled, hidden)
        main = int(sampled[0, 0])
        if main == config.eos_token_id:
            finish_reason = "eos"
            break
        frames[position + 1] = sampled[0]
        produced += 1
        if penalty_window is not None:
            penalty_window[:-1] = penalty_window[1:]
            penalty_window[-1] = main

    elapsed = time.perf_counter() - decode_started
    out = np.array(frames[prompt_rows : prompt_rows + produced], copy=True)
    return GenerationOutput(
        frames=out,
        finish_reason=finish_reason,
        elapsed=prefill_elapsed + elapsed,
    )
