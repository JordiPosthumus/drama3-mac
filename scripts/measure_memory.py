"""Measure real RAM usage per converted checkpoint (for the README table).

Loads each checkpoint, synthesizes one sentence, and reports process RSS and
MLX peak memory.  Weights are memory-mapped and fully touched during
generation, so RSS after generation is the honest "needs this much RAM"
number.

Run: PYTHONPATH=. python scripts/measure_memory.py [checkpoint ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx


def rss_gb() -> float:
    """Peak resident set size in GB (ru_maxrss is bytes on macOS)."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def main() -> int:
    checkpoints = sys.argv[1:] or [
        "checkpoints/drama3-q4",
        "checkpoints/drama3-q8",
        "checkpoints/drama3-bf16",
    ]
    from drama3_mac.engine import Drama3Engine

    print("| checkpoint | on disk | peak RSS |")
    print("|---|---|---|")
    for ckpt in checkpoints:
        disk = sum(f.stat().st_size for f in Path(ckpt).rglob("*") if f.is_file())
        t0 = time.perf_counter()
        engine = Drama3Engine(ckpt)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = engine.tts("Measuring memory usage on this machine.")
        gen_s = time.perf_counter() - t0
        peak = rss_gb()
        print(
            f"| {ckpt} | {disk / 1e9:.1f} GB | {peak:.1f} GB | "
            f"load {load_s:.0f}s, gen {gen_s:.0f}s, rtf {result.real_time_factor:.2f}",
            flush=True,
        )
        del engine
        mx.clear_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
