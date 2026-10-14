"""Probe: do [long pause] tags push the RAS sampler into repetition loops?

Generates the same script with and without pause tags in a single pass, then
analyzes the main-token history for repetition pressure (the RAS rule re-rolls
a semantic pick that already appeared in the last 16 frames).

Run: PYTHONPATH=. python scripts/probe_ras_pause.py
"""

from __future__ import annotations

import time

import numpy as np

from drama3_mac.engine import Drama3Engine

WITH_PAUSES = (
    "[calm] Listen carefully. [long pause] The cellar door creaks open. "
    "[long pause] Something breathes in the dark. [long pause] "
    "Then everything stops. [long pause] And the tape keeps rolling."
)
WITHOUT = (
    "[calm] Listen carefully. The cellar door creaks open. "
    "Something breathes in the dark. Then everything stops. "
    "And the tape keeps rolling."
)

SEMANTIC_BEGIN = 248099
SEMANTIC_END = 252194


def analyze(name: str, frames: np.ndarray, elapsed: float) -> None:
    mains = frames[:, 0]
    semantic = (mains >= SEMANTIC_BEGIN) & (mains <= SEMANTIC_END)
    codes = mains[semantic] - SEMANTIC_BEGIN
    # run lengths of identical main tokens
    runs, current, length = [], None, 0
    for m in mains:
        if m == current:
            length += 1
        else:
            if current is not None:
                runs.append(length)
            current, length = m, 1
    runs.append(length)
    # fraction of semantic picks repeating within a 16-frame window (RAS pressure)
    repeats = 0
    for i in range(len(mains)):
        window = mains[max(0, i - 16) : i]
        repeats += bool(len(window) and (window == mains[i]).any() and semantic[i])
    print(
        f"{name}: {len(frames)} frames ({len(frames) / 21.533:.1f}s), {elapsed:.1f}s\n"
        f"  semantic picks: {semantic.sum()}, in-window repeats: {repeats} "
        f"({100 * repeats / max(1, semantic.sum()):.0f}%)\n"
        f"  longest identical-main run: {max(runs)} frames\n"
        f"  unique semantic codes: {len(set(codes.tolist()))}"
    )


def main() -> None:
    engine = Drama3Engine("checkpoints/drama3-q4")
    for name, text in (("with-pauses", WITH_PAUSES), ("without-pauses", WITHOUT)):
        t0 = time.perf_counter()
        result = engine.tts(text, single_chunk=True, chunk_gap=0.0)
        analyze(
            name,
            result.generation.frames,
            time.perf_counter() - t0,
        )
        print(f"  finish={result.generation.finish_reason}")


if __name__ == "__main__":
    main()
