#!/usr/bin/env python
"""A/B sampling-config comparison on one auto-chunked render of the Jabberwocky script.

Configs:
  fish      temp 0.7, top_p 0.7, top_k 30, no penalty   (official SDK defaults;
            repetition_penalty is carried but IGNORED by Fish for Drama 3)
  ours      temp 0.8, top_p 0.9, top_k 30, no penalty   (port defaults)
  ours+pen  temp 0.8, top_p 0.9, top_k 30, penalty 1.2  (opt-in loop guard)

Metrics per render: finish status, audio seconds, RTF, and the in-window
semantic repeat rate (fraction of generated main tokens equal to any of the
previous 16 -- the RAS window; the loop proxy).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drama3_mac.engine import Drama3Engine  # noqa: E402

RAS_WINDOW = 16
REF_WAV = "assets/ref-en0.wav"
REF_TEXT = "We asked over twenty different people, and they all said it was his."
SCRIPT = Path("examples/jabberwocky/script.txt").read_text()

CONFIGS = [
    ("fish", dict(temperature=0.7, top_p=0.7, repetition_penalty=1.0)),
    ("ours", dict(temperature=0.8, top_p=0.9, repetition_penalty=1.0)),
    ("ours+pen1.2", dict(temperature=0.8, top_p=0.9, repetition_penalty=1.2)),
]


def repeat_rate(mains: list[np.ndarray]) -> float:
    """Fraction of generated main tokens repeating within the previous RAS_WINDOW."""
    total = repeats = 0
    for seq in mains:
        for i in range(1, len(seq)):
            window = seq[max(0, i - RAS_WINDOW) : i]
            total += 1
            repeats += int((window == seq[i]).any())
    return repeats / total if total else 0.0


def main() -> None:
    engine = Drama3Engine("checkpoints/drama3-q4")
    print(f"script: {len(SCRIPT)} chars\n")
    for name, kwargs in CONFIGS:
        started = time.perf_counter()
        result = engine.tts(
            SCRIPT,
            reference_audio=[REF_WAV],
            reference_text=[REF_TEXT],
            **kwargs,
        )
        wall = time.perf_counter() - started
        out = Path(f"outputs/ab-{name.replace('+', '-')}.wav")
        engine.save(result, out)
        rate = repeat_rate(list(result.chunk_main))
        print(
            f"{name:14s} {out}  {len(result.audio)/result.sample_rate:6.1f}s audio  "
            f"rtf {result.real_time_factor:.2f}  wall {wall:.0f}s  "
            f"chunks {result.chunks}  repeat-rate {rate:.1%}"
        )


if __name__ == "__main__":
    main()
