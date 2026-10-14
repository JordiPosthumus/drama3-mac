"""First end-to-end smoke run: clone the model-card reference voice."""

import sys
import time

import numpy as np

from drama3_mac.engine import Drama3Engine

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/drama3-q4"
REF_WAV = "assets/ref-en0.wav"
REF_TEXT = "We asked over twenty different people, and they all said it was his."
TEXT = "Hello from the Mac! This is Drama three, running entirely on Apple silicon."

t0 = time.perf_counter()
engine = Drama3Engine(MODEL_DIR)
print(f"[load] {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
codes = engine.encode_reference(REF_WAV)
print(f"[encode] codes {codes.shape} in {time.perf_counter() - t0:.2f}s")

t0 = time.perf_counter()
result = engine.tts(TEXT, reference_codes=codes, reference_text=REF_TEXT)
gen = result.generation
print(
    f"[generate] {len(gen.frames)} frames ({len(gen.frames) / 21.53:.1f}s audio) "
    f"in {gen.elapsed:.1f}s -> {gen.frames_per_second:.1f} frames/s, finish={gen.finish_reason}"
)
print(f"[rtf] {result.real_time_factor:.2f} (<=1.0 means faster than real-time)")

path = engine.save(result, "outputs/e2e-first.wav")
print(f"[saved] {path}")
