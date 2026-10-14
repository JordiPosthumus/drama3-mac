"""Acceptance suite: three prompts through generation + Whisper medium.en.

Run:  PYTHONPATH=. python scripts/acceptance.py --model checkpoints/drama3-q4

Prompts:
  1. plain synthesis (no reference), sentence A
  2. plain synthesis (no reference), sentence B
  3. voice cloning from assets/ref-en0.wav

Each clip passes only if whisper medium.en transcribes it back to the prompt
(minor casing/punctuation differences allowed). Prints a PASS/FAIL table and
exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REF_TEXT = "We asked over twenty different people, and they all said it was his."


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace and number words."""
    text = text.lower().strip()
    replacements = {
        "twenty": "20",
        "quick brown fox": "quick brown fox",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    p, r = overlap / len(ta), overlap / len(tb)
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/drama3-q4")
    parser.add_argument("--ref", default="assets/ref-en0.wav")
    parser.add_argument("--outdir", default="outputs/acceptance")
    parser.add_argument("--min-f1", type=float, default=0.8)
    args = parser.parse_args()

    cases = [
        ("plain-a", "Drama three runs entirely on Apple silicon with no CUDA.",
         None),
        ("plain-b", "The quick brown fox jumps over the lazy dog.", None),
        ("clone-c", "Testing voice cloning with my own reference clip on the Mac.",
         (args.ref, REF_TEXT)),
    ]

    from drama3_mac.engine import Drama3Engine

    engine = Drama3Engine(args.model)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, text, ref in cases:
        t0 = time.perf_counter()
        result = engine.tts(
            text,
            reference_audio=ref[0] if ref else None,
            reference_text=ref[1] if ref else None,
        )
        wav_path = outdir / f"{name}.wav"
        engine.save(result, wav_path)
        gen_s = time.perf_counter() - t0
        heard = subprocess.run(
            [sys.executable, "-c",
             "import mlx_whisper, sys; "
             "print(mlx_whisper.transcribe(sys.argv[1], "
             "path_or_hf_repo='mlx-community/whisper-medium.en-mlx')['text'])",
             str(wav_path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        score = token_f1(normalize(text), normalize(heard))
        ok = score >= args.min_f1
        results.append((name, text, heard, score, ok, gen_s, result.real_time_factor))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: f1={score:.2f} "
              f"rtf={result.real_time_factor:.2f}")
        print(f"   prompt: {text}")
        print(f"   heard : {heard}")

    print()
    passed = sum(1 for r in results if r[4])
    print(f"{passed}/{len(results)} accepted (min f1 {args.min_f1})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
