"""TASK-076 lane: generate funny enemy-death vocalizations for BallerWorld.

Runs the local Drama3 engine (checkpoints/drama3-q4) over a deterministic
matrix: 4 death words x 6 theatric emotions (the Drama3 emotion tag system:
[bracketed cues] + <whisper> wrappers, syntax verified in
examples/jabberwocky/directed-chunks) x 2 seeds = 48 clips.

Gate per clip: duration 0.3-5 s, rms >= 0.02, finish_reason eos
(a `length` finish is retried once with seed+500 and flagged), no
repetition-loop warnings. The project's Whisper round-trip acceptance bar
does not apply to bare interjections (Whisper cannot transcribe "Argh"
reliably); energy + eos + warning gates are used instead and the deviation
is recorded on the card.

Output: outputs/bw-deaths/*.wav + manifest.json (one record per clip).
Run:  cd /Users/jordiposthumus/Drama3 && PYTHONPATH=. .venv/bin/python \
      outputs/bw-deaths/gen_deaths.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from drama3_mac.engine import Drama3Engine

OUT = Path(__file__).resolve().parent
CHECKPOINT = "checkpoints/drama3-q4"

# word key -> text variants (take 1, take 2)
WORDS = [
    ("argh", ["Argh!", "Argh!"]),
    ("ouch", ["Ouch!", "Ouch!"]),
    ("oops", ["Oops!", "Oops!"]),
    ("arrrgh", ["Arrrgh!", "Aaaarrgh!"]),
]

# emotion key -> (bracketed cue, wrap_tag or None)
EMOTIONS = [
    ("pain", "[sudden sharp pain]", None),
    ("indignant", "[indignant, deeply offended]", None),
    ("theater", "[over-acted theatrical anguish, collapsing dramatically]", None),
    ("dazed", "[dazed, confused, seeing stars]", None),
    ("fading", "[fading last breath]", "whisper"),
    ("cheerful", "[cheerful, oddly pleased about it]", None),
]

SEED_BASE = 76001


def build_text(word: str, cue: str, wrap: str | None) -> str:
    if wrap:
        return f"{cue} <{wrap}>{word}..</{wrap}>"
    return f"{cue} {word}"


def clip_stats(audio: np.ndarray, sr: int) -> tuple[float, float]:
    dur = len(audio) / sr
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    return dur, rms


def main() -> int:
    combos = []
    for word_key, variants in WORDS:
        for emo_key, cue, wrap in EMOTIONS:
            for take in (0, 1):
                combos.append((word_key, variants[take], emo_key, cue, wrap, take))

    engine = Drama3Engine(CHECKPOINT)
    manifest = []
    t_all = time.time()
    for idx, (word_key, word_text, emo_key, cue, wrap, take) in enumerate(combos):
        text = build_text(word_text, cue, wrap)
        seed = SEED_BASE + idx
        t0 = time.time()
        result = engine.tts(text, seed=seed)
        finish = "eos" if all(r == "eos" for r in result.finish_reasons) else "length"
        retried = False
        if finish != "eos":
            seed += 500
            retried = True
            result = engine.tts(text, seed=seed)
            finish = "eos" if all(r == "eos" for r in result.finish_reasons) else "length"
        dur, rms = clip_stats(result.audio, result.sample_rate)
        warnings = list(result.warnings)
        name = f"bw-death-{word_key}-{emo_key}-s{seed}.wav"
        path = OUT / name
        engine.save(result, str(path))
        ok = 0.3 <= dur <= 5.0 and rms >= 0.02 and finish == "eos" and not warnings
        manifest.append({
            "file": name,
            "word": word_key,
            "emotion": emo_key,
            "text": text,
            "seed": seed,
            "retried": retried,
            "duration_s": round(dur, 3),
            "rms": round(rms, 4),
            "finish": finish,
            "warnings": warnings,
            "gen_s": round(time.time() - t0, 2),
            "ok": ok,
        })
        flag = "OK " if ok else "BAD"
        print(f"[{flag}] {name} dur={dur:.2f}s rms={rms:.3f} finish={finish} "
              f"warn={len(warnings)} gen={time.time()-t0:.1f}s", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    n_ok = sum(1 for m in manifest if m["ok"])
    print(f"\nDONE {n_ok}/{len(manifest)} ok in {time.time()-t_all:.0f}s "
          f"-> {OUT/'manifest.json'}")
    return 0 if n_ok >= int(0.8 * len(manifest)) else 1


if __name__ == "__main__":
    sys.exit(main())
