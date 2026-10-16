"""Cadence probe (Phase 3): 8 pause mechanisms, one wav, whisper-measured.

Pause length = start("that") - end("routine") via whisper word timestamps.
Target: 200-500 ms natural beat, no duration explosion.
"""

import base64
import json
import urllib.request

import mlx_whisper

HOST = "http://127.0.0.1:8843"


def post(path, payload, timeout=1800):
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


VARIANTS = [
    ("baseline", "When your whole life is a routine, that matters."),
    ("[break]", "When your whole life is a routine, [break] that matters."),
    ("[long-break]", "When your whole life is a routine, [long-break] that matters."),
    ("ellipsis", "When your whole life is a routine... that matters."),
    ("(break) V1.6", "When your whole life is a routine, (break) that matters."),
    ("<slow>", "When your whole life is a routine, <slow>that matters</slow>."),
    ("pacing brief", "[measured pace, letting the point land] When your whole life is a routine, that matters."),
    ("full stop", "When your whole life is a routine. That matters."),
]

lines = [{"text": t, "voice": "abvoice-b", "line_id": i, "seed": 777}
         for i, (name, t) in enumerate(VARIANTS)]
env = post("/v1/dialogue", {"lines": lines, "gap_s": 0.6, "history_turns": 0})
print("ok:", env["ok"], "finishes:", [r.get("finish_reason") for r in env["results"]])
print("durations:", [round(r["duration_s"], 1) for r in env["results"]])
open("outputs/probe-pause.wav", "wb").write(base64.b64decode(env["audio_b64"]))

# whisper each line separately from the cached envelopes for clean timestamps
GAP = 0.6
print(f"\n{'variant':<15} {'dur':>5} {'pause(ms)':>9}")
base_pause = None
for (name, text), r in zip(VARIANTS, env["results"]):
    raw = base64.b64decode(r["audio_b64"])
    open("/tmp/probe-line.wav", "wb").write(raw)
    res = mlx_whisper.transcribe("/tmp/probe-line.wav",
                                 path_or_hf_repo="mlx-community/whisper-medium.en-mlx",
                                 word_timestamps=True)
    words = [w for seg in res["segments"] for w in seg.get("words", [])]
    end_routine = start_that = None
    for w in words:
        word = "".join(c for c in w["word"].lower() if c.isalnum())
        if word.startswith("routine") and end_routine is None:
            end_routine = w["end"]
        if word.startswith("that") and end_routine is not None and start_that is None:
            start_that = w["start"]
    pause_ms = (start_that - end_routine) * 1000 if (start_that and end_routine) else None
    marker = ""
    if pause_ms is not None:
        if base_pause is None:
            base_pause = pause_ms
        marker = "  <== TARGET" if 200 <= pause_ms <= 500 else ""
    print(f"{name:<15} {r['duration_s']:>5.1f} "
          f"{f'{pause_ms:.0f}' if pause_ms is not None else 'n/a':>9}{marker}")
print("saved outputs/probe-pause.wav")
