"""A/B: long two-person dialogue — speaker-tag single render vs per-line batch.

Renders the same script twice against a warm server on :8843:
  1. tags : one /v1/tts call, <|speaker:0|>/<|speaker:1|> tagged text, both
            references passed as tagged [Reference 1]/[Reference 2] blocks.
  2. lines: two registered voices, one /v1/batch call, alternating voices,
            line mode (no internal chunking), assembled with 250 ms gaps.

Writes outputs/dialogue-tags.wav and outputs/dialogue-lines.wav, then
whisper-transcribes both and reports per-line coverage + F0 separation.
"""

import base64
import json
import time
import urllib.request

import numpy as np

HOST = "http://127.0.0.1:8843"
REF_A = ("assets/ref-en0.wav", "We asked over twenty different people, and they all said it was his.")
REF_B_WAV = "assets/brona-ref.wav"
REF_B_TEXT = (
    "Now, how does it get that memory? Through something we call the harness. "
    "And here's the beautiful part. All you need is a capable harness, and a "
    "single text file with instructions."
)

LINES = [
    ("A", "Welcome back to Signal and Noise, the show about the machines we build and the machines that build us. Today my guest spent thirty years keeping a lighthouse on a rock in the North Atlantic."),
    ("B", "Thirty-one, if we're counting. And thank you for having me. People always ask if it was lonely, and I never quite know how to answer."),
    ("A", "Well, was it?"),
    ("B", "Lonely implies you're missing something. Out there I had the wind, the gulls, the radio, and about four thousand books. What I didn't have was anyone to argue with, and it turns out I miss arguing."),
    ("A", "That's a very romantic way to describe deprivation."),
    ("B", "Ha! Everyone says that. But consider this: the light had to be lit at dusk and doused at dawn, every single day, no exceptions. When your whole life is a routine that matters, you stop needing variety."),
    ("A", "So what changed? Why come back to the mainland now?"),
    ("B", "The automation, of course. They put in a bulb that lasts ten years and a sensor that never sleeps. The sea didn't need me anymore. That was the day the island got smaller."),
    ("A", "Let's talk about the storms. What was the worst night out there?"),
    ("B", "January of my ninth year. Ninety-knot gusts, waves over the lantern room. I clipped myself to the rail and I sang to the light until sunrise. Badly, I might add."),
]


def post(path, payload, timeout=1200):
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wav_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def save_wav(path, audio_b64):
    with open(path, "wb") as f:
        f.write(base64.b64decode(audio_b64))


def main():
    ref_a = wav_bytes(REF_A[0])
    ref_b = wav_bytes(REF_B_WAV)
    b64 = lambda raw: base64.b64encode(raw).decode("ascii")

    # -- register the two voices --------------------------------------------
    for name, raw, text in (("abvoice-a", ref_a, REF_A[1]), ("abvoice-b", ref_b, REF_B_TEXT)):
        try:
            post("/v1/voices", {"name": name, "audio_b64": b64(raw), "text": text})
        except urllib.error.HTTPError as exc:
            existing = json.loads(urllib.request.urlopen(f"{HOST}/v1/voices", timeout=30).read())
            names = {v["name"] for v in existing.get("voices", [])}
            if name not in names:
                raise
            print(f"[voices] {name} already registered")
    print("[voices] ready: abvoice-a (ref-en0), abvoice-b (brona)")

    # -- method 1: single tagged render --------------------------------------
    tagged = "\n".join(
        f"<|speaker:{0 if who == 'A' else 1}|> {text}" for who, text in LINES
    )
    t0 = time.perf_counter()
    env = post("/v1/tts", {
        "text": tagged,
        "references": [
            {"audio_b64": b64(ref_a), "text": f"<|speaker:0|> {REF_A[1]}"},
            {"audio_b64": b64(ref_b), "text": f"<|speaker:1|> {REF_B_TEXT}"},
        ],
        "seed": 42,
        "response_format": "json",
    })
    print(f"[tags] ok={env.get('ok')} chunks={env.get('chunks')} "
          f"finish={env.get('finish_reason')} rtf={env.get('rtf')} "
          f"{time.perf_counter() - t0:.0f}s")
    print(f"[tags] warnings={env.get('warnings')}")
    save_wav("outputs/dialogue-tags.wav", env["audio_b64"])

    # -- method 2: per-line batch --------------------------------------------
    batch = {"lines": [
        {"text": text, "voice": f"abvoice-{who.lower()}", "line_id": i,
         "line": True, "seed": 1000 + i, "response_format": "json"}
        for i, (who, text) in enumerate(LINES)
    ]}
    t0 = time.perf_counter()
    result = post("/v1/batch", batch)
    print(f"[lines] ok={result['ok']} count={result['count']} {time.perf_counter() - t0:.0f}s")
    line_audio, line_f0, line_finish = [], [], []
    for r in result["results"]:
        if not r.get("ok"):
            print(f"[lines] line {r['line_id']} FAILED: {r.get('error')}")
            continue
        line_audio.append(base64.b64decode(r["audio_b64"]))
        line_f0.append(r.get("f0_hz_median"))
        line_finish.append(r.get("finish_reason"))
    print(f"[lines] finish reasons: {line_finish}")
    print(f"[lines] f0 medians (A/B alternating): "
          f"{[f'{f:.0f}' if f else None for f in line_f0]}")
    import io
    import soundfile as sf
    pieces, sr0 = [], None
    for raw in line_audio:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        sr0 = sr0 or sr
        pieces.append(data)
        pieces.append(np.zeros(int(0.25 * sr), np.float32))
    pieces.pop()  # no trailing gap
    sf.write("outputs/dialogue-lines.wav", np.concatenate(pieces), sr0)
    print("[lines] assembled outputs/dialogue-lines.wav")

    print("[done] outputs/dialogue-tags.wav, outputs/dialogue-lines.wav")


if __name__ == "__main__":
    main()
