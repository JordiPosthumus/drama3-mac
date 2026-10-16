"""Verify the two dialogue renders: per-line whisper coverage + per-line F0.

Aligns each script line to transcript word spans in order, then measures
median F0 (autocorrelation) per line for both renders.
"""

import wave

import mlx_whisper
import numpy as np

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


def norm_words(text):
    return ["".join(c for c in w.lower() if c.isalnum()) for w in text.split()]


def load(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    return audio, sr


def f0_median(audio, sr, lo=60.0, hi=350.0):
    """Median F0 via autocorrelation over 40 ms voiced windows."""
    if len(audio) < sr // 10:
        return None
    frame, hop = int(0.04 * sr), int(0.02 * sr)
    lag_min, lag_max = int(sr / hi), int(sr / lo)
    values = []
    for start in range(0, len(audio) - frame, hop):
        x = audio[start : start + frame]
        if np.sqrt(np.mean(x * x)) < 0.01:
            continue
        x = x - x.mean()
        ac = np.correlate(x, x, "full")[frame - 1 :]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        window = ac[lag_min : lag_max]
        peak = np.argmax(window) + lag_min
        if ac[peak] > 0.5:
            values.append(sr / peak)
    return float(np.median(values)) if values else None


def analyze(path, label):
    print(f"\n== {label} ({path}) ==")
    audio, sr = load(path)
    result = mlx_whisper.transcribe(path, path_or_hf_repo="mlx-community/whisper-medium.en-mlx",
                                    word_timestamps=True)
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append((norm_words(w["word"])[0], w["start"], w["end"]))
    transcript = " ".join(w for w, _, _ in words)
    print(f"whisper: {transcript[:160]}...")

    cursor, rows = 0, []
    for who, text in LINES:
        target = norm_words(text)
        remaining = list(target)
        matched_idx = []
        i = cursor
        while i < len(words) and remaining:
            if words[i][0] == remaining[0]:
                matched_idx.append(i)
                remaining.pop(0)
            i += 1
        if len(matched_idx) == 0:
            rows.append((who, 0.0, None, "MISSING"))
            continue
        cover = len(matched_idx) / len(target)
        t0, t1 = words[matched_idx[0]][1], words[matched_idx[-1]][2]
        seg = audio[int(t0 * sr) : int(t1 * sr) + 1]
        rows.append((who, cover, f0_median(seg, sr), f"{t0:.1f}-{t1:.1f}s"))
        cursor = matched_idx[-1] + 1
    print(f"{'spk':>3} {'cover':>5} {'f0':>6}  span")
    for who, cover, f0, span in rows:
        print(f"{who:>3} {cover:>5.0%} {f0 if f0 else 0:>6.0f}  {span}")
    a_f0 = [f for (who, c, f, s) in rows if who == 'A' and f]
    b_f0 = [f for (who, c, f, s) in rows if who == 'B' and f]
    if a_f0 and b_f0:
        print(f"F0 A: {min(a_f0):.0f}-{max(a_f0):.0f} Hz   F0 B: {min(b_f0):.0f}-{max(b_f0):.0f} Hz")
    missing = sum(1 for r in rows if r[1] == 0)
    weak = sum(1 for r in rows if 0 < r[1] < 0.8)
    print(f"coverage: {10 - missing}/10 lines found, {weak} lines below 80% words")


if __name__ == "__main__":
    analyze("outputs/dialogue-tags.wav", "TAGS (single render, 6 chunks)")
    analyze("outputs/dialogue-lines.wav", "LINES (per-line batch)")
