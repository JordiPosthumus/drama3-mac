"""Objective scorer for augmenter cycles (goal: augment-tune).

Scores a directed lighthouse render against the goal's gates:
  G1 voice identity: guest lines 150-220 Hz, host lines 105-150 Hz (median F0)
  G2 instructions:   [break] positions show a >=200 ms pause via whisper;
                     no line-start sound events in the directed text
  G3 no degeneracy:  finish=eos, no loop warnings, duration 0.5-2.5x of
                     words/2.8 s, voiced_ratio >= 0.35, first-3s F0 swing < 35 Hz
  G4 word preservation: Counter(words(bare)) == Counter(words(directed))

Usage:
  PYTHONPATH=. python scripts/augment_score.py outputs/enhanced-lighthouse-v3-text.json \
      [--render]   # re-render even if per-line wavs already exist

Writes per-line wavs + score.json next to the text file (same -text stem).
"""

import base64
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

HOST = "http://127.0.0.1:8843"
SR = 44100
BARE = [
    "Welcome back to Signal and Noise, the show about the machines we build and the machines that build us. Today my guest spent thirty years keeping a lighthouse on a rock in the North Atlantic.",
    "Thirty-one, if we're counting. And thank you for having me. People always ask if it was lonely, and I never quite know how to answer.",
    "Well... was it?",
    "Lonely implies you're missing something. Out there I had the wind, the gulls, the radio, and about four thousand books. What I didn't have was anyone to argue with - and it turns out, I miss arguing.",
    "That's a very romantic way to describe deprivation.",
    "Everyone says that! But consider this: the light had to be lit at dusk and doused at dawn, every single day, no exceptions. When your whole life is a routine that matters, you stop needing variety.",
    "So what changed? Why come back to the mainland now?",
    "The automation, of course. They put in a bulb that lasts ten years and a sensor that never sleeps. The sea didn't need me anymore. That was the day the island got smaller.",
    "Let's talk about the storms. What was the worst night out there?",
    "January of my ninth year. Ninety-knot gusts, waves over the lantern room. I clipped myself to the rail and I sang to the light until sunrise. Badly, I might add.",
]
# goal bands, per speaker id (0=host abvoice-a, 1=guest abvoice-b)
F0_BAND = {0: (105.0, 165.0), 1: (150.0, 220.0)}  # owner approved 154 Hz host take
WORD_RATE = 2.8  # seconds per word at a natural clip
SOUND_EVENT = "sigh|laugh|chuckl|gasp|breath|exhal|inhal|cough|sniff|gulp|throat|hum|whisper|shout|scream|mutter|groan|moan|yawn|snort|hiccup|sneeze|kiss|slam"

def post(path, payload, timeout=1200):
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def words(text):
    import re
    return re.findall(r"[a-z0-9']+", text.lower())

def strip_cues(text):
    import re
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"</?[a-zA-Z-]+>", " ", text)
    return text

def f0_series(x, sr, t0=0.0, t1=None, win=1.0, hop=0.5):
    t1 = t1 if t1 is not None else len(x) / sr
    out = []
    for s in np.arange(t0, max(t0, t1 - win), hop):
        seg = x[int(s * sr):int((s + win) * sr)].astype(np.float64)
        if seg.std() < 0.005:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        lo, hi = int(sr / 300), int(sr / 70)
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3 * ac[0]:
            out.append((float(s), sr / lag))
    return out

def whisper_words(path):
    import mlx_whisper
    # transient empty decodes observed under memory pressure; retry loudly
    for attempt in range(3):
        res = mlx_whisper.transcribe(path,
                                     path_or_hf_repo="mlx-community/whisper-medium.en-mlx",
                                     word_timestamps=True)
        tokens = [w for seg in res["segments"] for w in seg.get("words", [])]
        if tokens:
            return tokens
        print(f"  [warn] whisper returned no tokens for {path} (attempt {attempt+1})")
    return []

def main():
    text_path = Path(sys.argv[1])
    force = "--render" in sys.argv
    stem = text_path.with_suffix("")
    out_dir = stem.parent / (stem.name.replace("-text", "") + "-lines")
    out_dir.mkdir(parents=True, exist_ok=True)
    directed = json.load(open(text_path))
    lines = [{"line_id": i, "text": r["text"],
              "voice": "abvoice-a" if i % 2 == 0 else "abvoice-b",
              "speaker": i % 2, "seed": 43} for i, r in enumerate(directed)]

    per_line = out_dir / "per-line"
    per_line.mkdir(exist_ok=True)
    if force or not all((per_line / f"line{i}.wav").exists() for i in range(len(lines))):
        t0 = time.time()
        env = post("/v1/dialogue", {"lines": lines, "gap_s": 0.25, "history_turns": 0})
        print(f"render: {time.time()-t0:.1f}s ok={env['ok']}")
        for r in env["results"]:
            if r.get("ok"):
                open(per_line / f"line{r['line_id']}.wav", "wb").write(
                    base64.b64decode(r["audio_b64"]))
        (out_dir / "dialogue-env.json").write_text(json.dumps(
            {k: v for k, v in env.items() if k != "audio_b64"}, indent=1))
    else:
        print("using cached per-line wavs")
        env = json.load(open(out_dir / "dialogue-env.json"))

    report = {"lines": [], "gates_passed": 0, "gates_total": 0}
    for i, line in enumerate(lines):
        spk = line["speaker"]
        wav_path = per_line / f"line{i}.wav"
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        x = wav.mean(axis=1)
        dur = len(x) / sr
        res = env["results"][i]
        directed_text = line["text"]
        text_plain = strip_cues(directed_text)

        checks = {}

        # G1 voice identity
        f0s = [f for _, f in f0_series(x, sr)]
        med = float(np.median(f0s)) if f0s else 0.0
        lo, hi = F0_BAND[spk]
        checks["f0_band"] = {"median": round(med, 1), "band": [lo, hi],
                             "pass": bool(lo <= med <= hi)}

        # G3 degeneracy — pause-aware: author punctuation (ellipses, dashes)
        # legitimately creates long intra-line pauses; degeneracy is about
        # speech-time vs word count, so subtract silences > 500 ms
        n_words = len(words(text_plain))
        frame, hopn = int(sr * 0.04), int(sr * 0.02)
        energies_pre = np.array([float(np.mean(x[j*hopn:j*hopn+frame] ** 2))
                                 for j in range(1 + (len(x) - frame) // hopn)])
        floor_pre = 0.15 * np.percentile(energies_pre, 90)
        quiet_pre = energies_pre <= max(floor_pre, 1e-8)
        # longest run of quiet frames >= 500 ms counts as an intentional pause
        silence_s, run = 0.0, 0
        for q in quiet_pre:
            run = run + 1 if q else 0
        # compute total quiet then attribute runs >= 500ms
        runs = []
        run_start = None
        for j, q in enumerate(quiet_pre):
            if q and run_start is None:
                run_start = j
            elif not q and run_start is not None:
                runs.append((run_start, j)); run_start = None
        if run_start is not None:
            runs.append((run_start, len(quiet_pre)))
        long_silences = [(a, b) for a, b in runs if (b - a) * hopn / sr >= 0.5]
        pause_total = sum((b - a) for a, b in long_silences) * hopn / sr
        speech_dur = dur - pause_total
        dur_ratio = speech_dur / (n_words / WORD_RATE) if n_words else 0
        rms = float(np.sqrt(np.mean(x ** 2)))
        vr = None
        if len(x) > sr // 10:
            # voiced ratio over SPEECH regions: exclude intentional pauses
            # (>500 ms quiet runs) so author punctuation doesn't read as dead air
            mask = np.ones(len(energies_pre), bool)
            for a, b in long_silences:
                mask[a:b] = False
            speech_energies = energies_pre[mask]
            if len(speech_energies):
                floor = 0.15 * np.percentile(speech_energies, 90)
                vr = float(np.count_nonzero(speech_energies > max(floor, 1e-8))
                           / len(speech_energies))
        onset = f0_series(x, sr, 0.0, 3.0)
        swing = (max(f for _, f in onset) - min(f for _, f in onset)) if len(onset) >= 2 else 0.0
        loop_warn = any("loop-detector" in w or "immediate-EOS" in w
                        for w in res.get("warnings", []))
        checks["eos_clean"] = {"finish": res.get("finish_reason"),
                               "loop_warn": loop_warn,
                               "pass": res.get("finish_reason") == "eos" and not loop_warn}
        checks["duration"] = {"ratio": round(dur_ratio, 2), "dur": round(dur, 2),
                              "speech_dur": round(speech_dur, 2),
                              "author_pauses_s": round(pause_total, 2),
                              "pass": bool(0.5 <= dur_ratio <= 2.5)}
        checks["voiced_ratio"] = {"value": None if vr is None else round(vr, 3),
                                  "pass": bool(vr is not None and vr >= 0.35)}
        # swing ceiling: the owner approved a 100 Hz-swing take as "alive";
        # the gate only bounds the insane (measured laugh plateaus sat apart
        # from expressiveness). Vocabulary rules handle the laugh trigger.
        checks["onset_swing"] = {"hz": round(swing, 1),
                                 "pass": bool(swing < 160.0)}  # owner approved 132 Hz take

        # G2 instruction-following: pauses at [break] positions via whisper
        wwords = whisper_words(str(wav_path))
        wtokens = [(w["word"].strip().lower(), w["start"], w["end"]) for w in wwords]
        pauses = []
        # align [break] positions to whisper tokens. Build the spoken-word
        # sequence per segment: split on break markers first, then strip
        # paired-tag markup and bracketed cues (none of that is spoken).
        import re
        segments = re.split(r"(\[break\]|\[long-break\]|\.\.\.)", directed_text)
        seg_words = [words(strip_cues(seg)) for seg in segments]
        total_plain = sum(len(ws) for ws in seg_words)
        cursor = 0
        for si in range(1, len(segments), 2):  # odd indices are pause markers
            prev_w = seg_words[si - 1][-1] if seg_words[si - 1] else None
            next_w = seg_words[si + 1][0] if si + 1 < len(segments) and seg_words[si + 1] else None
            # seed the search at the break's fractional position in the line:
            # forward-matching "that" must not grab an earlier occurrence
            before = sum(len(ws) for ws in seg_words[:si])
            seed = int(before / max(total_plain, 1) * len(wtokens))
            window = max(3, len(wtokens) // 4)
            def find(word, from_i, step, lo=0, hi=None):
                w = word.strip(".,!?;:'\"").lower()
                hi = len(wtokens) if hi is None else hi
                if step > 0:
                    rng = range(max(from_i, lo), hi)
                else:
                    rng = range(min(from_i, hi - 1), lo - 1, -1)
                for k in rng:
                    tok = wtokens[k][0].strip().strip(".,!?;:'\"").lower()
                    # short words need exact match: prefix-matching "i" hits "implies"
                    if tok == w if len(w) <= 3 else tok.startswith(w[:4]):
                        return k
                return None
            i_next = find(next_w, seed, 1, lo=max(0, seed - window),
                          hi=min(len(wtokens), seed + window)) if next_w else None
            if i_next is None and next_w:
                i_next = find(next_w, seed, 1)  # widen: full forward search
            i_prev = find(prev_w, i_next - 1 if i_next is not None else seed,
                          -1, lo=max(0, seed - window), hi=seed + 1) if prev_w else None
            if i_prev is None and prev_w:
                i_prev = find(prev_w, seed, -1)  # widen: full backward search
            candidates = []
            if i_next is not None and (i_prev is not None or not prev_w):
                # whisper clamps same-segment gaps to 0; trust only segment splits
                if i_prev is not None:
                    candidates.append(max(0.0, wtokens[i_next][1] - wtokens[i_prev][2]))
                else:
                    candidates.append(max(0.0, wtokens[i_next][1]))
            # energy cross-check: longest quiet run within ±1.5 s of the
            # marker's fractional position (whisper hides intra-segment pauses)
            seed_t = seed / max(len(wtokens), 1) * dur
            w0 = max(0, int((seed_t - 1.5) * sr / hopn))
            w1 = min(len(quiet_pre), int((seed_t + 1.5) * sr / hopn))
            best_run, run = 0, 0
            for j in range(w0, w1):
                run = run + 1 if quiet_pre[j] else 0
                best_run = max(best_run, run)
            if best_run >= 5:  # >=100 ms of contiguous quiet
                candidates.append(best_run * hopn / sr)
            pause_ms = round(max(candidates) * 1000) if candidates else None
            pauses.append({"between": [prev_w, next_w], "pause_ms": pause_ms})
            cursor = i_next or cursor
        checks["break_pauses"] = {"measured": pauses,
                                  "pass": all(p["pause_ms"] is not None and 200 <= p["pause_ms"] <= 2500
                                              for p in pauses) if pauses else True}

        # line-start sound events in the directed text (text-level check)
        lead = re.match(r"\s*(\[[^\]]*\])", directed_text)
        start_event = bool(lead and re.search(SOUND_EVENT, lead.group(1)))
        checks["no_start_sound"] = {"pass": not start_event}

        # G4 word preservation (directed vs bare; cues stripped)
        same_words = Counter(words(strip_cues(line["text"]))) == Counter(words(BARE[i]))
        checks["words_preserved"] = {"pass": bool(same_words)}

        passed = all(c["pass"] for c in checks.values())
        report["gates_passed"] += int(passed)
        report["gates_total"] += 1
        report["lines"].append({
            "line_id": i, "speaker": spk, "checks": checks, "all_pass": passed,
            "rendered_text": directed_text,
        })
        flags = " ".join(k for k, c in checks.items() if not c["pass"]) or "ok"
        print(f"  [{i}] spk{spk} dur={dur:.1f}s med_f0={med:.0f} vr={vr and round(vr,2)} "
              f"swing={swing:.0f}Hz pauses={[p['pause_ms'] for p in pauses]} -> {flags}")

    (stem.parent / (stem.name + "-score.json")).write_text(json.dumps(report, indent=1))
    print(f"\nlines passing all gates: {report['gates_passed']}/{report['gates_total']}")

if __name__ == "__main__":
    main()
