"""TASK-076 lane, repair pass: re-render the BAD clips from gen_deaths.py.

Fixes applied (recorded per-entry as repair_of):
- fading/<whisper> clips: the wrapper renders bare interjections as
  near-silence or run-on silence -> rewritten as a breathy bracketed cue,
  no wrapper.
- immediate-EOS duds + sampler loop: re-render at the hot recipe
  (temperature 0.85, top_p 0.9) with a fresh seed, per the postflight hint.
- run-on-silence seeds: fresh seed, same prompt.

Replaces each BAD entry in manifest.json in place and removes the old wav
after its replacement passes the same gate (0.3-5 s, rms >= 0.02, eos, no
warnings). Leaves BAD wavs on disk if their repair also fails.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from drama3_mac.engine import Drama3Engine

OUT = Path(__file__).resolve().parent
CHECKPOINT = "checkpoints/drama3-q4"
GATE = lambda dur, rms, finish, warnings: (
    0.3 <= dur <= 5.0 and rms >= 0.02 and finish == "eos" and not warnings)

# file -> (new text, hot?)
REPAIRS = {
    "bw-death-argh-pain-s76001.wav": ("[sudden sharp pain] Argh!", True),
    "bw-death-argh-pain-s76002.wav": ("[sudden sharp pain] Argh!", True),
    "bw-death-argh-fading-s76009.wav": ("[last gasp, breathy and fading away] Argh..", False),
    "bw-death-argh-fading-s76010.wav": ("[last gasp, breathy and fading away] Argh..", True),
    "bw-death-ouch-fading-s76021.wav": ("[last gasp, breathy and fading away] Ouch..", False),
    "bw-death-ouch-fading-s76022.wav": ("[last gasp, breathy and fading away] Ouch..", False),
    "bw-death-oops-fading-s76034.wav": ("[last gasp, breathy and fading away] Oops..", False),
    "bw-death-arrrgh-indignant-s76040.wav": ("[indignant, deeply offended] Arrrgh!", False),
    "bw-death-arrrgh-dazed-s76044.wav": ("[dazed, confused, seeing stars] Arrrgh?", False),
    "bw-death-arrrgh-fading-s76046.wav": ("[last gasp, breathy and fading away] Arrrgh..", True),
    "bw-death-arrrgh-cheerful-s76048.wav": ("[cheerful, oddly pleased about it] Arrrgh!", False),
}


def clip_stats(audio: np.ndarray, sr: int) -> tuple[float, float]:
    dur = len(audio) / sr
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    return dur, rms


def main() -> int:
    manifest = json.loads((OUT / "manifest.json").read_text())
    by_file = {m["file"]: m for m in manifest}
    engine = Drama3Engine(CHECKPOINT)
    repaired, still_bad = 0, []
    for old_file, (text, hot) in REPAIRS.items():
        old = by_file[old_file]
        seed = old["seed"] + 1000
        kwargs = {"seed": seed, "temperature": 0.85, "top_p": 0.9} if hot else {"seed": seed}
        t0 = time.time()
        result = engine.tts(text, **kwargs)
        finish = "eos" if all(r == "eos" for r in result.finish_reasons) else "length"
        dur, rms = clip_stats(result.audio, result.sample_rate)
        warnings = list(result.warnings)
        ok = GATE(dur, rms, finish, warnings)
        if not ok and not hot:
            # one more try at the hot recipe before giving up
            seed += 100
            result = engine.tts(text, seed=seed, temperature=0.85, top_p=0.9)
            finish = "eos" if all(r == "eos" for r in result.finish_reasons) else "length"
            dur, rms = clip_stats(result.audio, result.sample_rate)
            warnings = list(result.warnings)
            ok = GATE(dur, rms, finish, warnings)
        new_name = old_file.replace(f"s{old['seed']}.wav", f"s{seed}.wav")
        engine.save(result, str(OUT / new_name))
        entry = {
            "file": new_name,
            "word": old["word"],
            "emotion": old["emotion"],
            "text": text,
            "seed": seed,
            "retried": True,
            "repair_of": old["seed"],
            "repair": "fading-cue-rewrite" if "fading" in old_file and "breathy" in text
                      else ("hot-retry" if hot else "seed-retry"),
            "duration_s": round(dur, 3),
            "rms": round(rms, 4),
            "finish": finish,
            "warnings": warnings,
            "gen_s": round(time.time() - t0, 2),
            "ok": ok,
        }
        if ok:
            repaired += 1
            manifest[manifest.index(old)] = entry
            (OUT / old_file).unlink()
        else:
            still_bad.append(new_name)
        flag = "OK " if ok else "BAD"
        print(f"[{flag}] {new_name} dur={dur:.2f}s rms={rms:.3f} finish={finish} "
              f"warn={len(warnings)} gen={time.time()-t0:.1f}s", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    n_ok = sum(1 for m in manifest if m["ok"])
    print(f"\nREPAIR {repaired}/{len(REPAIRS)} fixed; pool now {n_ok}/{len(manifest)} ok")
    if still_bad:
        print("STILL BAD:", ", ".join(still_bad))
    return 0 if n_ok >= int(0.9 * len(manifest)) else 1


if __name__ == "__main__":
    sys.exit(main())
