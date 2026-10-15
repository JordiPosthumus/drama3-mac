"""Final fading-slot repair: comic dizzy-faint delivery (audible by design)."""
import json, sys, time
from pathlib import Path
import numpy as np
from drama3_mac.engine import Drama3Engine

OUT = Path(__file__).resolve().parent
FIXES = [
    ("bw-death-argh-fading", "[wobbly, dizzy, about to faint] Argh.. whoa.."),
    ("bw-death-ouch-fading", "[wobbly, dizzy, about to faint] Ouch.. ohh.."),
    ("bw-death-oops-fading", "[wobbly, dizzy, about to faint] Oops.. oh dear.."),
]
SEEDS = [78101, 78102, 78103]

engine = Drama3Engine("checkpoints/drama3-q4")
manifest = json.loads((OUT / "manifest.json").read_text())
for (base, text), seed in zip(FIXES, SEEDS):
    old = next((m for m in manifest if m["file"].startswith(base) and not m["ok"]), None)
    t0 = time.time()
    r = engine.tts(text, seed=seed)
    finish = "eos" if all(x == "eos" for x in r.finish_reasons) else "length"
    dur = len(r.audio) / r.sample_rate
    rms = float(np.sqrt(np.mean(np.square(r.audio)))) if len(r.audio) else 0.0
    warns = list(r.warnings)
    ok = 0.3 <= dur <= 5.0 and rms >= 0.02 and finish == "eos" and not warns
    name = f"{base}-s{seed}.wav"
    engine.save(r, str(OUT / name))
    entry = {"file": name, "word": base.split("-")[2], "emotion": "fading", "text": text,
             "seed": seed, "retried": True, "repair_of": old["seed"] if old else None,
             "repair": "dizzy-faint-rewrite", "duration_s": round(dur, 3), "rms": round(rms, 4),
             "finish": finish, "warnings": warns, "gen_s": round(time.time()-t0, 2), "ok": ok}
    print(f"[{'OK ' if ok else 'BAD'}] {name} dur={dur:.2f}s rms={rms:.3f} finish={finish} gen={time.time()-t0:.1f}s", flush=True)
    if ok and old:
        manifest[manifest.index(old)] = entry
        (OUT / old["file"]).unlink(missing_ok=True)
    elif ok:
        manifest.append(entry)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
n_ok = sum(1 for m in manifest if m["ok"])
print(f"POOL {n_ok}/{len(manifest)} ok")
sys.exit(0 if n_ok >= int(0.9 * len(manifest)) else 1)
