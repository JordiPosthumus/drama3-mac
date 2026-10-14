# AGENTS.md — driving drama3-mac from a coding agent

Everything an autonomous agent needs to synthesize speech on this machine
without guessing. All commands assume the repo root `/Users/jordiposthumus/Drama3`
and the project venv (`.venv`, Python 3.12).

## State of the machine (verified 2026-10-14)

- Model downloaded: `model/` (HF `fishaudio/drama-3`, 40 GB, includes
  `model/codec.pth`).
- Converted checkpoints: `checkpoints/drama3-q4` (default), `drama3-q8`,
  `drama3-bf16`. All experts-only quantization (projections bf16 — see
  README warning; never re-convert with `--quantize-projections`).
- Reference clip for cloning tests: `assets/ref-en0.wav` (3.9 s) with
  transcript "We asked over twenty different people, and they all said it
  was his." — do not commit it or any generated audio (`outputs/`,
  `assets/`, `*.wav` are git-ignored).

## Generate speech (CLI)

```bash
cd /Users/jordiposthumus/Drama3
PYTHONPATH=. .venv/bin/python -m drama3_mac.cli synthesize "Text here." \
    -o outputs/out.wav \
    --ref assets/ref-en0.wav \
    --ref-text "We asked over twenty different people, and they all said it was his."
```

- Long text is **auto-chunked** (sentence boundaries, ~28 s per chunk,
  50 ms stitch gaps). `--single-chunk` forces one pass (needs
  `--max-new-tokens` big enough; long single passes degrade — don't).
- Exit code 0 on success; the stdout line ends with `finish=eos|length` and
  `rtf=`. `finish=length` on a chunk means the frame budget was hit — fine
  for long scripts, suspicious for a short one.
- First call in a fresh process pays ~10–60 s of MLX compile/warmup before
  the first audio; steady-state RTF ≈ 0.68 (faster than real time).

## HTTP server

```bash
PYTHONPATH=. .venv/bin/python -m drama3_mac.cli serve --port 8843 &
curl -s localhost:8843/health
curl -s localhost:8843/v1/tts -H 'Content-Type: application/json' \
    -d '{"text": "Hello agent.", "reference_audio_b64": "...", "reference_text": "..."}' \
    -o out.wav
```

- Request JSON keys: `text` (required), `reference_audio_b64` +
  `reference_text` (cloning; base64 audio — repeat requests with the same
  bytes skip the encode via content-hash cache), `temperature` (0.8),
  `top_k` (30), `top_p` (0.9), `max_new_tokens` (0 = auto), `single_chunk`
  (false), `chunk_gap` (0.05), `response_format` (`wav`|`pcm_f32`).
- Response headers worth asserting: `X-Real-Time-Factor`, `X-Frames`,
  `X-Generation-Seconds`.
- `repetition_penalty`: accepted, default 1.0 = off. Fish's own engine ignores
  this field for Drama 3 (`repetition_penalty is ignored: S2-Pro and Drama 3
  use repetition-aware sampling instead`); the port implements an opt-in
  CTRL-style penalty over the last 64 main tokens, but the measured A/B
  (scripts/ab_sampling.py) found 1.2 degrades output — leave it off unless
  experimenting.

## Python API

```python
from drama3_mac.engine import Drama3Engine

engine = Drama3Engine("checkpoints/drama3-q4")
result = engine.tts(
    "Text to speak.",
    reference_audio=["assets/ref-en0.wav"],           # list => multi-reference
    reference_text=["We asked over twenty different people, and they all said it was his."],
)
engine.save(result, "outputs/out.wav")
result.real_time_factor   # audio_seconds / generate_seconds
result.chunks             # number of generated chunks (long text)
```

## Verify before claiming success

1. `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q` → 19 passed.
2. The output wav exists, is > 1 s, and `mlx_whisper.transcribe(...,
   path_or_hf_repo='mlx-community/whisper-medium.en-mlx')` returns text
   matching the prompt (this is the project's acceptance bar; spectral
   checks are NOT sufficient — see docs/JOURNEY.md).
3. For long scripts: listen for loop artifacts at chunk seams; every chunk
   should end `finish=eos`.

## Known limits

- Quantized projections are forbidden (gibberish; see README).
- `<|speaker:N|>` tags: at most 3 distinct speakers per auto-chunk.
- RTF on short clips (~3 s audio) reads 1.0–1.4 because warmup amortizes
  poorly; use the steady-state number from `scripts/bench_decode.py`.
