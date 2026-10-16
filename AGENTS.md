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
- **Guardrails**: preflight FAILs (unclosed `<tag>`, prompt-capacity overflow
  from a long reference + long text, broken/short reference clips, missing
  checkpoint files) refuse the job with a `PreflightError` naming the numbers;
  postflight WARNs (suspicious chunk endings, repetition loops in the codes)
  print on stderr and never fail the run. `--verify` additionally STT-round-
  trips the output with mlx-whisper medium.en and exits 1 below word-F1 0.8.
- Per-chunk sampling (`engine.tts(sampling_plan=[(0.7, 0.85), (0.8, 0.9)])`)
  renders the measured hybrid recipe in one call: cool 0.7/0.85 narration,
  hot 0.8/0.9 emotional peaks, ~30 s multi-take reference.
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

1. `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q` → 23 passed.
2. The output wav exists, is > 1 s, and `mlx_whisper.transcribe(...,
   path_or_hf_repo='mlx-community/whisper-medium.en-mlx')` returns text
   matching the prompt (this is the project's acceptance bar; spectral
   checks are NOT sufficient — see docs/JOURNEY.md).
3. For long scripts: listen for loop artifacts at chunk seams; every chunk
   should end `finish=eos`.

## HTTP server

The server is a production TTS API for pipeline consumers (designed for a
per-line podcast renderer). Start it with `drama3 serve --port 8843`.

- **Voices**: `POST /v1/voices {name, audio_b64, text, recipe{temperature,
  top_p, top_k, repetition_penalty}}` persists a named voice (refs on disk,
  git-ignored; multi-take = `audio_b64` as an array). Renders accept
  `voice=name`; the voice recipe fills sampling defaults, explicit request
  params win. `GET /v1/voices`, `GET /v1/voices/{name}`,
  `DELETE /v1/voices/{name}`.
- **JSON envelope**: `response_format=json` returns `{audio_b64, text,
  duration_s, rms, f0_hz_median, voiced_ratio, finish_reason, degraded,
  rtf, seed, temperature, request_echo, attempts, warnings, cache, ok}` —
  metrics are measured on the returned audio (cheap numpy autocorrelation
  f0; `voiced_ratio` exposes long silent gaps that rms hides), so callers
  can gate lines before saving. `seed`/`temperature` report the winning
  attempt (replay-pin those). Raw `wav` stays the default.
- **Seeds + cache**: `seed` makes renders deterministic (same seed + params
  = bit-identical take, verified); results cache under
  hash(voice, text, params, seed); `GET /v1/cache` gives hit/miss stats.
- **Retry ladder (loop-aware)**: for short-ish text the server retries until a
  **clean** take — EOS *and* no loop-detector/immediate-EOS warnings — keeping
  the first clean one; per-attempt entries carry `loop: bool` and the final
  attempt escalates temperature to 1.0 when ≤0.9 (breaks direction-content
  loop basins; skipped when `sampling_plan` is explicit). `max_retries`
  (default 2, max 5) sizes the ladder; resplit order 300→150→75 B.
  Exhaustion is HTTP 422 with the attempt list. `accept_degraded=true`
  ships the best non-looping take (marked `degraded: true` + warning)
  instead of 422; looped takes never ship. `retry=false` opts out entirely.
- **Line mode**: `line=true` renders one clean segment — no internal
  chunking, no baked gaps or edge fades (the consumer's assembler owns
  junction flow).
- **Batch**: `POST /v1/batch {lines: [{text, voice, line_id, ...}]}` renders
  sequentially through one warm model, `mx.clear_cache()` per line,
  per-line envelope array with error capture. Lines accept `speaker`
  (0–15) and any render knob (`seed`, `temperature`, `retry`,
  `max_retries`, `accept_degraded`, …).
- **Dialogue (reactive)**: `POST /v1/dialogue {lines: [{text, voice,
  speaker?, line_id?, gap_before_s?}], history_turns?, gap_s?}` renders
  turn by turn; each line hears the previous line's text + audio as
  completed history turns (upstream `previous_takes` semantics), so replies
  react to the voice they answer. `speaker` (0–15) tags the line text; pair
  with reference transcripts tagged to the same ids or voices bleed into
  each other. Per-line `gap_before_s` overrides `gap_s` before that line.
  `history_turns: 0` reproduces deaf per-line rendering (A/B hook). Returns
  per-line envelopes (each with `start_s`) plus the assembled conversation
  wav (default `gap_s` 0.25 s) and aggregate top-level `warnings`. Engine-
  level: `engine.tts(..., history=[(text, codes), ...])` forces the single-
  chunk path and counts history frames in the capacity preflight.
  `/v1/tts` also accepts `speaker: N` directly (symmetric with dialogue).
- **Augmenter**: `POST /v1/enhance {lines: [{line_id?, text}], tone_hints?}`
  proposes delivery tags for bare author lines via an LLM (Star Gate
  PoolModel @ 127.0.0.1:30000 by default — `DRAMA3_ENHANCE_BASE_URL/MODEL/
  API_KEY/TIMEOUT` override; never touches MLX). Output feeds
  `/v1/dialogue` verbatim: per-line `{line_id, text, rationale, warnings}`.
  **OWNER RULE: never cap the LLM's output** — no `max_tokens`, no output
  restrictions of any kind; the models think as long as they need
  (`DRAMA3_ENHANCE_TIMEOUT=0` disables the liveness timeout too). An early
  2048 cap burned the whole budget on reasoning and returned null content —
  do not reintroduce caps. Validators strip `<|...|>`, drop unknown paired
  tags, remove line-start sound events, enforce ≤2 paired tags + ≤80-char
  directions, downcase ALL-CAPS, and fall back to the original line if word
  content changed. LLM down ⇒ HTTP 503 with the original lines echoed.
- **Point tags**: inline delivery cues (`[sighs]`, `[laughs]`,
  `<whisper>...</whisper>`, `[bracketed]` cues) are first-class text — the
  chunker never splits them. Documented upstream on the HF model card; test
  dialogue expressiveness with them before building fancier mechanisms.

## Known limits

- Quantized projections are forbidden (gibberish; see README).
- `<|speaker:N|>` tags: at most 3 distinct speakers per auto-chunk.
- RTF on short clips (~3 s audio) reads 1.0–1.4 because warmup amortizes
  poorly; use the steady-state number from `scripts/bench_decode.py`.
