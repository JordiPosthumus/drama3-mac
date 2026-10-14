# drama3-mac — Fish Audio Drama 3 TTS on Apple Silicon

An MLX port of [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3)
(35B-parameter MoE speech model, ~3B active) that runs entirely on Apple
Silicon — no CUDA anywhere. The backbone and audio-decoder tower run in MLX
(Metal), the DAC codec in PyTorch (MPS). Voice cloning (multi-reference),
automatic long-text chunking, theatric tag support, a CLI, and a local HTTP
server included.

> **Status (2026-10-14): working and verified.**
> Acceptance bar: a clip is accepted only if Whisper medium.en transcribes it
> back to the prompt. 3/3 accepted on the default checkpoint (two plain
> prompts + one cloning run), long theatric scripts round-trip when chunked,
> and the layer-parity suite is green. The one serious bug found so far
> (gibberish from over-quantized projections) is fixed and documented in
> [docs/JOURNEY.md](docs/JOURNEY.md).

**Performance** on an M3 Ultra (512 GB unified memory), experts-q4 checkpoint:
31.4 audio frames/s, steady-state **RTF ≈ 0.68** — faster than real time.
First synthesis in a fresh process pays ~10–60 s of MLX compile/warmup; short
clips therefore show RTF 1.0–1.4, long clips converge toward 0.7.

This is believed to be the first open Mac/MLX port of Drama 3.

## Installation

Requires an Apple Silicon Mac (M1 or later; M3 Ultra used for all numbers
here), macOS with Metal, and ~120 GB free disk for the download + one
converted checkpoint.

### 1. Python environment

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/JordiPosthumus/drama3-mac.git
cd drama3-mac
uv venv .venv --python 3.12
uv pip install -e .
uv pip install pytest  # optional, for the test suite
```

Or with plain venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Core dependencies (installed by `-e .`): mlx ≥ 0.29, torch ≥ 2.6 (MPS),
numpy, safetensors, tokenizers, einops, soundfile, soxr.

### 2. Download the original checkpoint (~40 GB)

```bash
huggingface-cli download fishaudio/drama-3 --local-dir model
```

This lands `model/codec.pth` (the DAC codec) next to the safetensors shards —
that is exactly where the engine looks for it. If you keep the codec
elsewhere, pass `--codec /path/to/codec.pth`.

### 3. Convert to an MLX checkpoint

```bash
PYTHONPATH=. python -m drama3_mac.convert --checkpoint model --out checkpoints/drama3-q4 --bits 4
```

| variant | command | on disk | peak RAM (measured) | use it for |
|---|---|---|---|---|
| **experts-q4** (default) | `--bits 4` | 25.5 GB | **27.3 GB** | everyday use, fastest |
| experts-q8 | `--bits 8` | 41.6 GB | **42.4 GB** | marginally higher fidelity |
| bf16 | `--bits 16` | 71.8 GB | **70.5 GB** | validation/reference |

RAM measured as peak process RSS during a real generation
(`scripts/measure_memory.py`); you want headroom above the listed number.

> ⚠️ **Never pass `--quantize-projections`.** Quantizing the attention/GDN
> projections (not just the experts) compounds noise through the 40-layer
> residual stream until the model speaks fluent gibberish. The FP8 release
> keeps everything except the experts in bf16, and so does this converter by
> default. Full root-cause story in [docs/JOURNEY.md](docs/JOURNEY.md).

### 4. Sanity check

```bash
PYTHONPATH=. python -m pytest tests/ -q          # 19 tests, ~3 s
PYTHONPATH=. python -m drama3_mac.cli synthesize "The quick brown fox jumps over the lazy dog." -o outputs/first.wav
```

## Usage

### CLI

```bash
# plain synthesis
drama3 synthesize "Hello from Drama 3, running natively on Apple Silicon." -o hello.wav

# voice cloning (5–10 s clean clip + its transcript; multiple references help)
drama3 synthesize "Now I sound like the reference." -o cloned.wav \
    --ref sample.wav --ref-text "Transcript of the reference clip."

# long text is auto-chunked at sentence boundaries (28 s budget per chunk,
# 50 ms stitch gaps); force one pass with --single-chunk (not recommended
# for long theatric scripts)

# serve
drama3 serve --model checkpoints/drama3-q4 --port 8843
```

### HTTP server

```bash
curl -s localhost:8843/health
curl -s localhost:8843/v1/tts -H 'Content-Type: application/json' \
  -d '{"text": "Local server test."}' -o out.wav
```

Request fields: `text` (required), `reference_audio_b64` + `reference_text`
(optional, cloning; repeat requests with the same clip bytes skip the encode
via a content-hash cache), `temperature` (0.8), `top_k` (30), `top_p` (0.9),
`max_new_tokens` (0 = auto), `single_chunk` (false), `chunk_gap` (0.05),
`response_format` (`wav` | `pcm_f32`). Responses carry `X-Generation-Seconds`,
`X-Frames`, `X-Real-Time-Factor` headers. `repetition_penalty` is deliberately
ignored — Drama 3 uses repetition-aware sampling instead. Requests are
serialized; all inference runs on the server's main thread.

### Direct Python

```python
from drama3_mac.engine import Drama3Engine

engine = Drama3Engine("checkpoints/drama3-q4")
result = engine.tts(
    "Long scripts are chunked automatically at sentence boundaries.",
    reference_audio=["a.wav", "b.wav"],            # multi-reference cloning
    reference_text=["transcript a", "transcript b"],
)
engine.save(result, "out.wav")
```

## Directing the performance (tag syntax)

Drama 3 reads stage directions natively. See
[examples/jabberwocky](examples/jabberwocky) for a full six-part showcase.

- **`[square-bracket cues]`** — free-form directions, pauses and sounds the
  model interprets but does not speak:
  `[pause]`, `[long pause]`, `[small laugh]`, `[calm, measured voice]`,
  `[door creaks slowly]`.
- **`<tag>…</tag>` paired tags** change delivery of the enclosed span:
  `whisper`, `emphasis`, `soft`, `fast`, `slow`, `stress`, `forceful`.

  ```text
  <whisper>Beware the Jabberwock, my son!</whisper>
  The spreadsheet... looked... <emphasis>fine</emphasis>.
  ```

- **Speakers**: prefix a line with `<|speaker:0|>` … `<|speaker:15|>`; the
  auto-chunker keeps at most 3 speakers per chunk.
- Cues and paired tags are protected spans — chunking never splits them.

## Performance notes (M3 Ultra)

| metric (experts-q4) | value |
|---|---|
| decode step | 31.8 ms mean (p50 31.9, p90 32.4) |
| frames/s | 31.4 (need 21.5 for real time) |
| steady-state RTF | ≈ 0.68 (experts-q4) / ≈ 0.77 (bf16) |
| prefill | ~1.9 s for a 31-row prompt |
| RAM | 27.3 GB peak RSS |

bf16 is only ~13% slower than q4 in steady state — decode is launch-bound, so
precision costs almost nothing at this scale; pick checkpoints for RAM, not
speed.

Decode is GPU kernel-launch-bound (~1500 small Metal kernels per step), not
bandwidth-bound: the ~7 GB of weights read per step would allow ~9 ms at peak
bandwidth. CUDA-graph capture does not exist on Metal; per-layer compiled
graphs (`mx.compile`) with explicitly threaded caches already removed the CPU
dispatch overhead. Benchmarks: `scripts/bench_decode.py <checkpoint>`.

## How it works

1. **Prompt building** — the transcript becomes a `[rows, 11]` token matrix:
   one main token (text/special tokens) + 10 audio codebook tokens per row,
   wrapped in Drama 3's `<|im_start|>`/`<|text|>`/`<|voice|>` template with a
   speaker tag (`<|speaker:0|>` … `<|speaker:15|>`).
2. **Reference encode** — the DAC codec converts each reference clip to
   10-row codes (row 0 = semantic codebook, vocab 4096; rows 1–9 = residual,
   1024), ~21.5 frames/s of audio. All references become `[Reference N]`
   prompt blocks.
3. **Long-text chunking** — text over ~300 bytes is split at sentence
   boundaries (protected spans kept whole); each chunk is generated with a
   ~28 s frame budget, and the first + latest chunks feed back as history so
   voice and pacing stay consistent.
4. **Prefill** — chunked teacher-forced pass over the prompt rows builds the
   backbone's recurrent (GatedDeltaNet) and KV caches.
5. **Decode** — each step: backbone predicts the next main token over
   (EOS + semantic band); the tower predicts the 10 codebook tokens for that
   frame. RAS sampling re-rolls the first codebook within a 16-frame window
   when a repeated candidate wins, keeping long generations stable.
6. **Reconstruction** — `codec.pth` (modded DAC + downsampled RVQ) decodes
   each chunk's code matrix to 44.1 kHz audio; chunks are stitched with a
   small gap.

Numerics were validated layer-by-layer against the fish-speech PyTorch
reference (`tests/`): norms, RoPE, full attention, GatedDeltaNet prefill and
recurrent decode, MoE routing, tower prefill/decode, and the chunk splitter
(byte-identical) — 19 tests, all green.

## Repository layout

```
drama3_mac/
  config.py      # fish_omni config parser (Qwen3-Next flags)
  convert.py     # HF checkpoint -> MLX (FP8 dequant, expert stacking, quantize)
  module.py      # minimal nn.Module with KV-cache registry
  layers.py      # norms, RoPE, full attention, GatedDeltaNet (+ Apple kernel)
  moe.py         # sparse MoE: gathered (decode) and grouped (prefill) paths
  backbone.py    # 40-layer hybrid decoder
  tower.py       # Codebook-MTP audio tower + RAS sampling
  chunking.py    # long-text chunking + history (port of fish_omni.chunking)
  codec_dac.py   # vendored DAC codec (inference only)
  codec.py       # encode/decode wrappers
  prompt.py      # Drama 3 prompt template
  generate.py    # chunked prefill + decode loop
  engine.py      # high-level TTS engine (chunking, multi-reference, stitching)
  cli.py         # `drama3` command
  server.py      # local HTTP server
  gated_delta.py # vendored Apple Metal kernel (MIT)
scripts/         # benchmark, acceptance suite, memory measurement, probes
tests/           # parity tests vs the PyTorch reference + chunking tests
examples/        # Jabberwocky theatric showcase
docs/JOURNEY.md  # dated port log with every bug hunt
AGENTS.md        # quickstart for coding agents
```

## License

This repository derives from [fish-speech](https://github.com/fishaudio/fish-speech)
and is therefore distributed under the **Fish Audio Community License 1.0**
(see LICENSE): free for research and personal use, and for commercial use
while annual revenue is below USD 500,000; hosted-service use always requires
a separate license from Fish Audio. `drama3_mac/gated_delta.py` is vendored
from mlx-lm (Apple Inc., MIT). The Drama 3 model weights themselves are
governed by Fish Audio's own terms and are **not** distributed here.

## Credits

- [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) — reference implementation, DAC codec
- [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) — model
- [mlx-lm](https://github.com/mlx-explore/mlx-lm) — GatedDeltaNet Metal kernel
- [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) — DAC architecture
