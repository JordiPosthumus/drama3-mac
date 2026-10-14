# drama3-mac — Fish Audio Drama 3 TTS on Apple Silicon

An MLX port of [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3)
(35B-parameter MoE speech model, ~3B active) that runs entirely on Apple
Silicon — no CUDA anywhere. The backbone and audio-decoder tower run in MLX
(Metal), the DAC codec in PyTorch (MPS). Voice cloning from a short reference
clip, a CLI, and a local HTTP server included.

> **Status (2026-10-14): intelligible. Acceptance suite passed.**
> A clip is accepted only if Whisper medium.en transcribes it back to the
> prompt. On the default experts-q4 checkpoint (25.5 GB) with natural sampling
> defaults: 3/3 accepted — two plain-synthesis prompts (f1 0.90 / 1.00) and a
> voice-cloning run (f1 1.00). Steady-state decode RTF ≈ 0.68. The earlier
> gibberish-audio bug was over-quantization of the attention/GDN projections;
> the converter now quantizes experts only, mirroring the FP8 release.
> Full story in [docs/JOURNEY.md](docs/JOURNEY.md).

**Performance** on an M3 Ultra (512 GB unified memory): 31.4 audio frames/s
during decode, steady-state **RTF ≈ 0.68** — faster than real time, verified
intelligible via Whisper round-trip. The default experts-q4 checkpoint needs
~25 GB of RAM; experts-q8 ~42 GB; bf16 ~72 GB.

This is believed to be the first open Mac/MLX port of Drama 3.

## What's implemented

| Component | Approach |
|---|---|
| Text backbone (Qwen3-Next-style hybrid) | MLX: 30 GatedDeltaNet (linear attention) + 10 full-attention layers, sigmoid-gated outputs, zero-centered RMSNorm, partial neox RoPE, QK-norm |
| MoE (128 experts, top-8, sigmoid gating + shared expert) | FP8 block-scaled weights are **dequantized at conversion time** (Metal has no FP8 GEMM), then optionally quantized to 4/8-bit for MLX `quantized_matmul` |
| GatedDeltaNet kernel | Apple's packed Metal kernel, vendored from [mlx-lm](https://github.com/mlx-explore/mlx-lm) (MIT) |
| Audio tower (Codebook-MTP, 10 codebooks) | MLX, with RAS retry sampling (window 16 frames, top-k 30, top-p 0.95) |
| DAC codec (44.1 kHz) | PyTorch/MPS, vendored inference path from fish-speech's modded DAC |
| Decode hot path | Per-layer compiled graphs (`mx.compile`); caches threaded explicitly; position-dependent writes via `put_along_axis` |

## Setup

Requires an Apple Silicon Mac, Python ≥ 3.10, and ~40 GB free disk for the
model. MLX ≥ 0.29, PyTorch ≥ 2.6 (MPS), soundfile, soxr, tokenizers, einops,
safetensors, numpy.

```bash
# 1. download the original checkpoint (~40 GB)
huggingface-cli download fishaudio/drama-3 --local-dir model

# 2. convert to an MLX checkpoint (4-bit experts by default; projections stay bf16)
python -m drama3_mac.convert --checkpoint model --out checkpoints/drama3-eq4 --bits 4
#    experts-q8: --bits 8 (~42 GB)   bf16: --bits 16 (~72 GB)
#    NOTE: do not pass --quantize-projections; quantizing the attention/GDN
#    projections compounds noise until the model speaks gibberish (see docs/JOURNEY.md)

# 3. put codec.pth where the engine can find it (default: model/codec.pth)
```

## Usage

### CLI

```bash
# plain synthesis (voice #0 by default)
drama3 synthesize "Hello from Drama 3, running natively on Apple Silicon." -o hello.wav

# voice cloning: 5-10 s clean clip + its transcript
drama3 synthesize "Now I sound like the reference." -o cloned.wav \
    --ref sample.wav --ref-text "Transcript of the reference clip."

# serve
drama3 serve --port 8843
```

### HTTP server

```bash
drama3 serve --model checkpoints/drama3-q4 --port 8843
```

```bash
curl -s localhost:8843/health
curl -s localhost:8843/v1/tts -H 'Content-Type: application/json' \
  -d '{"text": "Local server test."}' -o out.wav
```

Request fields: `text` (required), `reference_audio_b64` + `reference_text`
(optional, enables cloning), `temperature` (0.8), `top_k` (30), `top_p` (0.9),
`max_new_tokens` (2048), `response_format` (`wav` | `pcm_f32`).
Responses carry `X-Generation-Seconds`, `X-Frames`, `X-Real-Time-Factor`
headers. Reference clips are encoded once and cached by content hash.
Requests are serialized; all inference runs on the server's main thread.

## How it works

1. **Prompt building** — the transcript becomes a `[rows, 11]` token matrix:
   one main token (text/special tokens) + 10 audio codebook tokens per row,
   wrapped in Drama 3's `<|im_start|>`/`<|text|>`/`<|voice|>` template with a
   speaker tag (`<|speaker:0|>` … `<|speaker:15|>`).
2. **Reference encode** — the DAC codec converts the reference clip to 10-row
   codes (row 0 = semantic codebook, vocab 4096; rows 1–9 = residual, 1024),
   ~21.5 frames/s of audio.
3. **Prefill** — chunked teacher-forced pass over the prompt rows builds the
   backbone's recurrent (GatedDeltaNet) and KV caches.
4. **Decode** — each step: backbone predicts the next main token over
   (EOS + semantic band); the tower predicts the 10 codebook tokens for that
   frame. RAS sampling re-rolls the first codebook within a 16-frame window
   when a low-rank candidate wins, keeping long generations stable.
5. **Reconstruction** — `codec.pth` (modded DAC + downsampled RVQ) decodes the
   code matrix to 44.1 kHz audio.

Numerics were validated layer-by-layer against the fish-speech PyTorch
reference (`tests/`): norms, RoPE, full attention, GatedDeltaNet prefill and
recurrent decode, MoE routing, and tower prefill/decode all match within
2e-3 (GDN) / 2e-4 (everything else) at bf16.

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
  codec_dac.py   # vendored DAC codec (inference only)
  codec.py       # encode/decode wrappers
  prompt.py      # Drama 3 prompt template
  generate.py    # chunked prefill + decode loop
  engine.py      # high-level TTS engine
  cli.py         # `drama3` command
  server.py      # local HTTP server
  gated_delta.py # vendored Apple Metal kernel (MIT)
scripts/         # conversion checks, benchmark, first end-to-end run
tests/           # parity tests vs the PyTorch reference
```

## Performance notes

Measured on M3 Ultra, q4 checkpoint, 31-row prompt (scripts/bench_decode.py):

| metric | value |
|---|---|
| prefill (31 rows) | 1.9 s |
| decode step | 31.8 ms mean (p50 31.9, p90 32.4) |
| frames/s | 31.4 (need 21.5 for real time) |
| steady-state RTF | ≈ 0.68 (experts-q4) |
| RAM (experts-q4) | ~25 GB |

The step time is GPU-kernel-launch-bound (~1500 small kernels per step on
Metal), not bandwidth-bound (~7 GB of weights read per step would allow ~9 ms
at peak bandwidth). CUDA-graph-style capture isn't available on Metal; the
per-layer compiled graphs and explicit cache threading already removed the
CPU dispatch overhead.

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
