# Porting Fish Audio Drama 3 to Apple Silicon — Journey Log

Dated entries, newest last. This is the story of getting
[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) (35B MoE TTS,
~3B active) running natively on an M3 Ultra with MLX, and of the bugs found
along the way.

## 2026-10-13 — Setup and reconnaissance

- Verified the machine: Apple M3 Ultra, 512 GB unified memory, Metal 4,
  32 CPUs. Python 3.12 venv with MLX 0.32.3, torch 2.12 (MPS).
- Cloned `fish-speech` (PyTorch reference implementation) and
  `sglang-omni` (serving stack) into local `research/` (untracked) and read
  the FishOmni model code end to end: hybrid backbone (30 GatedDeltaNet
  linear-attention layers + 10 full-attention layers per 40-layer block),
  Codebook-MTP audio tower (10 codebooks, RAS retry sampling), and the
  modded-DAC codec (44.1 kHz, downsampled RVQ, 21.53 frames/s).
- Started the 40 GB HF download of `fishaudio/drama-3` early (8 bf16/fp8
  safetensors shards + `codec.pth`). The release stores MoE experts as
  **block-FP8** (`bsgemm-moe`, 128×128 blocks, fp32 scales) — Metal has no
  FP8 GEMM, so a conversion pass is mandatory.

## 2026-10-13 — Architecture decisions

- **MLX for the backbone + tower; PyTorch-MPS for the DAC codec.** The codec
  is a small conv/transformer stack where PyTorch is fine and the vendored
  reference code helps; the backbone is where MLX shines.
- **Dequantize FP8 at conversion time.** MLX has no fp8 dtype; per-launch
  dequantization would waste bandwidth anyway. Experts are dequantized to
  bf16, stacked per layer (`w13` = gate rows then up rows, `w2` = down), then
  optionally quantized to 4/8-bit for `quantized_matmul`.
- **Vendored Apple's packed GatedDeltaNet Metal kernel** from mlx-lm (MIT) —
  it is generic in Hk/Hv, so 16 k-heads vs 32 v-heads works.
- Numerics policy: bf16 residual stream, fp32 norms and GDN state, matching
  the reference.

## 2026-10-13 — Conversion pipeline

`drama3_mac/convert.py` reads the HF shards lazily, dequantizes FP8 expert
blocks (verified against the fp32 reference path), stacks experts, quantizes
to 4/8-bit or keeps bf16, and copies embeddings/lm-head/router/norms/tower
verbatim in bf16. First conversion bug found and fixed the same day: a
doubled `.weight` suffix on quantized projection keys.

## 2026-10-13 — Parity tests

`tests/` compares the MLX port against the fish-speech PyTorch reference
module-by-module with the real checkpoint weights: RMSNorm variants, RoPE
(neox partial for the backbone, interleaved for the tower), full attention
with QK-norm and output gate, GatedDeltaNet chunked prefill + recurrent
decode + multi-chunk state handoff, MoE routing (gathered and grouped
paths), tower prefill and greedy decode-step. Nine tests, all green:
≤2e-4 everywhere except GDN long-context (≤5e-3).

## 2026-10-13 — First end-to-end audio

Text + 3.9 s reference clip → 2.1 s of audio with a clean EOS finish.
Spectral check looked healthy (96%+ energy in the speech band) — **this
 acceptance bar later proved far too weak** (see the bug hunt below).

## 2026-10-13 — Performance work

- Per-layer `mx.compile` decode graphs: GDN steps thread conv/recurrent
  caches explicitly in and out (no position-dependent slicing in the GDN
  decode path), MoE decode uses the gathered path compiled with weights as
  constants, the top-k/top-p sampler is compiled with the random noise
  passed in as an argument so compiled traces stay deterministic.
- Full-attention decode reads the whole cache behind a position mask and
  writes through `put_along_axis` (position as an array), keeping every
  shape static for compilation.
- Result: 36 frames/s (need 21.5 for real time), steady-state RTF ≈ 0.59
  with 4-bit experts. Step time is Metal kernel-launch-bound (~1500 small
  kernels/step), not bandwidth-bound; CUDA-graph capture does not exist on
  Metal, so this is the practical floor for this stack.

## 2026-10-14 — CLI, server, packaging

- `drama3 synthesize` and `drama3 serve` (dependency-free stdlib HTTP
  server; all inference on the loader thread because MLX streams are
  thread-local — first version crashed with cross-thread stream errors).
- Reference clips cached by content hash; `/v1/tts` returns audio/wav with
  `X-Generation-Seconds` / `X-Real-Time-Factor` headers.
- Repo published (initially private, now public as a WIP) with the Fish
  Audio Community License 1.0 (derivative of fish-speech), Apple MIT notice
  for the vendored kernel, gitleaks scan clean before every push.

## 2026-10-14 — The intelligibility bug hunt

Independent verification failed: Whisper (base.en and medium.en) heard
gibberish — "speech-shaped audio, clean EOS, wrong words" (e.g. "Koppo hai
no manto"). Re-opened the goal with a proper acceptance bar: **a clip is
accepted only if Whisper medium.en transcribes it back to the prompt**.

Evidence gathered, in order:

1. **Codec vindicated.** `decode(encode(reference clip))` → Whisper hears
   the original transcript verbatim. (The earlier sine-wave round-trip was
   too weak a test; real speech through the full chain is the real test.)
2. **Prompt builder vindicated.** My prompt rows are byte-identical to
   `build_drama3_prompt` for both cloning and plain synthesis.
3. **Sequencing vindicated.** `generate.py` matches the reference loop
   step for step (prefill pairing, cache slots, EOS handling).
4. **bf16 checkpoint is perfect.** The unquantized MLX checkpoint
   transcribes "The quick brown fox jumps over the lazy dog." perfectly.
   The port core is correct; the defect lives in the quantized checkpoints.
5. **Greedy decoding is still wrong on q4/q8.** q4 near-greedy says "We
   think like we think it." — deterministic, fluent, wrong content. Not
   sampling noise.
6. **Teacher-forced divergence.** Feeding identical frames to the q8 and
   bf16 models, decode-step logit error starts at ~8% and *compounds* to
   ~28% over six steps, while bf16-vs-bf16 stays exact.
7. **State autopsy after prefill.** GDN recurrent-state error grows layer
   by layer: 0.9% (layer 0) → 3% (layer 2) → **55% (layer 29)**; layer-39
   KV cache ~65-70% error; final hidden ~44% error. Classic compounding of
   per-layer quantization noise through a 40-layer residual stream.
8. **But the reference runs FP8 experts (~6% noise) just fine** — which
   falsifies "the model is quantization-fragile" and points at *what* I
   quantized: the HF release keeps **everything except the experts in
   bf16**. My converter also quantized the attention/GDN/shared-expert
   projections (~2.5 GB), deviating from the reference precision layout.

**Root cause: over-quantization.** Quantizing the projections adds per-layer
noise exactly where the recurrent state and KV caches accumulate it; the
semantic codebook candidates are separated by ~0.05-0.4 logits, so by the
time the residual stream carries 25-45% relative error, phoneme selection
is scrambled — fluent speech-shaped audio saying the wrong words.

**Fix: experts-only quantization** (mirrors the FP8 release). Converter
default changed; `--quantize-projections` exists for the adventurous.

- `checkpoints/drama3-q8` (experts q8, 41.6 GB): greedy round-trip
  **"The quick brown fox jumps over the lazy dog."** — perfect.
- `checkpoints/drama3-q4` (experts q4, 25.5 GB): greedy round-trip
  **"The quick brown fox jumps over the lazy dog."** — perfect.

## Status

- **ACCEPTED 2026-10-14**: full suite on the default experts-q4 checkpoint
  (25.5 GB), natural sampling defaults (temperature 0.8, top-k 30, top-p 0.9),
  Whisper medium.en round-trip, token-F1 ≥ 0.8:

  | case | prompt | heard | f1 | RTF |
  |---|---|---|---|---|
  | plain-a | Drama three runs entirely on Apple silicon with no CUDA. | Drama 3 runs entirely on Apple Silicon with no CUDA. | 0.90 | 1.20 |
  | plain-b | The quick brown fox jumps over the lazy dog. | The quick brown fox jumps over the lazy dog. | 1.00 | 0.85 |
  | clone-c (cloning) | Testing voice cloning with my own reference clip on the Mac. | Testing voice cloning with my own reference clip on the Mac. | 1.00 | 1.00 |

- Steady-state decode (experts-q4): 31.4 frames/s, RTF ≈ 0.68 (31.8 ms/step).
  Slightly slower than the earlier broken q4 (27.5 ms) because the attention/
  GDN projections are now bf16 — the correct trade for intelligibility.
- experts-q8 (41.6 GB) and bf16 (71.8 GB) also verified intelligible.
- Suite: `scripts/acceptance.py` (run it against any checkpoint).

## 2026-10-14 (afternoon) — Long-text chunking, more bugs found, docs pass

- **Long-text chunking landed.** `drama3_mac/chunking.py` ports
  `fish_omni.chunking` exactly (~300-byte chunks at sentence → clause → char
  boundaries, protected spans for `[cues]` and `<paired> tags`, speaker
  regrouping ≤3 speakers/chunk, pinned-first-chunk history with fallback
  candidates, 600-frame ≈ 28 s budget per chunk). `engine.tts` auto-chunks,
  stitches with a configurable gap (50 ms default), and uses one fixed cache
  capacity per run so compiled decode graphs are built once. Splitter parity
  with the reference is byte-identical (5 text shapes in tests).
- **The failing script now round-trips**: the 1234-char theatric script that
  degenerated at ~23 s in a single 142 s pass ("Thanks for watching!" loops,
  finish=length) generates 90.7 s of audio chunked, and Whisper medium.en
  hears every story beat; the final "So! How are YOU?" lands correctly.
- **1-row prefill crash found and fixed** (hit while rendering an 885-row
  long-tagged poem with history): a prompt with rows % 128 == 1 makes the
  final prefill chunk exactly one row, which was routed into the compiled
  decode step and crashed on the missing position. Routing now requires the
  position argument; regression tests added (unit + real-model script with a
  385-row prompt).
- **[long pause] / RAS probe**: pause tags raise in-window repeat pressure
  (39% vs 16% of semantic picks) but do NOT trap the RAS sampler — both probe
  generations finished with EOS. The degeneration was a long-single-pass
  problem; chunking bounds it. No RAS change made, probe kept as
  `scripts/probe_ras_pause.py`.
- **Multi-reference cloning** exposed in `engine.tts` (lists of clips +
  transcripts → multiple `[Reference N]` blocks); owner's adherence study
  measured 0.939 speaker-embedding cosine with a 3-take reference vs 0.928
  single-take.
- **Measured memory table** (`scripts/measure_memory.py`, peak RSS during a
  real generation): q4 27.3 GB, q8 42.4 GB, bf16 70.5 GB. bf16 steady-state
  RTF ≈ 0.77 — only 13% slower than q4; decode is Metal launch-bound, so
  precision is nearly free and checkpoints should be picked for RAM.
- **Docs pass**: README rewritten (human install guide with exact uv/venv
  commands, measured memory table, performance section, tag-syntax docs),
  `AGENTS.md` added for coding agents, `examples/jabberwocky/` added as the
  six-part theatric showcase (whisper/terror/shout/glee directions).
