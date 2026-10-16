# Drama 3 port — measured discoveries (2026-10-15)

Every claim below was measured on this machine (q4 checkpoint, seed-locked,
whisper word-timestamps / F0 / energy analysis), most of them twice. File
paths reference the probes that produced the evidence. This document is the
"why" behind the rules hard-coded in `drama3_mac/enhance.py` and
`scripts/augment_score.py`.

## 1. The pause ladder (what the model actually does with punctuation)

Measured gap between two target words, whisper word-timestamps, seed 777
(`outputs/probe-pause/`, `scripts/probe_cadence.py`):

| text | pause |
|---|---|
| `routine, that` (comma) | **0 ms** — the model ignores commas |
| `<slow>that matters</slow>` | 520 ms |
| `(break)` (legacy V1.6 cue — bleeds into Drama 3) | 520 ms |
| `routine... that` (ellipsis) | 560 ms |
| `[measured pace, letting the point land]` brief | 640 ms |
| `routine. That` (full stop) | 1000 ms |
| `[break]` | 820 ms **on a short /v1/tts line** — see §2 |
| `[long-break]` | 1320 ms **on the same probe** — see §2 |

## 2. Bracket breaks are bimodal in dialogue renders — banned

Across five tuning cycles, `[break]` in `/v1/dialogue` renders either as
**0 ms (vanishes)** or as an **absurd silence** (4.9 s, worst: 22.8 s).
`[long-break]`: exploded once, vanished once, clean once. The ellipsis
rendered true **every single time** (520-560 ms), including inside author
text. Verdict: `[break]`/`[long-break]` are banned in the director prompt and
the validator normalizes both to `...`. Evidence: `research/augment-tune-log.md`
cycles 1-4, `outputs/grin-probe/`, `outputs/cycle*-lines/`.

## 3. The manic laugh: "grinning" + `<emphasis>` (owner ear-confirmed)

`[grinning, leaning in]` at a line start produced laughing-while-speaking
(F0 excursions to 165 Hz on a ~125 Hz voice). Systematic ablation: the laugh
did NOT reproduce with the brief alone, the tag alone, or the tag swapped to
`smiling` — only the **co-occurrence of the facial-verb brief with
`<emphasis>`** reproduced it deterministically at seed 43. Rule: never
"grinning"/"smirking" in briefs; "delighted"/"warm"/"smiling" instead.
Probe set: `outputs/grin-probe/` (the user's own hypothesis, confirmed by
bisect after a null first round — see tune log).

## 4. Tag spans are voice identity: fractions, never wholes

Whole-sentence `<soft>`/`<slow>` spans made the guest render at **122.5 Hz —
the host's register** ("she totally loses her voice id", owner ears) with a
6-second unvoiced hole. The same line with **short spans** renders at
169.6 Hz — identity intact. Short spans shade the performance; sentence-long
spans replace the performer. Rule: paired tags wrap a few words, max 2 per
line, none on lines shorter than 8 words (tiny lines + tags → noise fills,
measured twice: 14.7 s of pitchless sound, vr 1.0 with f0 absent).

## 5. Line-start cues instantiate state; mid-line cues shade

`[sighs]` at line start → 8 s of silence. `[grinning, leaning in]` at line
start → laugh onset. The same cues mid-line shade delivery cleanly
(`[excited]` measured +37 Hz). Onset cues set the opening *state* and the
model vocalizes what it cannot show. Rule: small sounds never start a line;
the validator strips leading sound-event cues.

## 6. Opening pitch discipline + the expressiveness calibration

`<emphasis>` on the FIRST word of a line → 176 Hz onset swing. Owner
calibration by ear: a 100 Hz-swing take = "alive", a 132 Hz-swing host intro
= "nice pod intro, not flat, not exaggerated". Swing ceiling set to 160 Hz —
it bounds the insane, not the expressive; the vocabulary rules handle the
laugh trigger. Host F0 band widened to 105-165 Hz on the same evidence.

## 7. Speaker tags are acoustically inert with untagged reference transcripts

`<|speaker:N|>` prefixes produced **bit-identical audio** with and without
the tag (same duration to the millisecond, same F0 pattern) when the voice
references carry untagged transcripts. Either tag the reference transcripts
to match, or don't rely on speaker tags in per-line rendering.

## 8. Whisper measurement gotchas (for anyone scoring renders)

- Word timestamps **clamp intra-segment pauses to 0**: a pause that doesn't
  split the segment reads as contiguous speech. Cross-check every pause with
  an energy-gap measurement around the marker's position; take the max.
- Under memory pressure whisper can return **zero tokens** (transient) — retry.
- Normalize number words for word-F1 ("thirty-one" → "31", "four thousand" →
  "4000") or every F1 is falsely low.
- Never trust a single take: the laugh needed a 6-cell ablation to attribute.

## 9. Ops: memory pressure is a render killer

bf16 (~40 GB mapped) + a resident 282 GB llama-server = page thrashing,
~10 min per line (weights re-read from disk every pass). q4 (~16 GB) renders
fine alongside it. Keep one big model resident at a time; mmap'd RSS lies —
check swap and page pressure, not RSS.

## 10. The system this produced

- `drama3_mac/enhance.py` — `/v1/enhance`: Star Gate PoolModel director +
  7 validator layers (special tokens, unknown paired tags, line-start sound
  events, bracket balance, tag budget, ALL-CAPS, word preservation with
  fallback). Never touches MLX; never caps LLM output (owner rule).
- `scripts/augment_score.py` — objective scorer: F0 bands per voice,
  pause verification (whisper + energy cross-check), duration vs word count
  (pause-aware), voiced_ratio (speech regions), onset swing, word preservation.
- `research/augment-tune-log.md` — the one-rule-per-cycle protocol and every
  cycle's table. Final lighthouse benchmark: **10/10 lines pass all gates,
  0.990 per-line word-F1 (min 0.94, bar 0.8)**, pauses verified at
  640-1100 ms including the owner's "routine... that matters" beat at 760 ms.
