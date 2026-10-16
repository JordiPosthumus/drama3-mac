# Audit: where the port falls short of native Drama 3 (dialogue focus)

Verdict first: the core generation path is a faithful port — prompt builder, chunking
constants/rules (300 B / 3 speakers / 600 frames / 1 recent turn) and repetition-aware
sampling are line-level identical to upstream (verified against fish_omni/{generate,chunking,
tower}.py on 2026-10-14). The gaps are in what is *exposed* and *tested*, not in the math.

## Gap table

| # | Capability | Upstream | Port | Verdict | Dialogue impact |
|---|---|---|---|---|---|
| 1 | Multi-turn audio context (reactive dialogue) | `generate(history=...)` in fish_omni; `previous_takes` on SGLang-Omni API | `build_prompt` supports history turns; used only for chunk continuity; `engine.tts()` does not expose it | **Missing exposure** | **HIGH** — the only mechanism for lines that respond to the heard line |
| 2 | Point tags (`[sighs]`, `[laughs]`, `<whisper>`) | Documented on HF card; chunker protects them | Chunker protects them; never documented, never tested, no guardrail guidance | Present but **undocumented/untested** | **HIGH** — zero-code expressiveness lever |
| 3 | 2-voice tag->role binding | Speaker tags are "the main Drama 3 usage" | Works (F0 separation verified A 138-155 / B 172-196 Hz across 6 chunks) but no test covers two references; the only multi-speaker test uses one reference | Untested | HIGH — core podcast path |
| 4 | Sampling defaults | Docs: 0.7/0.7/30; HF bench: 1.0/0.9/30 | 0.8/0.9/30; 0.7/0.7 measured to loop (q4: 25+ loop warns; bf16: 4) | Deliberate deviation, defensible | MEDIUM — re-verify per release |
| 5 | Streaming, mp3/opus, msgpack, WebUI, /v1/tts/defaults | Official + SGLang | Absent; port adds voices/batch/line-mode/envelope metrics/retry ladder instead | Missing (low dialogue impact) | LOW |
| 6 | Reference length limit | 60 s hard fail (SGLang) | WARN >30 s (guardrails), no hard fail | Looser | LOW |
| 7 | Precision | FP8 experts / bf16, int4 official | q4/q8/bf16 MLX; 0.7/0.7 loops even on bf16 (mild) | Platform deviation | MEDIUM — keep 0.8/0.9 |
| 8 | Line-level reactivity in batch mode | n/a upstream | Batch lines are deaf renders with uniform 250 ms gaps | Design gap in our consumer story | HIGH for podcast naturalness |

## Test-surface gaps (tests/)
- `test_multi_speaker_tagged_text_auto_chunk` (2026-10-15 blocker regression) uses ONE
  reference: proves no-crash, never proves two voices, never measures expressiveness.
- No test renders the same line with vs without audio history (reactivity A/B).
- No test exercises point tags end-to-end.

## Top 5 gaps by dialogue impact (ranked)
1. Expose conversation-turn history (text + audio) through `engine.tts` and the server
   (mirror SGLang's `previous_takes`); render B's line hearing A's actual audio.
2. Test + document point tags; add a dialogue test with two tagged references asserting
   per-turn F0 separation.
3. Add reactivity expressiveness metric (per-turn F0 variance / contour) to the envelope so
   flat takes can be auto-detected and re-rolled.
4. Per-line emotion control story (request-level temperature overrides exist; document a
   narration-vs-peaks recipe like the measured hybrid plan).
5. Reference-clip guidance for expressive roles (energetic multi-take refs; 5-30 s sweet spot).

## Immediate zero-code experiments
- Point tags: render the A/B dialogue with `[sighs]`, `[laughs]`, `<whisper>...</whisper>`
  inline and A/B against the plain render.
- `previous_takes`-style continuation: chain a line render with the previous line's text +
  codes via a small engine-level script (build_prompt already accepts history) before doing
  the full server exposure.
