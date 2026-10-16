# MakeAPod integration plan — synthesis of the five-lane council review

Sources: lane-engine (scout), lane-cadence (worker), lane-augmenter (worker),
lane-reliability (reviewer), lane-contract (reviewer). All read-only; full reports
in the workflow receipt / agent logs (2026-10-15).

## Phase 1 — Reliability core (build first; every lane converges here)

1. **Loop-aware retry ladder** (engine S / reliability M — same edit). The single
   control point is `service.py:352`: today `finish==eos` breaks the ladder even
   when loop-detector warnings fired. Change to: `clean = eos AND no loop/`
   `immediate-EOS warnings`; break only on clean; track the FIRST clean take
   (a looped retry must not displace it); add `"loop": bool` to attempts[];
   on exhaustion raise only if NO attempt was clean. Policy numbers: keep 3
   attempts, keep `seed+attempt` (deterministic, basins are seed-locked),
   final attempt escalates temperature to 1.0 when current ≤0.9 (explicit
   high-temp requests untouched). Record the winning attempt's actual
   temperature in attempts[]. With p(loop)≈⅓, 3 attempts ⇒ ~96% clean/line.
2. **`max_retries` request knob** (clamp ≤5) + optional explicit `seeds:[...]`
   sweep for best-of-N with deterministic replay.
3. **`voiced_ratio` metric** (new; metrics.py f0 already computes the voiced
   frame set — expose the ratio). The only engine-adjacent change; it is what
   catches silent-gap takes (the 8 s `[sighs]` explosion) that pass rms/f0 today.
4. **Envelope additions** (additive, S): per-line `text` (exact rendered text
   incl. speaker tag), `speaker`, `sample_rate`, `start_s` in dialogue results;
   aggregate top-level `warnings` in the dialogue envelope.
5. **`accept_degraded` request flag** (default false): on ladder exhaustion,
   if the best attempt has NO loop warning, return it `ok:true, degraded:true`
   instead of a permanent hole in the conversation; looped takes stay 422.
6. **Per-line no-ears gate** (MakeAPod-side, cheap): accept iff eos + no
   loop/finish-discipline warnings + `duration_s ≥ 0.35` and ≥ ~40% of
   expected chars/15 + `rms ≥ 0.01` + f0 within ±40% of the voice's rolling
   baseline + `voiced_ratio ≥ 0.35`. Whisper round-trip only on: assembled
   conversation, lines that burned ≥2 attempts, 1-in-20 sample.

## Phase 2 — Contract edits (small, additive)

1. **Seed pinning is the keystone** (contract lane): MakeAPod stores per-line
   seeds from render #1 (`envelope.seed`) and always re-posts them; then
   editing line N re-renders ONLY lines N and N+1 (N+2 history unchanged ⇒
   cache hit). Without pinning, every revision re-renders everything.
2. **`speaker` accepted by `/v1/tts` and `/v1/batch` lines** (symmetry with
   dialogue; kills the hand-crafted-tag bug class).
3. **Per-line `gap_before_s` override** in dialogue (serves cadence at line
   seams without engine changes).
4. **Docs**: `/v1/dialogue` is the primary path (server docstring omits it!),
   README headline example = register two tagged voices → dialogue; `/v1/tts`
   = power tool. Document the already-working per-line temperature/seed/voice
   pass-through. Assembly ownership: client-side (server gap_s = preview).
5. P2: batch lines default to line mode? document instead; voice recipe PATCH;
   script persistence/async jobs deferred.

## Phase 3 — Cadence probe (one wav, q4, after restart)

8 lines: baseline / `[break]` / `[long-break]` / ellipsis / V1.6 `(break)` /
`<slow>that matters</slow>` / pacing brief `[measured pace, letting the point
land]` / full stop. Measure with whisper word-timestamps: pause = start("that")
− end("routine"); target 200-500 ms, no duration explosion. NOTE the lane
conflict this resolves: cadence lane wants `[break]` adopted if it works;
engine lane says build no pause tags (the `[sighs]` explosion, RAS pressure).
The probe decides; either way the AUGMENTER owns pause semantics, with
`gap_before_s` for seam control.

## Phase 3 — Cadence probe — DONE (2026-10-15, bf16→q4, seed 777, voice abvoice-b)

Measured pause (whisper word-timestamps, start("that") − end("routine")):
baseline comma 0 ms · <slow> 520 · ellipsis 560 · (break) 520 · pacing brief 640 ·
[break] 820 · full stop 1000 · [long-break] 1320. Verdict: graded pause ladder
works; subtle beat = <slow>/ellipsis (in 200-500 ms band ±); deliberate = [break];
scene shift = [long-break]/full stop. V1.6 (paren) cues bleed into Drama 3 —
recognized but not to be relied on. Baseline comma produced NO pause, confirming
the cadence complaint. Augmenter pause vocabulary = this ladder with measured
values; default to the subtle end. Probe wavs: outputs/probe-pause/.

## Phase 4 — v2 augmenter (`POST /v1/enhance`)

- Never touches MLX: LLM HTTP call + regex validators on the handler thread.
- Default LLM: LM Studio qwen3.8-27b @ 127.0.0.1:1234 (GLM-5.3 @ 8020 = quality
  override via env DRAMA3_ENHANCE_*; cap conversation length — KV growth is
  the memory risk).
- Request: lines (bare or tagged), tone_hints, intensity, existing_tags
  keep|rewrite. Response: same line shape + per-line rationale + warnings —
  output feeds `/v1/dialogue` verbatim.
- Director prompt ships verbatim (turn briefs physical/behavioral ≤10 words,
  small mid-sentence sounds, ≤2 paired tags/line, [pause] before key phrases,
  NEVER caps/new tags/speaker markers/word changes) + 7 validation layers:
  strip `<|...|>`, paired-tag allowlist (KNOWN_PAIRED_TAGS), bracket balance
  + max 2 paired tags + directions ≤80 chars, no line-start sound events,
  ALL-CAPS downcasing, word-preservation check w/ fallback, dry-run
  validate_text + check_tag_balance. LLM timeout ⇒ 503 with original lines.
- Build order: validators DONE (18 tests), endpoint DONE (live smoke vs
  Star Gate PoolModel: 15 s, validated output, rationales). LLM defaults
  switched to Star Gate PoolModel @ 127.0.0.1:30000 (user's pool endpoint;
  LM Studio qwen3.8-27b was a 60 s+ timeout trap on the director task).
- **OWNER RULE (2026-10-15, said twice, angry the second time): never cap
  the LLM's output. No max_tokens, no truncation, no output limits — the
  owner's own local models think as long as they need.
  DRAMA3_ENHANCE_TIMEOUT=0 = no timeout at all. Do not reintroduce caps.**
  (The 2048 cap burned the entire budget on reasoning_content and returned
  null content → crash; fixed by removing the cap + null-content guard →
  clean 503.)
- **Grin-probe (2026-10-15, q4, seed 43, abvoice-a, /v1/tts line mode): NULL
  RESULT.** [grinning, leaning in] at line start did NOT reproduce the manic
  laugh (4/4 takes fine, user-confirmed by ear). Numbers: bare 125 Hz/9 Hz
  swing · grin-start 174 Hz/22 (whole take re-keyed upward) · grin-mid 122
  Hz/40 (local effect only) · [an enthusiastic podcast introduction] 162
  Hz/100 (wildest — user says fine, ALIVE). Wavs: outputs/grin-probe/.
  Conclusion: grinning→laugh is NOT deterministic; the lighthouse line-0
  laugh was context-dependent. Candidate differentiators vs probe: (a)
  dialogue path prepends <|speaker:0|> tag, (b) <emphasis>the machines that
  build us</emphasis> in lighthouse text, (c) stochastic. REPRODUCTION TEST
  (pending): re-run exact lighthouse line-0 request (tag+emphasis+dialogue,
  seed 43); if laugh returns, bisect tag vs emphasis (2 renders); if not,
  treat as stochastic → fix = onset F0-swing gate (first-3s swing >~35 Hz →
  re-roll), not prompt surgery. Do not over-extrapolate from single takes
  (owner instruction).
- First full-wheel run (turn 1 of prompt tuning): 10-line lighthouse script
  enhanced in 129 s, rendered all-eos in 117 s → outputs/enhanced-lighthouse.wav
  (+ enhanced-lighthouse-text.json with rationales). PoolModel chose
  [break]/[long-break] with correct intent (measured ladder), <soft>/<emphasis>
  per emotional beat. Director prompt is a FIRST DRAFT — tuning targets:
  brief intensity (s43's over-the-top physical briefs are the bar the ≤10-word
  cap likely falls short of), sound-event placement, pause-vocabulary intent,
  under/over-direction calibration. REMAINING: listen loop → prompt tuning.

## Carry to MakeAPod (the client contract)

1. Register voices once (`/v1/voices`, reference transcripts tagged
   `<|speaker:N|>` matching line speaker ids).
2. Author bare script → `POST /v1/enhance` → review rationales → edited lines.
3. `POST /v1/dialogue` with lines `{line_id, voice, speaker, seed, text}`,
   `history_turns: 1`, `gap_s` preview only. Store `envelope.seed` per line.
4. Gate each line with the no-ears rule; re-roll failures via the ladder
   (`max_retries`) or explicit seed+10; assemble client-side (mono 44.1k,
   your junction flow); whisper-verify the assembled conversation.
5. Revisions: re-post full dialogue with pinned seeds + edited line N —
   cache makes everything but N and N+1 free.
6. Server should run q4 for iteration; bf16 is a final-render option only.
