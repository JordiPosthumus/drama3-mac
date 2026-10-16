# Augmenter tuning log (goal: augment-tune)

Benchmark: 10-line lighthouse, voices abvoice-a/b, seed 43, q4, /v1/dialogue.
Scorer: scripts/augment_score.py (gates: G1 F0 bands host 105-150 / guest
150-220 · G2 [break] pause ≥200 ms via whisper+energy-fallback, no line-start
sound events · G3 eos+no-loop, duration 0.5-2.5× words/2.8 s, vr ≥0.35,
onset swing <110 Hz (calibrated: owner approved a 100 Hz-swing take as
"alive") · G4 word preservation).

## Baseline (v3 prompt: short-span tags rule + payoff-[break] rule)
7/10 lines pass. Failures:
- line 6: line-start [long-break] rendered 0 ms (ignored)
- line 1: [a dry chuckle] → 18.0 s (ratio 2.29), vr 0.28
- line 2: <whisper>+ellipsis on 3-word line → 2.3 s, vr 0.26 (dead air)
- line 5: whisper tail undecodable — energy fallback found the 1320 ms pause ✓
  (gate now passes; [break] sits AFTER "that matters," — placement intent
  still open, owner wants the beat BEFORE the payoff phrase)
Infrastructure fixes during baseline: whisper retry on empty decode;
break-alignment via fraction-seeded neighbor-word search (leading-space strip,
short-word exact match); energy-gap fallback for undecodable tails;
swing gate recalibrated 35→110 Hz against owner-approved data point.

## Cycle 1 — rule: NEVER [long-break] (prompt) + validator normalizes to [break]
Result: 5/10 (down from 7 — new failures outweighed the fix). 
- [long-break] eliminated everywhere ✓ (ban works)
- line 6 fixed ([gently] brief instead), line 1 fixed (7.9 s, vr 0.51 —
  chuckle moved position, no explosion), line 5's [break] measured 840 ms ✓
- NEW failure line 2: <soft>was it?</soft> on 3-word author line with author
  ellipsis → 14.7 s pitchless noise fill (f0=0, vr=1.0). Worst failure seen.
- NEW failure lines 7/9: swing 116/143 Hz where [break] sits directly next to
  <soft>/<a dry chuckle>; line 7's [break] pause = 0 ms.
- NEW failure line 4: host median 155 Hz (band ≤150) — [teasing]+<emphasis>
  re-keyed the host UP. Voice drift is not only a soft/slow phenomenon.
Learned: (a) short lines + added paired tags = noise fills (2nd occurrence);
(b) [break] adjacent to tags/cues destabilizes; (c) upward drift exists.
NOTE: server process must be restarted after prompt edits (Python imports at
startup) — cycle 1's first enhance ran against stale code, caught by output
identity, re-run after restart.

## Cycle 2 — rule: short lines (<8 words) get one brief max, no paired tags
Result: 7/10 with refined scorer (pause-aware duration: author ellipses no
longer count as rambling; vr measured over speech regions only; break pause
bounded 200-2500 ms). Line 2 fixed (vr 0.57). Line 3's 4.9 s break now
correctly fails. Remaining: line 7 [break]=0 ms (adjacent to <soft>), line 3
4940 ms break, line 9 swing 137 Hz.

## Cycle 3 — rule: [break] never adjacent to paired tags/cues
Result: 6/10. Adjacency rule did NOT help because [break] itself is broken:
line 0 exploded to a 22,760 ms silence (worst yet), lines 3/7/9 breaks
rendered 0 ms. [break] tally across cycles in dialogue path: ~50% vanish,
~50% explode (0 / 620 / 840 / 880 / 1080 / 1220 / 1660 ok vs 0 / 0 / 0 /
4940 / 22760). The ellipsis NEVER failed in any cycle (520-560 ms every
render, including inside author text).

## Cycle 4 — rule: [break]/[long-break] banned, normalized to "..." (prompt+validator)
Result: 7/10. Zero bracket breaks; LLM proposed no pauses at all (prompt was
descriptive, not directive about inserting ellipses). New failures: lines
0/3/7 swing 132-188 Hz — LLM shifted to stacking <emphasis> tags; line 0
host median 154 Hz (over 105-150 band). Guest voice ID holds everywhere
(172-197 Hz). vr healthy everywhere (0.47-0.71) — no noise fills, no dead
air, no explosions. The swing failures are expressive-by-design renders that
exceed the 110 Hz ceiling calibrated on the owner's approved 100 Hz take.

## Standing after cycle 4
- Reliability machinery works: no degeneracy class has recurred since its
  rule landed (noise fills, dead air, explosion pauses, [long-break] fills).
- Open: (a) swing ceiling vs expressiveness calibration — owner ears needed
  on cycle4 line 0; (b) host upward drift with heavy emphasis (line 0 154 Hz);
  (c) the "that matters" beat — [break] dead, so the pause must come from an
  author ellipsis or a directive prompt rule to insert "..." before payoff
  phrases; (d) [a dry chuckle] instability untested since cycle 1 (cycle 4
  line 1 passes with it mid-line after "counting").

## Cycle 5 — rule: directive ellipsis ("... where a natural reader would beat")
Result: 9/10. **The owner's beat landed**: "a routine... that matters" with
the pause verified at 700 ms (energy cross-check; whisper clamps same-segment
gaps — scorer infrastructure fix #4). Line 1 failed swing 176 Hz (<emphasis>
on the first word). Calibration applied from owner ears: swing ceiling
110→160 Hz, host band →105-165 (132 Hz-swing "nice pod intro", 154 Hz host
median approved).

## Cycle 6 — rule: no paired tag on a line's first word
Result: **10/10 — first full pass.** All F0 in band (host 118-133, guest
170-179), pauses 640-1100 ms including routine→that at 760 ms + <emphasis>that
matters</emphasis>, zero degeneracy, word preservation intact. Whisper
acceptance: per-line word-F1 mean 0.990, min 0.94 (bar 0.8), number-normalized.
Assembled conversation: outputs/cycle6-final.wav (93.4 s).
Handed to owner for the listen the goal reserves as the final gate.
