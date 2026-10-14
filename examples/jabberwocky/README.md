# Jabberwocky — the theatric tag showcase

A six-part performance of Lewis Carroll's *Jabberwocky* (public domain) that
exercises the full Drama 3 direction system through this port: hushed
storytelling, whispered warnings, proud resolve, shrieking terror, and a
gleeful finale.

## Run it

```bash
# automatic: the script is split into chunks at sentence boundaries and
# stitched with small gaps
drama3 synthesize "$(cat examples/jabberwocky/script.txt)" \
    -o jabberwocky.wav \
    --ref your-voice.wav --ref-text "Transcript of your reference clip."

# or exactly as directed: generate the six hand-split chunks yourself
drama3 synthesize "$(cat examples/jabberwocky/directed-chunks/chunk-1.txt)" -o jab1.wav --ref ...
# (repeat per chunk; see directed-chunks/)
```

## What each part demonstrates

| part | direction focus | tags used |
|---|---|---|
| chunk-1 | hushed campfire-story opening | `[hushed, conspiratorial storyteller, almost a whisper]` |
| chunk-2 | urgent warning | `<whisper>…</whisper>`, `[suddenly urgent, warning]`, `[stern, trembling]` |
| chunk-3 | resolute hero | `[proud, resolute, gathering courage]` |
| chunk-4 | dread building | `[hushed dread, building slowly]` |
| chunk-5 | the battle | ALL CAPS shouting, `[shrieking terror]`, `[exhausted, triumphant]` |
| chunk-6 | joy + bookend | `[bursting with joy, shouting with glee]`, `[gleeful laugh]`, `[chortling]`, `[hushed wonder, soft fade]` |

## Tag syntax

- **[square-bracket cues]** — free-form stage directions, sound effects and
  pauses: `[pause]`, `[long pause]`, `[small laugh]`, `[door creaks]`. The
  model interprets them; they are not spoken.
- **`<tag>…</tag>` paired tags** change the delivery of the enclosed span:
  `whisper`, `emphasis`, `soft`, `fast`, `slow`, `stress`, `forceful`.
- Cues and paired tags are **protected spans**: automatic long-text chunking
  never splits them across chunks.
- Speaker changes: `<|speaker:0|>` … `<|speaker:15|>` prefix a line; the
  chunker keeps at most 3 speakers per chunk.

Long theatric scripts **must** be chunked (the default): a single 142 s pass
degenerates into loops; chunked generation of the same script round-trips
cleanly through Whisper. See `docs/JOURNEY.md` for the evidence.
