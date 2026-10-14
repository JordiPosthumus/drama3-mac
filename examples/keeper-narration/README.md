# Chunk-consistency demo — a 91 s story from 3 automatic chunks

`consistency.wav` renders [`narration.txt`](narration.txt) (an original short
story, ~1.2 KB) through the engine's automatic long-text chunking: the text is
split at sentence boundaries into 3 chunks, each generated with the same
voice, and stitched with 50 ms gaps and edge fades.

Rendered with the **cool narration recipe** — temperature 0.7, top-p 0.85,
~30 s multi-take reference:

```bash
drama3 synthesize "$(cat examples/keeper-narration/narration.txt)" \
    -o consistency.wav \
    --ref your-reference.wav --ref-text "Transcript of the reference." \
    --temperature 0.7 --top-p 0.85
```

91.2 s of audio from 89.4 s of generation (RTF 0.98, faster than real time
including chunking). The point of the demo: the keeper's voice, pacing and
energy are the same in chunk 3 as in chunk 1 — chunk seams are inaudible at
sentence boundaries, and the whole script round-trips through Whisper
medium.en verbatim.
