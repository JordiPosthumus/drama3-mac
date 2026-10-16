# Audit: native Drama 3 capabilities (upstream)

Sources: speech.fish.audio/inference + /install (scraped 2026-10-14), HF model card
fishaudio/drama-3 (scraped), SGLang-Omni Drama 3 Cookbook (scraped), upstream source
raw.githubusercontent.com/fishaudio/fish-speech/main: fish_omni/{generate,chunking,tower,model}.py,
tools/api_server.py.

## Positioning
- "Built for multi-character scenes and long-form content" (HF model card). Multi-speaker
  generation, voice cloning, and **natural-language control** are headline features.
- 35B MoE (3B active), 10-codebook MTP audio decoder, FP8 experts + bf16; official int4
  checkpoint exists upstream (37.0 -> 20.8 GB) — quantization is mainstream, not a port hack.

## Speaker + text control
- Speaker tags `<|speaker:0|>`..`<|speaker:15|>` decide the role; "reference order does not
  matter"; one text can carry up to 16 characters (speaker ids). References are auto-tagged
  speaker 0 unless their transcript carries a tag; the tag must follow `Text: ` directly
  (fish_omni/generate.py build_prompt).
- **Point tags** (HF model card, "Example Usage"): inline bracketed events land where written:
  `"[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair."`
  First-class official expressiveness mechanism. Consistent with the chunker's protected spans
  (paired tags like `<whisper>...</whisper>`, `[bracketed]` cues are never split —
  fish_omni/chunking.py PROTECTED_SPAN), i.e. delivery tags inside long text survive chunking.
- GitHub issue #1280 ("bracket emotion control appears weak on emotion-neutral transcripts",
  S2-Pro) confirms bracket emotion control is a known, imperfect feature.

## Multi-turn / context mechanisms
- Long text: auto-chunk ~300 UTF-8 bytes, max 3 speakers per chunk, cuts at sentence ends;
  first + latest chunk fed back as user text + assistant audio (fish_omni/generate.py
  `history: Sequence[tuple[str, Tensor]]`, "written as completed user and assistant voice turns").
- **Editing as continuation** (HF card + cookbook): `previous_takes` — "just send the previous
  VQ tokens ... as normal language models do"; chains of prior takes stay in context. This is
  the official multi-turn audio-context API on SGLang-Omni.
- Limits: max_new_tokens 1024/chunk default (0 => 600, single_chunk 2048 ~95 s), hard cap
  10,240 frames minus prompt; references max 60 s each (SGLang-Omni, HTTP 400 above).

## Sampling
- Docs page: defaults temperature 0.7 / top-p 0.7 / top-k 30. repetition_penalty accepted but
  ignored — built-in repetition-aware sampling (tower.py: window 16 frames, retry draw at
  1.0/30/0.95, semantic tokens only, window spans the prompt boundary).
- HF model card production bench: temperature **1.0**, top_p 0.9, top_k 30, max_new_tokens 500.
  Fish publishes different sampling for different contexts — defaults are content-dependent.

## Serving surface (upstream)
- Official: msgpack POST /v1/tts, /v1/tts/defaults, streaming, formats wav/pcm/mp3/opus,
  references CRUD (form upload), WebUI at /ui, api-key auth, --max-text-length.
- SGLang-Omni: /v1/audio/speech with stream:true, pcm streaming, per-stage GPU placement.
- Runtime: NVIDIA-only, FP8 experts, CUDA graphs, multi-GPU with codec on last device.

## Expressive-dialogue mechanisms found upstream (summary)
1. Point tags ([sighs], [laughs], <whisper>...</whisper>) — inline, chunker-protected.
2. Speaker tags + multi-reference prompting (role binding).
3. Multi-turn audio context: history turns / previous_takes with VQ tokens — the only
   mechanism by which a generated line can *react* to a heard line. Exposed by SGLang-Omni
   (previous_takes); present in fish_omni.generate (history param); NOT exposed by the
   official CLI/Python inference docs.
4. Reference-clip expressiveness: cloning follows the reference take's delivery (multi-take
   references measurably improve adherence — port measurement, 0.939 vs 0.928 cosine).
No SSML, no emotion parameter, no style prompt is documented anywhere upstream.
