"""Real-model regression for the 1-row prefill chunk crash.

Builds a prompt whose row count is 1 mod 128 (so the final prefill chunk is a
single row with continue_state=True) and runs the chunked prefill + one
logits call.  Before the fix this raised ValueError('decode step needs the
position').  Needs a converted checkpoint; ~2 minutes.

Run: PYTHONPATH=. python scripts/regress_1row_prefill.py
"""

import mlx.core as mx, numpy as np
from drama3_mac.model import load_model
from drama3_mac.prompt import build_prompt
from drama3_mac.config import load_config

cfg = load_config("checkpoints/drama3-q4")
tok = __import__("tokenizers").Tokenizer.from_file("model/tokenizer.json")
# build a prompt with 129 rows: reference-like audio rows + text rows
from drama3_mac.codec import load_codec, encode_audio
codec = load_codec("model/codec.pth", device="mps")
wav = np.zeros((44100 * 4,), np.float32)  # 4s silence -> 86 frames
codes = encode_audio(codec, wav, 44100, device="mps")
for rep in range(1, 60):
    text = "One two three four five six seven eight nine ten. " * rep
    prompt = build_prompt(tok, cfg, text, prompt_texts=["Reference line."], prompt_tokens=[codes])
    if len(prompt) % 128 == 1:
        break
print("prompt rows:", len(prompt), "rows % 128 =", len(prompt) % 128)
assert len(prompt) % 128 == 1, "need the 1-row tail chunk shape"

m = load_model("checkpoints/drama3-q4")
bb, tw = m.backbone, m.tower
R = len(prompt)
bb.setup_caches(1, R + 8); tw.setup_caches(R + 8)
frames = mx.zeros((R + 8, 11), mx.int32); frames[:R] = prompt
from drama3_mac.generate import PREFILL_CHUNK_SIZE
for start in range(0, R, PREFILL_CHUNK_SIZE):
    end = min(start + PREFILL_CHUNK_SIZE, R)
    emb = tw.embed_backbone(frames[start:end], bb.embed(frames[start:end, 0])[None])
    hidden = bb(emb, start=start, kv_len=end, continue_state=start > 0, use_cache=True)
    paired_end = min(end, R - 1)
    if paired_end > start:
        tw.prefill(hidden[0][: paired_end - start], mx.arange(start, paired_end),
                   frames[start:paired_end], frames[start + 1:paired_end + 1])
    mx.eval(hidden)
hidden = hidden[0][-1:]
logits = bb.compute_logits(hidden)
mx.eval(logits)
print("no crash; logits finite:", bool(mx.isfinite(logits).all()), "shape:", logits.shape)
