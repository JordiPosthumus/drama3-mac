"""Steady-state decode benchmark: warm up, then measure frames/s and step phases."""

import sys
import time

import mlx.core as mx
import numpy as np

from drama3_mac.model import load_model
from drama3_mac.prompt import build_prompt

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/drama3-q4"
N_WARMUP = 32
N_MEASURE = 256

m = load_model(MODEL_DIR)
prompt = build_prompt(
    m.tokenizer, m.config, "This is a benchmark sentence for the decode loop."
)
R = len(prompt)
bb, tw = m.backbone, m.tower
cap = R + N_WARMUP + N_MEASURE
bb.setup_caches(1, cap)
tw.setup_caches(cap)
frames = mx.zeros((cap, prompt.shape[1]), mx.int32)
frames[:R] = mx.array(prompt)

t0 = time.perf_counter()
end = R
emb = tw.embed_backbone(frames[:end], bb.embed(frames[:end, 0])[None])
hidden = bb(emb, start=0, kv_len=end, continue_state=False, use_cache=True)
tw.prefill(hidden[0], mx.arange(end), frames[:end], frames[1:end + 1])
mx.eval(hidden)
hidden = hidden[0][-1:]
print(f"prefill {R} rows in {time.perf_counter() - t0:.2f}s")

times = []
for step in range(N_WARMUP + N_MEASURE):
    position = R - 1 + step
    t0 = time.perf_counter()
    logits = bb.compute_logits(hidden)
    sampled = tw.decode_step(
        logits, hidden, mx.array([position]), frames[position:position + 1],
        frames[: position + 1, 0], temperature=0.8, top_k=30, top_p=0.9,
    )
    emb = tw.embed_backbone(sampled, bb.embed(sampled[:, 0])[None])
    hidden = bb(emb, start=position + 1, kv_len=position + 2, continue_state=True, use_cache=True,
                position=mx.array(position + 1, mx.int32))[0]
    mx.eval(sampled, hidden)
    dt = time.perf_counter() - t0
    times.append(dt)
    if step == N_WARMUP - 1:
        print(f"warmup done ({N_WARMUP} steps)")

times = np.array(times[N_WARMUP:])
print(f"steady-state: {len(times)} steps, mean {times.mean()*1000:.1f} ms, "
      f"p50 {np.percentile(times, 50)*1000:.1f}, p90 {np.percentile(times, 90)*1000:.1f}")
print(f"frames/s: {1/times.mean():.1f} (need 21.5 for real-time)")
