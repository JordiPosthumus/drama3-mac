"""Convert the fishaudio/drama-3 checkpoint into a fast MLX checkpoint.

The HF release stores MoE experts as block-FP8 (``bsgemm-moe``, 128x128 blocks
with fp32 scales); MLX has no fp8 path, and dequantizing on every launch would
waste the M-series memory bandwidth.  This pass runs once per precision:

* experts are dequantized to bf16, stacked per layer (``w13`` = gate rows then
  up rows, ``w2`` = down), and quantized to 4/8-bit (or kept bf16 with ``--bits 16``)
* attention / linear-attention projections are quantized the same way
* embeddings, the lm head, the router, norms and the audio tower stay bf16
  (the lm head and logits need the range; everything else is tiny)

Usage::

    python -m drama3_mac.convert --checkpoint model --bits 4 --out checkpoints/drama3-q4
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

HIDDEN = "model.language_model."
EXPERTS = (
    ("gate_proj", 0),
    ("up_proj", 1),
    ("down_proj", -1),
)
QUANTIZED_PROJ_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
)
COPY_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


class ShardReader:
    """Lazy per-key reader over the checkpoint shards, one mmap at a time."""

    def __init__(self, checkpoint: Path) -> None:
        index = checkpoint / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
        else:
            weight_map = {"model.safetensors": "model.safetensors"}
        self.checkpoint = checkpoint
        self.weight_map = weight_map
        self._open_path: str | None = None
        self._handle: safe_open | None = None

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard != self._open_path:
            self._handle = safe_open(str(self.checkpoint / shard), framework="pt")
            self._open_path = shard
        return self._handle.get_tensor(key)

    def close(self) -> None:
        self._handle = None
        self._open_path = None


def dequantize_fp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Block-FP8 [.., rows, cols] with fp32 scales [.., rows/128, cols/128] -> fp32."""
    rows, cols = weight.shape[-2:]
    row_blocks, col_blocks = scale_inv.shape[-2:]
    assert row_blocks * 128 == rows and col_blocks * 128 == cols
    blocks = weight.view(*weight.shape[:-2], row_blocks, 128, col_blocks, 128)
    scaled = blocks.to(torch.float32) * scale_inv[..., :, None, :, None]
    return scaled.view(*weight.shape[:-2], rows, cols)


def to_mx(tensor: torch.Tensor, dtype=mx.bfloat16) -> mx.array:
    return mx.array(tensor.to(torch.float32).numpy()).astype(dtype)


def quantize(mx_weight: mx.array, bits: int, group_size: int):
    packed, scales, biases = mx.quantize(
        mx_weight.astype(mx.bfloat16), group_size=group_size, bits=bits
    )
    return {
        "weight": packed,
        "scales": scales,
        "biases": biases,
    }


def layer_experts(
    reader: ShardReader, prefix: str, num_experts: int, intermediate: int, hidden: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantized stacked experts: w13 [E, 2I, H] (gate rows first) and w2 [E, H, I]."""
    w13 = torch.empty((num_experts, 2 * intermediate, hidden), dtype=torch.bfloat16)
    w2 = torch.empty((num_experts, hidden, intermediate), dtype=torch.bfloat16)
    for expert in range(num_experts):
        for proj, half in EXPERTS:
            weight = reader.get(f"{prefix}.experts.{expert}.{proj}.weight")
            scale = reader.get(f"{prefix}.experts.{expert}.{proj}.weight_scale_inv")
            value = dequantize_fp8(weight, scale).to(torch.bfloat16)
            if half == -1:
                w2[expert] = value
            elif half == 0:
                w13[expert, :intermediate] = value
            else:
                w13[expert, intermediate:] = value
    return w13, w2


def convert(
    checkpoint: Path,
    out: Path,
    *,
    bits: int,
    group_size: int = 64,
) -> None:
    started = time.perf_counter()
    config = json.loads((checkpoint / "config.json").read_text())
    text = config["text_config"]
    num_layers = text["num_hidden_layers"]
    num_experts = text["num_experts"]
    hidden = text["hidden_size"]
    moe_intermediate = text["moe_intermediate_size"]
    if config.get("quantization_config", {}).get("quant_method") != "bsgemm-moe":
        raise SystemExit("this converter expects the FP8 bsgemm-moe release")
    for size in (hidden, moe_intermediate):
        if size % group_size:
            raise SystemExit(f"group_size {group_size} does not divide {size}")

    reader = ShardReader(checkpoint)
    weights: dict[str, mx.array] = {}
    quantized = bits in (4, 8)

    def put(name: str, tensor: torch.Tensor, dtype=mx.bfloat16) -> None:
        weights[name] = to_mx(tensor, dtype)

    def put_linear(name: str, tensor: torch.Tensor) -> None:
        if quantized:
            weights.update(
                (f"{name}.{k}", v)
                for k, v in quantize(to_mx(tensor), bits, group_size).items()
            )
        else:
            put(name + ".weight", tensor)

    print(f"[convert] {num_layers} layers, bits={bits}, group={group_size}")
    for layer in range(num_layers):
        base = f"{HIDDEN}layers.{layer}"
        w13, w2 = layer_experts(
            reader, f"{base}.mlp", num_experts, moe_intermediate, hidden
        )
        if quantized:
            weights.update(
                (f"{base}.mlp.experts.w13.{k}", v)
                for k, v in quantize(to_mx(w13), bits, group_size).items()
            )
            weights.update(
                (f"{base}.mlp.experts.w2.{k}", v)
                for k, v in quantize(to_mx(w2), bits, group_size).items()
            )
        else:
            put(f"{base}.mlp.experts.w13.weight", w13)
            put(f"{base}.mlp.experts.w2.weight", w2)
        del w13, w2

        for suffix in QUANTIZED_PROJ_SUFFIXES:
            key = f"{base}.{suffix}"
            if key in reader.weight_map:
                put_linear(key[: -len(".weight")], reader.get(key))

        for name in (
            "mlp.gate.weight",
            "mlp.shared_expert_gate.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.norm.weight",
            "linear_attn.conv1d.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        ):
            key = f"{base}.{name}"
            if key in reader.weight_map:
                put(key, reader.get(key))
        for name in ("linear_attn.dt_bias", "linear_attn.A_log"):
            key = f"{base}.{name}"
            if key in reader.weight_map:
                put(key, reader.get(key), mx.float32)
        print(f"[convert] layer {layer + 1}/{num_layers} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    put(f"{HIDDEN}embed_tokens.weight", reader.get(f"{HIDDEN}embed_tokens.weight"))
    put(f"{HIDDEN}norm.weight", reader.get(f"{HIDDEN}norm.weight"))
    put("lm_head.weight", reader.get("lm_head.weight"))

    tower_keys = [k for k in reader.weight_map if k.startswith("audio_decoder")]
    for key in tower_keys:
        put(key, reader.get(key))
    print(f"[convert] tower copied ({len(tower_keys)} tensors)")
    reader.close()

    out.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        if (checkpoint / name).exists():
            shutil.copy2(checkpoint / name, out / name)
    mx.save_safetensors(str(out / "weights.safetensors"), weights)
    meta = {
        "source": str(checkpoint),
        "bits": bits,
        "group_size": group_size,
        "num_layers": num_layers,
        "tensors": len(weights),
    }
    (out / "conversion.json").write_text(json.dumps(meta, indent=2))
    total = sum(w.nbytes for w in weights.values()) / 1e9
    print(f"[convert] wrote {out} ({total:.1f} GB) in {time.perf_counter() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    args = parser.parse_args()
    convert(args.checkpoint, args.out, bits=args.bits, group_size=args.group_size)


if __name__ == "__main__":
    main()
