"""Load a converted Drama 3 checkpoint into the MLX runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from .backbone import Backbone, build_backbone, strip_prefix
from .config import ModelConfig, load_config
from .module import Module
from .tower import Tower, build_tower

BACKBONE_PREFIX = "model.language_model."
TOWER_PREFIX = "audio_decoder"


@dataclass
class Drama3Model:
    config: ModelConfig
    backbone: Backbone
    tower: Tower
    tokenizer: Tokenizer
    bits: int

    @property
    def frame_width(self) -> int:
        return self.config.frame_width


def load_model(converted_dir: str | Path) -> Drama3Model:
    directory = Path(converted_dir)
    config = load_config(directory)
    weights: dict[str, mx.array] = dict(mx.load(str(directory / "weights.safetensors")))

    meta = {}
    if (directory / "conversion.json").exists():
        import json

        meta = json.loads((directory / "conversion.json").read_text())
    bits = int(meta.get("bits", 16))
    group_size = int(meta.get("group_size", 64))

    backbone_weights = {}
    for key, value in weights.items():
        if key.startswith(BACKBONE_PREFIX):
            backbone_weights["model." + key[len(BACKBONE_PREFIX):]] = value
        elif key == "lm_head.weight":
            backbone_weights[key] = value
    backbone = build_backbone(
        backbone_weights, config.text, group_size=group_size, bits=bits
    )

    tower_weights = {
        key: value for key, value in weights.items() if key.startswith(TOWER_PREFIX)
    }
    tower = build_tower(tower_weights, config)
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    return Drama3Model(
        config=config, backbone=backbone, tower=tower, tokenizer=tokenizer, bits=bits
    )


def count_parameters(model: Module) -> float:
    return sum(p.size for p in model.parameters().values()) / 1e9
