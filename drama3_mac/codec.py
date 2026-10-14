"""Loader and helpers for the Drama 3 codec (vendored in ``codec_dac.py``).

The codec converts 10-codebook frames at 21.53 Hz (44100 / 512 hop / 4x
quantizer downsample) to 44.1 kHz audio, and reference audio back to codes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from functools import partial

from .codec_dac import DAC, DownsampleResidualVectorQuantize, ModelArgs

FRAME_RATE = 44100 / 512 / 4  # 21.53 Hz


def _window_module() -> "WindowLimitedTransformer":
    from .codec_dac import ModelArgs as _MA, WindowLimitedTransformer

    config = _MA(
        block_size=2048,
        n_layer=8,
        n_head=16,
        dim=1024,
        intermediate_size=3072,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )
    return WindowLimitedTransformer(
        causal=True, window_size=128, input_dim=1024, config=config
    )


def build_codec() -> DAC:
    """Drama 3 ships the S2-Pro architecture codec with a 4096-entry semantic codebook."""
    general = partial(
        ModelArgs,
        block_size=8192,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.5,
        semantic_codebook_size=4096,
        downsample_factor=(2, 2),
        pre_module=_window_module(),
        post_module=_window_module(),
    )
    return DAC(
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        decoder_dim=1536,
        decoder_rates=[8, 8, 4, 2],
        quantizer=quantizer,
        sample_rate=44100,
        causal=True,
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=general,
    )


def load_codec(checkpoint: str | Path, device: str = "cpu", dtype=torch.float32) -> DAC:
    codec = build_codec()
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any("generator." in k for k in state):
        state = {k.replace("generator.", ""): v for k, v in state.items() if "generator." in k}
    non_persistent = {
        f"{module}.{buffer}"
        for module, mod in codec.named_modules()
        for buffer in mod._non_persistent_buffers_set
    }
    state = {k: v for k, v in state.items() if k not in non_persistent}
    result = codec.load_state_dict(state, strict=True)
    codec.eval()
    for module in list(codec.modules()):
        if is_parametrized(module, "weight"):
            remove_parametrizations(module, "weight", leave_parametrized=True)
        elif hasattr(module, "weight_g"):
            torch.nn.utils.remove_weight_norm(module)
    codec.to(device=device, dtype=dtype)
    return codec


@torch.inference_mode()
def encode_audio(codec: DAC, wav: np.ndarray, sample_rate: int, device: str = "cpu") -> np.ndarray:
    """wav float32 mono -> codes [10, T] int64 at the codec frame rate."""
    import soxr

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sample_rate != codec.sample_rate:
        wav = soxr.resample(wav, sample_rate, codec.sample_rate)
    audio = torch.from_numpy(wav.astype(np.float32))[None, None].to(device)
    model_dtype = next(codec.parameters()).dtype
    audio = audio.to(model_dtype)
    lengths = torch.tensor([audio.shape[-1]], device=device, dtype=torch.long)
    indices, feature_lengths = codec.encode(audio, lengths)
    return indices[0, :, : feature_lengths[0]].cpu().numpy()


@torch.inference_mode()
def decode_frames(codec: DAC, codes: np.ndarray, device: str = "cpu") -> np.ndarray:
    """codes [10, T] int -> float32 wav at codec.sample_rate."""
    indices = torch.from_numpy(np.ascontiguousarray(codes.astype(np.int64)))[None].to(device)
    audio = codec.from_indices(indices)
    return audio[0, 0].float().cpu().numpy()
