"""End-to-end Drama 3 engine: reference encode -> frames -> codec -> wav."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .codec import decode_frames, encode_audio, load_codec
from .config import ModelConfig
from .generate import GenerationOutput, generate
from .model import Drama3Model, load_model
from .prompt import build_prompt

DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 30
DEFAULT_TOP_P = 0.9


@dataclass
class TTSResult:
    audio: np.ndarray  # float32 mono at the codec sample rate
    sample_rate: int
    generation: GenerationOutput
    transcript: str

    @property
    def real_time_factor(self) -> float:
        audio_seconds = len(self.audio) / self.sample_rate
        return self.generation.elapsed / audio_seconds if audio_seconds else float("inf")


class Drama3Engine:
    """Loads the MLX model and the PyTorch codec; synthesizes speech on Apple Silicon."""

    def __init__(
        self,
        model_dir: str | Path,
        codec_path: str | Path | None = None,
        codec_device: str = "auto",
        codec_dtype: str = "float32",
    ):
        self.model_dir = str(model_dir)
        self.model: Drama3Model = load_model(model_dir)
        self.config: ModelConfig = self.model.config
        if codec_path is None:
            codec_path = Path(model_dir).parent.parent / "model" / "codec.pth"
            if not codec_path.exists():
                codec_path = Path(model_dir) / "codec.pth"
        import torch

        if codec_device == "auto":
            codec_device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[codec_dtype]
        self.codec = load_codec(codec_path, device=codec_device, dtype=dtype)
        self.codec_device = codec_device

    def encode_reference(self, audio: str | Path | np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """wav file or float32 array -> codes [10, T]."""
        if isinstance(audio, np.ndarray):
            wav = audio
            sr = sample_rate
        else:
            wav, sr = sf.read(str(audio), dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
        return encode_audio(self.codec, wav, sr, device=self.codec_device)

    def tts(
        self,
        text: str,
        reference_audio: str | Path | np.ndarray | None = None,
        reference_text: str | None = None,
        reference_codes: np.ndarray | None = None,
        max_new_tokens: int = 2048,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
    ) -> TTSResult:
        """Synthesize `text`; clone the voice of `reference_audio` when given."""
        if reference_codes is None and reference_audio is not None:
            reference_codes = self.encode_reference(reference_audio)
        if reference_codes is not None and not reference_text:
            raise ValueError("cloning needs the reference transcript too")
        prompt_texts = [reference_text] if reference_codes is not None else []
        prompt_tokens = [reference_codes] if reference_codes is not None else []
        prompt = build_prompt(
            self.model.tokenizer,
            self.config,
            text,
            prompt_texts=prompt_texts,
            prompt_tokens=prompt_tokens,
        )
        out = generate(
            self.model.backbone,
            self.model.tower,
            self.config,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if len(out.frames) == 0:
            raise RuntimeError("the model emitted EOS before any audio frame")
        audio = decode_frames(self.codec, out.codes, device=self.codec_device)
        return TTSResult(
            audio=audio,
            sample_rate=self.codec.sample_rate,
            generation=out,
            transcript=text,
        )

    def save(self, result: TTSResult, path: str | Path) -> Path:
        path = Path(path)
        sf.write(str(path), result.audio, result.sample_rate)
        return path
