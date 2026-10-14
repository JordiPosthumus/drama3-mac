"""End-to-end Drama 3 engine: reference encode -> frames -> codec -> wav."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .chunking import (
    CHUNK_MAX_NEW_TOKENS,
    Drama3History,
    chunk_prompt,
    split_drama3_text,
    utf8_len,
    validate_chunks,
)
from .codec import decode_frames, encode_audio, load_codec
from .config import ModelConfig
from .generate import DEFAULT_MAX_NEW_TOKENS, GenerationOutput, generate
from .model import Drama3Model, load_model
from .prompt import MAX_PROMPT_ROWS, build_prompt

DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 30
DEFAULT_TOP_P = 0.9


@dataclass
class TTSResult:
    audio: np.ndarray  # float32 mono at the codec sample rate
    sample_rate: int
    generation: GenerationOutput
    transcript: str
    chunks: int = 1  # number of generated chunks (long-text path)
    chunk_main: tuple[np.ndarray, ...] = ()  # main-token ids per chunk (analysis)

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
        reference_audio: str | Path | np.ndarray
        | list[str | Path | np.ndarray] | None = None,
        reference_text: str | list[str] | None = None,
        reference_codes: np.ndarray | list[np.ndarray] | None = None,
        *,
        max_new_tokens: int = 0,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        single_chunk: bool = False,
        chunk_gap: float = 0.05,
        repetition_penalty: float = 1.0,
    ) -> TTSResult:
        """Synthesize ``text``; clone the voice of ``reference_audio`` when given.

        Multiple references are supported (a list of clips + transcripts); they
        are all written into the prompt as ``[Reference N]`` blocks, which
        measurably improves speaker adherence for hard-to-clone voices (a long
        3-take reference measured 0.939 speaker-embedding cosine vs 0.928 for
        the best single take).

        Text longer than one chunk (~300 UTF-8 bytes) is automatically split at
        sentence/paragraph boundaries and generated chunk by chunk, mirroring the
        fish-speech reference: the first chunk and the latest one are fed back as
        history so voice and pacing stay consistent, and each chunk may produce at
        most ~600 frames (~28 s) which bounds the runaway-loop degeneration seen
        on single-pass long generations.  ``single_chunk=True`` forces one pass
        (needs ``max_new_tokens`` big enough for the whole text).  ``max_new_tokens
        <= 0`` means auto: 2048 for a single pass, 600 per chunk otherwise.
        ``chunk_gap`` is silence in seconds inserted between chunk audios.

        ``repetition_penalty`` is an opt-in, off-reference experiment: the
        fish-speech engine carries the field (SDK default 1.2) but ignores it
        for Drama 3, relying on repetition-aware sampling instead.  The port
        implements it as a CTRL-style penalty over the last 64 main tokens
        (semantic codes) for fighting hot-sampling loops; 1.0 = off = faithful.
        """
        refs: list[np.ndarray]
        if reference_codes is None and reference_audio is not None:
            audio_list = (
                reference_audio if isinstance(reference_audio, list) else [reference_audio]
            )
            refs = [self.encode_reference(a) for a in audio_list]
        elif reference_codes is not None:
            refs = (
                reference_codes if isinstance(reference_codes, list) else [reference_codes]
            )
        else:
            refs = []
        if isinstance(reference_text, str):
            texts = [reference_text]
        else:
            texts = list(reference_text or [])
        if refs and len(texts) != len(refs):
            raise ValueError(
                f"cloning needs one transcript per reference ({len(refs)} clips, "
                f"{len(texts)} transcripts)"
            )
        prompt_texts = texts
        prompt_tokens = refs

        chunks = [text] if single_chunk else split_drama3_text(text) or [text]
        validate_chunks(chunks)  # fail before generating anything
        if max_new_tokens <= 0:
            max_new_tokens = (
                DEFAULT_MAX_NEW_TOKENS if len(chunks) == 1 else CHUNK_MAX_NEW_TOKENS
            )

        started = time.perf_counter()
        if len(chunks) == 1:
            prompt = build_prompt(
                self.model.tokenizer,
                self.config,
                text,
                prompt_texts=prompt_texts,
                prompt_tokens=prompt_tokens,
            )
            out = self._generate_chunk(
                prompt, max_new_tokens, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty,
            )
            if len(out.frames) == 0:
                raise RuntimeError("the model emitted EOS before any audio frame")
            audio = decode_frames(self.codec, out.codes, device=self.codec_device)
            return TTSResult(
                audio=audio,
                sample_rate=self.codec.sample_rate,
                generation=out,
                transcript=text,
                chunk_main=(out.frames[:, 0],),
            )

        # Long-text path: one fixed cache capacity for every chunk so the
        # compiled decode graphs are built once.  The bound must cover the
        # reference block, every chunk's text rows, up to two history turns
        # worth of audio rows (~600 frames each) and this chunk's generation.
        ref_rows = len(
            build_prompt(
                self.model.tokenizer,
                self.config,
                chunks[0],
                prompt_texts=prompt_texts,
                prompt_tokens=prompt_tokens,
            )
        )
        text_rows = sum(utf8_len(chunk) for chunk in chunks)  # >= 1 row per token
        capacity = min(
            MAX_PROMPT_ROWS,
            ref_rows + text_rows + 2 * CHUNK_MAX_NEW_TOKENS + max_new_tokens + 64,
        )
        history = Drama3History()
        audios: list[np.ndarray] = []
        mains: list[np.ndarray] = []
        frames_total = 0
        elapsed = 0.0
        all_eos = True
        for index, chunk in enumerate(chunks):
            prompt = chunk_prompt(
                chunk,
                self.model.tokenizer,
                self.config,
                prompt_texts,
                prompt_tokens,
                history,
            )
            budget = min(max_new_tokens, MAX_PROMPT_ROWS - len(prompt))
            if budget < 1:
                raise ValueError(
                    f"chunk {index}: prompt has {len(prompt)} rows; no room to generate"
                )
            out = self._generate_chunk(
                prompt, budget, temperature, top_k, top_p,
                cache_capacity=capacity,
                repetition_penalty=repetition_penalty,
            )
            elapsed += out.elapsed
            if len(out.frames) == 0:
                print(f"[tts] chunk {index}: EOS before any audio frame; skipped")
                continue
            all_eos &= out.finish_reason == "eos"
            frames_total += len(out.frames)
            audios.append(decode_frames(self.codec, out.codes, device=self.codec_device))
            mains.append(out.frames[:, 0])
            history.push(chunk, out.codes)
        if not audios:
            raise RuntimeError("the model emitted EOS before any audio frame")
        fade = min(128, self.codec.sample_rate // 200)  # ~5 ms edge fades kill
        for piece in audios:                            # codec seam transients
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            piece[:fade] *= ramp
            piece[-fade:] *= ramp[::-1]
        gap = np.zeros((int(chunk_gap * self.codec.sample_rate),), np.float32)
        parts: list[np.ndarray] = []
        for i, piece in enumerate(audios):
            if i:
                parts.append(gap)
            parts.append(piece)
        audio = np.concatenate(parts)
        generation = GenerationOutput(
            frames=np.zeros((frames_total, self.config.frame_width), np.int32),
            finish_reason="eos" if all_eos else "length",
            elapsed=elapsed,
        )
        return TTSResult(
            audio=audio,
            sample_rate=self.codec.sample_rate,
            generation=generation,
            transcript=text,
            chunks=len(audios),
            chunk_main=tuple(mains),
        )

    def _generate_chunk(
        self,
        prompt: np.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        cache_capacity: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> GenerationOutput:
        return generate(
            self.model.backbone,
            self.model.tower,
            self.config,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            cache_capacity=cache_capacity,
        )

    def save(self, result: TTSResult, path: str | Path) -> Path:
        path = Path(path)
        sf.write(str(path), result.audio, result.sample_rate)
        return path
