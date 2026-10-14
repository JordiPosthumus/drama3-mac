"""End-to-end Drama 3 engine: reference encode -> frames -> codec -> wav."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

import mlx.core as mx

from .chunking import (
    CHUNK_BYTES,
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
from .guardrails import (
    Finding,
    PreflightError,
    check_capacity,
    check_reference,
    check_tag_balance,
    run_postflight,
)
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
    warnings: tuple[str, ...] = ()  # postflight findings, "[WARN] check: ..."
    finish_reasons: tuple[str, ...] = ()  # per-chunk "eos" | "length"

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

        from .guardrails import check_checkpoint

        codec_failures = [
            f for f in check_checkpoint(model_dir, codec_path) if f.level == "FAIL"
        ]
        if codec_failures:
            raise PreflightError(
                "; ".join(str(f) for f in codec_failures)
            )

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
        sampling_plan: list[tuple[float, float]] | None = None,
        seed: int | None = None,
        edge_fades: bool = True,
        chunk_bytes: int | None = None,
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

        ``sampling_plan`` optionally gives per-chunk ``(temperature, top_p)``
        pairs — the measured hybrid recipe cools narration (0.7, 0.85) and
        heats emotional peaks (0.8, 0.9); see README.  Its length must match
        the number of chunks.

        Preflight guardrails refuse bad jobs before generation (missing
        checkpoint files, unbalanced delivery tags, prompt-capacity overflow,
        broken reference clips — see guardrails.py); postflight findings are
        returned on ``TTSResult.warnings``.

        ``seed`` reseeds MLX's RNG before sampling: the same seed with the
        same text/params/voice reproduces a bit-identical take.  ``None``
        leaves the RNG state alone (non-deterministic across calls).

        ``edge_fades=False`` skips the ~5 ms fades applied to chunk edges
        (they kill codec seam transients); line-mode callers whose assembler
        owns junction flow want raw segments.  ``chunk_bytes`` overrides the
        ~300-byte auto-chunk target (the retry ladder resplits with it).
        """
        refs: list[np.ndarray] = []
        waveforms: list[tuple[np.ndarray, int]] = []  # for reference checks
        if reference_codes is None and reference_audio is not None:
            audio_list = (
                reference_audio if isinstance(reference_audio, list) else [reference_audio]
            )
            for a in audio_list:
                if isinstance(a, np.ndarray):
                    waveforms.append((a, 44100))
                    refs.append(self.encode_reference(a))
                else:
                    wav, sr = sf.read(str(a), dtype="float32", always_2d=True)
                    mono = wav.mean(axis=1)
                    waveforms.append((mono, sr))
                    refs.append(self.encode_reference(mono, sr))
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

        chunks = [
            text
            if single_chunk
            else split_drama3_text(text, max_bytes=chunk_bytes or CHUNK_BYTES) or [text]
        ]
        validate_chunks(chunks)  # fail before generating anything
        if max_new_tokens <= 0:
            max_new_tokens = (
                DEFAULT_MAX_NEW_TOKENS if len(chunks) == 1 else CHUNK_MAX_NEW_TOKENS
            )

        # ---- preflight guardrails: refuse dud jobs before any GPU work ----
        findings: list[Finding] = []
        for failure in check_tag_balance(text):
            if failure.level == "FAIL":
                raise PreflightError(str(failure))
            findings.append(failure)
        for wav, sr in waveforms:
            findings += check_reference(wav, sr)
        ref_frames = sum(codes.shape[-1] for codes in refs)
        if len(chunks) == 1:
            findings += check_capacity(
                text, ref_frames, utf8_len=utf8_len,
                max_new_tokens=max_new_tokens,
            )
        else:
            largest = max(chunks, key=utf8_len)
            findings += check_capacity(
                largest, ref_frames, utf8_len=utf8_len,
                max_new_tokens=min(max_new_tokens, CHUNK_MAX_NEW_TOKENS),
                history_rows=2 * CHUNK_MAX_NEW_TOKENS,
            )
        if sampling_plan is not None:
            plan = [tuple(p) for p in sampling_plan]
            if len(plan) != len(chunks):
                raise PreflightError(
                    f"[FAIL] sampling-plan: {len(plan)} (temperature, top_p) "
                    f"pairs for {len(chunks)} chunk(s); lengths must match"
                )
        else:
            plan = None

        started = time.perf_counter()
        if seed is not None:
            mx.random.seed(seed)
        if len(chunks) == 1:
            gen_temperature, gen_top_p = (
                plan[0] if plan is not None else (temperature, top_p)
            )
            prompt = build_prompt(
                self.model.tokenizer,
                self.config,
                text,
                prompt_texts=prompt_texts,
                prompt_tokens=prompt_tokens,
            )
            out = self._generate_chunk(
                prompt, max_new_tokens, gen_temperature, top_k, gen_top_p,
                repetition_penalty=repetition_penalty,
            )
            if len(out.frames) == 0:
                raise RuntimeError("the model emitted EOS before any audio frame")
            audio = decode_frames(self.codec, out.codes, device=self.codec_device)
            findings += run_postflight(
                [out.finish_reason], [len(out.frames)], [out.frames[:, 0]]
            )
            return TTSResult(
                audio=audio,
                sample_rate=self.codec.sample_rate,
                generation=out,
                transcript=text,
                chunk_main=(out.frames[:, 0],),
                warnings=tuple(str(f) for f in findings),
                finish_reasons=(out.finish_reason,),
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
        finish_reasons: list[str] = []
        frames_per_chunk: list[int] = []
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
            chunk_temperature, chunk_top_p = (
                plan[index] if plan is not None else (temperature, top_p)
            )
            out = self._generate_chunk(
                prompt, budget, chunk_temperature, top_k, chunk_top_p,
                cache_capacity=capacity,
                repetition_penalty=repetition_penalty,
            )
            elapsed += out.elapsed
            if len(out.frames) == 0:
                print(f"[tts] chunk {index}: EOS before any audio frame; skipped")
                continue
            all_eos &= out.finish_reason == "eos"
            frames_total += len(out.frames)
            finish_reasons.append(out.finish_reason)
            frames_per_chunk.append(len(out.frames))
            audios.append(decode_frames(self.codec, out.codes, device=self.codec_device))
            mains.append(out.frames[:, 0])
            history.push(chunk, out.codes)
        if not audios:
            raise RuntimeError("the model emitted EOS before any audio frame")
        findings += run_postflight(finish_reasons, frames_per_chunk, mains)
        if edge_fades:
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
            warnings=tuple(str(f) for f in findings),
            finish_reasons=tuple(finish_reasons),
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
