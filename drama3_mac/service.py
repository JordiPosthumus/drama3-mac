"""Production TTS service: the logic behind the HTTP API.

``TtsService`` sits between the HTTP layer and ``Drama3Engine`` and provides
everything the podcast pipeline needs:

- **voice registry integration** — renders accept ``voice=name``; the voice's
  recipe fills sampling defaults, explicit request parameters win;
- **JSON envelope** — {audio_b64, duration_s, rms, f0_hz_median,
  finish_reason, rtf, seed, request_echo, attempts, warnings, cache, ok};
- **seeds + line cache** — ``seed`` flows to ``engine.tts`` (same seed +
  params = bit-identical take); results are cached under
  hash(voice, text, params, seed) with GET /v1/cache stats;
- **auto-retry ladder** — on finish=length for short-ish text the service
  re-renders (resplitting into smaller chunks, or reseeding in line mode) up
  to 2 retries and only returns EOS-clean audio; ``retry=false`` opts out;
- **line mode** — ``line=true`` renders one clean segment: no internal
  chunking, no baked gaps or edge fades (the consumer's assembler owns
  junction flow);
- **batch** — ``batch(lines)`` renders sequentially through one warm model,
  one ``mx.clear_cache()`` per line, returning a per-line envelope array.

All engine calls go through the injected ``run`` callable (the server passes
its main-thread runner; tests pass a direct call or a fake engine).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
from collections import OrderedDict
from typing import Callable

import numpy as np
import soundfile as sf

from . import metrics
from .guardrails import PreflightError
from .voices import VoiceError, VoiceRegistry

LADDER_MAX_RETRIES = 2
LADDER_MAX_BYTES = 1200  # "short-ish": the ladder resplits; long texts skip it
LADDER_CHUNK_BYTES = (150, 75)  # resplit targets for retry 1 / retry 2
CACHE_MAX_ENTRIES = 128


class RenderError(RuntimeError):
    """The render could not produce EOS-clean audio after the ladder."""


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """float32 mono -> 16-bit PCM wav bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def aggregate_finish_reason(finish_reasons: tuple[str, ...] | list[str]) -> str:
    """"eos" only when every chunk finished cleanly; else "length"."""
    if not finish_reasons:
        return "length"
    return "eos" if all(r == "eos" for r in finish_reasons) else "length"


def sanitize_request(request: dict) -> dict:
    """Echo of the request for the envelope: audio payloads become sizes."""
    echo = {}
    for key, value in request.items():
        if key in ("audio_b64", "reference_audio_b64"):
            echo[key] = f"<{len(value) if isinstance(value, str) else 0} b64 chars>"
        elif key == "references":
            echo[key] = [
                {k: (f"<{len(v)} b64 chars>" if k == "audio_b64" else v) for k, v in item.items()}
                for item in value
            ]
        else:
            echo[key] = value
    return echo


class LineCache:
    """LRU cache of rendered envelopes keyed by an exact request fingerprint."""

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES):
        self.max_entries = max_entries
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    @staticmethod
    def key(fingerprint: dict) -> str:
        payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key]
            self.misses += 1
            return None

    def put(self, key: str, envelope: dict) -> None:
        with self._lock:
            self._entries[key] = envelope
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
            }


class TtsService:
    def __init__(
        self,
        engine,
        registry: VoiceRegistry,
        *,
        run: Callable = lambda fn, *a, **kw: fn(*a, **kw),
        cache: LineCache | None = None,
        sample_rate: int = 44100,
    ):
        self.engine = engine
        self.registry = registry
        self.run = run  # server passes its main-thread runner
        self.cache = cache or LineCache()
        self.sample_rate = sample_rate
        self._codes_cache: dict[str, list[np.ndarray]] = {}
        self._upload_codes: dict[str, np.ndarray] = {}
        self._codes_lock = threading.Lock()

    # -- voice resolution ------------------------------------------------------

    def _reference_codes(self, request: dict) -> tuple[list[np.ndarray], list[str], str | None]:
        """Codes + transcripts + cache-fingerprint fragment for the request.

        ``voice=name`` loads from the registry; explicit reference fields are
        the alternative.  Mixing both is refused (ambiguous precedence).
        """
        voice_name = request.get("voice")
        explicit = request.get("reference_audio_b64") or request.get("references")
        if voice_name and explicit:
            raise PreflightError(
                "[FAIL] voice-resolution: pass either voice=name or explicit "
                "reference audio, not both"
            )
        if voice_name:
            voice = self.registry.get(voice_name)  # VoiceError -> 404 upstream
            with self._codes_lock:
                cached = self._codes_cache.get(voice.content_hash())
            if cached is None:
                clips = self.registry.load_references(voice)
                cached = [
                    self.run(self.engine.encode_reference, wav, sr)
                    for wav, sr in clips
                ]
                with self._codes_lock:
                    self._codes_cache[voice.content_hash()] = cached
            texts = [voice.text] * len(cached) if len(cached) > 1 else [voice.text]
            return list(cached), texts, f"voice:{voice_name}:{voice.content_hash()[:16]}"

        texts = request.get("reference_text")
        if not explicit:
            return [], [], None
        if request.get("references"):
            items = request["references"]
            if any(not item.get("text") for item in items):
                raise PreflightError(
                    "[FAIL] voice-resolution: every reference needs 'text'"
                )
            codes = [
                self._encode_uploaded(item["audio_b64"]) for item in items
            ]
            texts = [item["text"] for item in items]
            digest = hashlib.sha256(
                json.dumps([item["audio_b64"][:2048] for item in items]).encode()
            ).hexdigest()[:16]
        else:
            if not request.get("reference_text"):
                raise PreflightError(
                    "[FAIL] voice-resolution: cloning with explicit audio needs "
                    "'reference_text' (or use voice=name)"
                )
            raw = base64.b64decode(request["reference_audio_b64"], validate=True)
            codes = [self._encode_uploaded(request["reference_audio_b64"])]
            texts = [request.get("reference_text", "")]
            digest = hashlib.sha256(raw).hexdigest()[:16]
        return codes, texts, f"refs:{digest}"

    def _encode_uploaded(self, audio_b64: str) -> np.ndarray:
        raw = base64.b64decode(audio_b64, validate=True)
        digest = hashlib.sha256(raw).hexdigest()
        with self._codes_lock:
            cached = self._upload_codes.get(digest)
        if cached is not None:
            return cached

        def encode() -> np.ndarray:
            wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            return self.engine.encode_reference(wav.mean(axis=1), sr)

        codes = self.run(encode)
        with self._codes_lock:
            self._upload_codes[digest] = codes
        return codes

    # -- parameters ------------------------------------------------------------

    def _resolve_params(self, request: dict, voice_recipe: dict | None) -> dict:
        """Voice recipe fills defaults; explicit request parameters win."""
        params = {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 30,
            "repetition_penalty": 1.0,
        }
        if voice_recipe:
            params.update(voice_recipe)
        for key in list(params):
            if key in request and request[key] is not None:
                params[key] = request[key]
        params["top_k"] = int(params["top_k"])
        return params

    # -- rendering -------------------------------------------------------------

    def render(self, request: dict) -> dict:
        """One render request -> envelope dict. Raises VoiceError/PreflightError."""
        request = dict(request)
        text = request.get("text")
        if not text or not text.strip():
            raise PreflightError("[FAIL] request: 'text' must be a non-empty string")
        line_mode = bool(request.get("line", False))

        codes, ref_texts, ref_fingerprint = self._reference_codes(request)
        voice_recipe = None
        if request.get("voice"):
            voice_recipe = self.registry.get(request["voice"]).recipe
        params = self._resolve_params(request, voice_recipe)

        seed = request.get("seed")
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "little")
        seed = int(seed)

        fingerprint = {
            "refs": ref_fingerprint,
            "text": text,
            "line": line_mode,
            "seed": seed,
            "retry": bool(request.get("retry", True)),
            "single_chunk": bool(request.get("single_chunk", False)) or line_mode,
            "max_new_tokens": request.get("max_new_tokens"),
            "chunk_bytes": request.get("chunk_bytes"),
            **params,
        }
        if request.get("sampling_plan") is not None:
            fingerprint["sampling_plan"] = request["sampling_plan"]
        cache_key = self.cache.key(fingerprint)
        cached = self.cache.get(cache_key)
        if cached is not None:
            envelope = dict(cached)
            envelope["cache"] = "hit"
            return envelope

        sampling_plan = request.get("sampling_plan")
        max_attempts = 1
        ladder_active = bool(request.get("retry", True)) and len(text.encode()) <= LADDER_MAX_BYTES
        if ladder_active:
            max_attempts = 1 + LADDER_MAX_RETRIES

        attempts: list[dict] = []
        result = None
        for attempt in range(max_attempts):
            attempt_seed = seed + attempt  # derived deterministically: replays work
            attempt_kwargs: dict = {
                "text": text,
                "reference_codes": codes or None,
                "reference_text": ref_texts or None,
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "top_k": params["top_k"],
                "repetition_penalty": params["repetition_penalty"],
                "seed": attempt_seed,
                "edge_fades": not line_mode,
                "chunk_gap": 0.0 if line_mode else request.get("chunk_gap", 0.05),
            }
            if line_mode:
                # one clean segment: no internal chunking at any attempt
                attempt_kwargs["single_chunk"] = True
            elif attempt > 0:
                # ladder: resplit finer each retry
                attempt_kwargs["chunk_bytes"] = LADDER_CHUNK_BYTES[attempt - 1]
            if sampling_plan is not None:
                attempt_kwargs["sampling_plan"] = sampling_plan
            if request.get("max_new_tokens") is not None:
                attempt_kwargs["max_new_tokens"] = int(request["max_new_tokens"])
            if request.get("single_chunk") and not line_mode:
                attempt_kwargs["single_chunk"] = True
            if request.get("chunk_bytes") is not None and attempt == 0:
                attempt_kwargs["chunk_bytes"] = int(request["chunk_bytes"])

            result = self.run(self.engine.tts, **attempt_kwargs)
            finish = aggregate_finish_reason(result.finish_reasons)
            attempts.append({
                "attempt": attempt + 1,
                "seed": attempt_seed,
                "finish_reason": finish,
                "chunks": result.chunks,
                "duration_s": round(metrics.duration_s(result.audio, result.sample_rate), 3),
            })
            if finish == "eos" or not ladder_active:
                break

        if result is None:
            raise RenderError("render produced nothing")
        finish = aggregate_finish_reason(result.finish_reasons)
        if finish != "eos" and ladder_active:
            raise RenderError(
                f"no EOS-clean take after {len(attempts)} attempts "
                f"({json.dumps(attempts)}); raise quality with a shorter text, "
                "a different seed, or retry=false to accept it anyway"
            )

        warnings = list(result.warnings)
        if finish != "eos":
            warnings.append(
                f"[WARN] finish-discipline: aggregate finish_reason is "
                f"{finish!r} (long-text renders bound degeneration by chunking)"
            )
        wav_bytes = encode_wav(result.audio, result.sample_rate)
        duration = metrics.duration_s(result.audio, result.sample_rate)
        envelope = {
            "ok": True,
            "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
            "duration_s": round(duration, 3),
            "rms": round(metrics.rms(result.audio), 4),
            "f0_hz_median": (
                round(f0, 1)
                if (f0 := metrics.f0_median_hz(result.audio, result.sample_rate)) is not None
                else None
            ),
            "finish_reason": finish,
            "rtf": round(result.real_time_factor, 3),
            "seed": int(attempts[-1]["seed"]),
            "request_echo": sanitize_request(request),
            "attempts": attempts,
            "warnings": warnings,
            "chunks": result.chunks,
            "cache": "miss",
        }
        self.cache.put(cache_key, envelope)
        return envelope

    # -- batch -----------------------------------------------------------------

    def batch(self, request: dict) -> dict:
        """Render lines sequentially through the warm model, per-line envelopes."""
        lines = request.get("lines")
        if not isinstance(lines, list) or not lines:
            raise PreflightError("[FAIL] request: 'lines' must be a non-empty array")
        results = []
        for i, line in enumerate(lines):
            if not isinstance(line, dict):
                line = {"text": str(line)}
            line = dict(line)
            line.setdefault("line_id", i)
            try:
                envelope = self.render(line)
                envelope["ok"] = True
                envelope["line_id"] = line.get("line_id", i)
            except (VoiceError, PreflightError, RenderError) as exc:
                envelope = {
                    "ok": False,
                    "line_id": line.get("line_id", i),
                    "error": f"{type(exc).__name__}: {exc}",
                    "request_echo": sanitize_request(line),
                }
            results.append(envelope)
            # MakeAPod law: mx.clear_cache() per line keeps memory bounded
            self.run(_clear_mlx_cache)
        return {"ok": all(r.get("ok") for r in results), "count": len(results), "results": results}


def _clear_mlx_cache() -> None:
    import mlx.core as mx

    mx.clear_cache()
