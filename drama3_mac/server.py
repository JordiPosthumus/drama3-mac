"""Local HTTP server for Drama 3 TTS (stdlib only, no extra dependencies).

Endpoints:
    GET  /health              -> {"ok": true, "model": ...}
    POST /v1/tts              -> audio/wav bytes

/v1/tts JSON body:
    {
      "text": "Text to speak",                       # required
      "reference_audio_b64": "<base64 wav/mp3/...>", # optional; enables cloning
      "reference_text": "transcript of the clip",    # required when cloning
      "temperature": 0.8, "top_k": 30, "top_p": 0.9,
      "max_new_tokens": 0,                # 0 = auto (2048 single pass, 600/chunk)
      "single_chunk": false,              # true = one pass, no chunking
      "chunk_gap": 0.05,                  # silence between chunk audios (s)
      "repetition_penalty": 1.0,          # opt-in loop guard; 1.0 = off = Fish-faithful
      "sampling_plan": [[0.7, 0.85], [0.8, 0.9]],  # optional per-chunk (temp, top_p);
                                          # length must equal the chunk count
      "response_format": "wav" | "pcm_f32"           # default wav
    }

All MLX/PyTorch work runs on the main thread (the model was created there and
MLX streams are thread-local); HTTP threads submit jobs and wait. Cloned
references are cached by content hash, so repeat requests with the same clip
skip the codec encode. Requests are serialized.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import io
import json
import queue
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf

from .engine import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, Drama3Engine

MAX_BODY_BYTES = 64 * 1024 * 1024  # reference clips can be large


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class MainThreadRunner:
    """Runs callables on the thread that created the model."""

    def __init__(self):
        self.jobs: queue.Queue = queue.Queue()

    def loop(self) -> None:
        while True:
            fn, args, box = self.jobs.get()
            if fn is None:
                break
            try:
                box["result"] = fn(*args)
            except BaseException as exc:  # surface everything to the caller
                box["error"] = exc
            box["done"].set()

    def call(self, fn, *args):
        box = {"done": threading.Event()}
        self.jobs.put((fn, args, box))
        box["done"].wait()
        if "error" in box:
            raise box["error"]
        return box["result"]


class _Handler(BaseHTTPRequestHandler):
    engine: Drama3Engine
    runner: MainThreadRunner
    refs: dict  # sha256(audio bytes) -> codes np.ndarray
    refs_lock: threading.Lock
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        print(f"[server] {self.address_string()} {fmt % args}")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/health", "/v1/health"):
            self._send_json(
                200,
                {
                    "ok": True,
                    "model": self.engine.model_dir,
                    "sample_rate": self.engine.codec.sample_rate,
                },
            )
        else:
            self._send_json(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:
        if self.path not in ("/v1/tts", "/v1/audio/speech"):
            self._send_json(404, {"error": f"unknown path {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_json(400, {"error": "bad Content-Length"})
                return
            request = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return

        text = request.get("text") or request.get("input")
        if not text or not isinstance(text, str):
            self._send_json(400, {"error": "missing 'text'"})
            return

        ref_b64 = request.get("reference_audio_b64")
        ref_text = request.get("reference_text")
        # multi-reference form: references: [{audio_b64, text}, ...]
        batch = request.get("references")
        if batch is not None:
            if not isinstance(batch, list) or not batch:
                self._send_json(400, {"error": "references must be a non-empty list"})
                return
            ref_b64 = [item.get("audio_b64") for item in batch]
            ref_text = [item.get("text") for item in batch]
            if any(not t for t in ref_text):
                self._send_json(400, {"error": "every reference needs 'text'"})
                return
        if ref_b64 is not None and not ref_text:
            self._send_json(400, {"error": "cloning needs 'reference_text'"})
            return
        reference_codes = None
        if ref_b64 is not None:
            b64_list = ref_b64 if isinstance(ref_b64, list) else [ref_b64]
            try:
                audio_list = [
                    base64.b64decode(b, validate=True) for b in b64_list
                ]
            except (ValueError, TypeError) as exc:
                self._send_json(400, {"error": f"bad reference audio: {exc}"})
                return
            keys = [hashlib.sha256(a).hexdigest() for a in audio_list]
            with self.refs_lock:
                cached_list = [self.refs.get(k) for k in keys]
            for i, (key, cached) in enumerate(zip(keys, cached_list)):
                if cached is not None:
                    continue
                try:
                    wav, sr = sf.read(
                        io.BytesIO(audio_list[i]), dtype="float32", always_2d=True
                    )
                except Exception as exc:
                    self._send_json(400, {"error": f"cannot decode reference audio: {exc}"})
                    return
                cached = self.runner.call(
                    self.engine.encode_reference, wav.mean(axis=1), sr
                )
                with self.refs_lock:
                    self.refs[key] = cached
                cached_list[i] = cached
            reference_codes = cached_list

        fmt = request.get("response_format", "wav")
        if fmt not in ("wav", "pcm_f32"):
            self._send_json(400, {"error": f"unknown response_format {fmt}"})
            return

        job = functools.partial(
            self.engine.tts,
            text,
            reference_codes=reference_codes,
            reference_text=ref_text if ref_b64 is not None else None,
            max_new_tokens=int(request.get("max_new_tokens", 0)),
            temperature=float(request.get("temperature", DEFAULT_TEMPERATURE)),
            top_k=int(request.get("top_k", DEFAULT_TOP_K)),
            top_p=float(request.get("top_p", DEFAULT_TOP_P)),
            single_chunk=bool(request.get("single_chunk", False)),
            chunk_gap=float(request.get("chunk_gap", 0.05)),
            repetition_penalty=float(request.get("repetition_penalty", 1.0)),
            sampling_plan=(
                [tuple(p) for p in request["sampling_plan"]]
                if request.get("sampling_plan") is not None
                else None
            ),
        )
        try:
            try:
                result = self.runner.call(job)
            except RuntimeError as exc:
                # MLX's lazy compile has a rare first-call race ("There is no
                # Stream(...) in current thread"); retrying on the same thread
                # succeeds because the graphs are compiled by then.
                if "Stream" not in str(exc):
                    raise
                print(f"[server] transient stream error, retrying once: {exc}", flush=True)
                result = self.runner.call(job)
            if fmt == "pcm_f32":
                body = result.audio.astype(np.float32).tobytes()
                content_type = "audio/f32;rate=%d" % result.sample_rate
            else:
                body = _encode_wav(result.audio, result.sample_rate)
                content_type = "audio/wav"
        except (RuntimeError, ValueError) as exc:
            traceback.print_exc()
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Generation-Seconds", f"{result.generation.elapsed:.3f}")
        self.send_header("X-Frames", str(len(result.generation.frames)))
        self.send_header("X-Real-Time-Factor", f"{result.real_time_factor:.3f}")
        if result.warnings:
            # postflight findings ride along as a header; WARNs never fail
            # the request, they report
            self.send_header("X-Warnings", " | ".join(result.warnings)[:8000])
        self.end_headers()
        self.wfile.write(body)


def serve(
    model_dir: str | Path,
    codec_path: str | Path | None = None,
    codec_device: str = "auto",
    codec_dtype: str = "float32",
    host: str = "127.0.0.1",
    port: int = 8843,
) -> None:
    print(f"[server] loading model from {model_dir} ...", flush=True)
    engine = Drama3Engine(
        model_dir, codec_path=codec_path, codec_device=codec_device, codec_dtype=codec_dtype
    )
    runner = MainThreadRunner()

    # Warm up MLX's compiled decode graphs before serving: the first call pays
    # ~10-60 s of compile. Runs inline on this thread, which IS the thread that
    # loaded the model — exactly where runner.loop() will execute all jobs.
    print("[server] warming up decode graphs ...", flush=True)
    engine.tts("Warmup.", single_chunk=True, max_new_tokens=12)
    print("[server] warmup done", flush=True)

    handler = type(
        "BoundHandler",
        (_Handler,),
        {"engine": engine, "runner": runner, "refs": {}, "refs_lock": threading.Lock()},
    )
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    print(
        f"[server] ready on http://{host}:{port}  "
        f"(POST /v1/tts, GET /health; codec {engine.codec_device} @ {engine.codec.sample_rate} Hz)",
        flush=True,
    )
    try:
        runner.loop()  # synthesis happens here, on the loader thread
    except KeyboardInterrupt:
        print("[server] shutting down")
    finally:
        httpd.shutdown()
