"""Local HTTP server for the Drama 3 TTS engine.

All inference runs on the thread that loaded the model (MLX streams and RNG
state are thread-local): HTTP threads submit jobs through ``MainThreadRunner``
and wait. The production logic — voice registry, JSON envelope, seeds, line
cache, retry ladder, line mode, batch — lives in ``service.TtsService``; this
module is HTTP plumbing around it.

Endpoints:
    GET    /health                 liveness + model info
    GET    /v1/voices              list registered voices
    GET    /v1/voices/{name}       one voice's manifest
    POST   /v1/voices              {name, audio_b64, text, recipe{...}}
    DELETE /v1/voices/{name}       remove a voice
    GET    /v1/cache               line-cache stats
    POST   /v1/tts                 render (wav | pcm_f32 | json envelope)
    POST   /v1/batch               {lines: [{text, voice, line_id, ...}]}
"""

from __future__ import annotations

import base64
import functools
import json
import queue
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .engine import Drama3Engine
from .guardrails import PreflightError
from .service import RenderError, TtsService
from .voices import VoiceError, VoiceRegistry

MAX_BODY_BYTES = 512 * 1024 * 1024  # 30 s multi-take references are big


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


class _Handler(BaseHTTPRequestHandler):
    service: TtsService
    runner: MainThreadRunner
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        print(f"[server] {self.address_string()} {fmt % args}")

    # -- plumbing ---------------------------------------------------------------

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_json(400, {"error": "bad Content-Length"})
                return None
            return json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return None

    def _run_with_stream_retry(self, fn, *args, **kwargs):
        """Run an inference job; MLX's lazy compile has a rare first-call race
        (\"There is no Stream(...) in current thread\") that a retry on the
        same thread resolves because the graphs are compiled by then."""
        try:
            return self.runner.call(functools.partial(fn, *args, **kwargs))
        except RuntimeError as exc:
            if "Stream" not in str(exc):
                raise
            print(f"[server] transient stream error, retrying once: {exc}", flush=True)
            return self.runner.call(functools.partial(fn, *args, **kwargs))

    # -- GET ----------------------------------------------------------------------

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/health", "/v1/health"):
            engine = self.service.engine
            self._send_json(
                200,
                {
                    "ok": True,
                    "model": engine.model_dir,
                    "sample_rate": engine.codec.sample_rate,
                    "cache": self.service.cache.stats(),
                },
            )
        elif path == "/v1/voices":
            self._send_json(200, {"voices": [v.describe() for v in self.service.registry.list()]})
        elif path.startswith("/v1/voices/"):
            name = path[len("/v1/voices/"):]
            try:
                self._send_json(200, self.service.registry.get(name).describe())
            except VoiceError as exc:
                self._send_json(404, {"error": str(exc)})
        elif path == "/v1/cache":
            self._send_json(200, self.service.cache.stats())
        else:
            self._send_json(404, {"error": f"unknown path {self.path}"})

    # -- DELETE -------------------------------------------------------------------

    def do_DELETE(self) -> None:
        path = self.path.rstrip("/")
        if path.startswith("/v1/voices/"):
            name = path[len("/v1/voices/"):]
            try:
                self.service.registry.delete(name)
                self._send_json(200, {"ok": True, "deleted": name})
            except VoiceError as exc:
                self._send_json(404, {"error": str(exc)})
        else:
            self._send_json(404, {"error": f"unknown path {self.path}"})

    # -- POST ---------------------------------------------------------------------

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        if path in ("/v1/tts", "/v1/audio/speech"):
            self._handle_tts()
        elif path == "/v1/voices":
            self._handle_voice_create()
        elif path == "/v1/batch":
            self._handle_batch()
        else:
            self._send_json(404, {"error": f"unknown path {self.path}"})

    def _handle_voice_create(self) -> None:
        request = self._read_json()
        if request is None:
            return
        name = request.get("name")
        audio_b64 = request.get("audio_b64")
        text = request.get("text")
        if not name or not isinstance(audio_b64, str) or not text:
            self._send_json(
                400,
                {"error": "voice creation needs 'name', 'audio_b64' and 'text'"},
            )
            return
        clips = audio_b64 if isinstance(audio_b64, list) else [audio_b64]
        try:
            raw_items = [base64.b64decode(c, validate=True) for c in clips]
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"error": f"bad audio_b64: {exc}"})
            return
        try:
            voice = self.service.registry.create_from_bytes(
                name, raw_items, text, request.get("recipe") or {}
            )
        except VoiceError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(201, voice.describe())

    def _handle_batch(self) -> None:
        request = self._read_json()
        if request is None:
            return
        try:
            payload = self._run_with_stream_retry(self.service.batch, request)
        except PreflightError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except (RenderError, VoiceError, RuntimeError, ValueError) as exc:
            traceback.print_exc()
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send_json(200, payload)

    def _handle_tts(self) -> None:
        request = self._read_json()
        if request is None:
            return
        text = request.get("text") or request.get("input")
        if not text or not isinstance(text, str):
            self._send_json(400, {"error": "missing 'text'"})
            return
        request["text"] = text

        fmt = request.get("response_format", "wav")
        if fmt not in ("wav", "pcm_f32", "json"):
            self._send_json(400, {"error": f"unknown response_format {fmt}"})
            return

        try:
            envelope = self._run_with_stream_retry(self.service.render, request)
        except VoiceError as exc:
            self._send_json(404, {"error": str(exc)})
            return
        except PreflightError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except RenderError as exc:
            self._send_json(422, {"error": str(exc)})
            return
        except (RuntimeError, ValueError) as exc:
            traceback.print_exc()
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return

        if fmt == "json":
            self._send_json(200, envelope)
            return

        audio = base64.b64decode(envelope["audio_b64"])
        if fmt == "pcm_f32":
            # envelope keeps wav bytes; decode back to raw f32 for pcm mode
            import io as _io
            import soundfile as _sf

            wav, sr = _sf.read(_io.BytesIO(audio), dtype="float32")
            body = wav.astype(np.float32).tobytes()
            content_type = f"audio/f32;rate={sr}"
        else:
            body = audio
            content_type = "audio/wav"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Generation-Seconds", f"{envelope['duration_s'] / max(envelope['rtf'], 1e-9):.3f}")
        self.send_header("X-Frames", str(int(envelope["duration_s"] * self.service.sample_rate)))
        self.send_header("X-Real-Time-Factor", f"{envelope['rtf']:.3f}")
        if envelope.get("warnings"):
            # postflight findings ride along as a header; WARNs never fail
            self.send_header("X-Warnings", " | ".join(envelope["warnings"])[:8000])
        self.end_headers()
        self.wfile.write(body)


def serve(
    model_dir: str | Path,
    codec_path: str | Path | None = None,
    codec_device: str = "auto",
    codec_dtype: str = "float32",
    host: str = "127.0.0.1",
    port: int = 8843,
    voices_dir: str | Path = "voices",
) -> None:
    print(f"[server] loading model from {model_dir} ...", flush=True)
    engine = Drama3Engine(
        model_dir, codec_path=codec_path, codec_device=codec_device, codec_dtype=codec_dtype
    )
    runner = MainThreadRunner()
    registry = VoiceRegistry(voices_dir)
    service = TtsService(
        engine,
        registry,
        run=runner.call,
        sample_rate=engine.codec.sample_rate,
    )

    # Warm up MLX's compiled decode graphs before serving: the first call pays
    # ~10-60 s of compile. Runs inline on this thread, which IS the thread that
    # loaded the model — exactly where runner.loop() will execute all jobs.
    print("[server] warming up decode graphs ...", flush=True)
    runner.call(
        functools.partial(engine.tts, "Warmup.", single_chunk=True, max_new_tokens=12)
    )
    print("[server] warmup done", flush=True)

    handler = type("BoundHandler", (_Handler,), {"service": service, "runner": runner})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    print(
        f"[server] ready on http://{host}:{port}  "
        f"(POST /v1/tts | /v1/batch | /v1/voices, GET /health | /v1/voices | /v1/cache, "
        f"DELETE /v1/voices/{{name}}; codec {engine.codec_device} @ {engine.codec.sample_rate} Hz, "
        f"voices in {voices_dir})",
        flush=True,
    )
    try:
        runner.loop()  # synthesis happens here, on the loader thread
    except KeyboardInterrupt:
        print("[server] shutting down")
    finally:
        httpd.shutdown()
