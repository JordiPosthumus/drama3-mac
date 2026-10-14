"""End-to-end HTTP API test against the REAL model (boots the actual server).

Skipped automatically when no converted checkpoint exists. Slow (~2-3 min:
model load, warmup, several short renders, one whisper-free determinism
check) — this is the acceptance gate for the production TTS API.
"""

import base64
import io
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import soundfile as sf

from drama3_mac.server import serve

CHECKPOINT = Path("checkpoints/drama3-q4")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="requires the converted drama3-q4 checkpoint"
)

REF_WAV = Path("assets/ref-en0.wav")
REF_TEXT = "We asked over twenty different people, and they all said it was his."


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    voices_dir = tmp_path_factory.mktemp("voices")

    # find a free port, then hand it to serve()
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    threading.Thread(
        target=serve, args=(CHECKPOINT,), kwargs={"voices_dir": voices_dir, "port": port},
        daemon=True,
    ).start()

    deadline = time.time() + 300  # model load + compile warmup
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            break
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2)
    else:
        pytest.fail("server did not come up within 300 s")
    yield port


def _post(port: int, path: str, payload: dict, timeout: int = 600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(port: int, path: str):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _delete(port: int, path: str):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method="DELETE"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _ref_b64() -> str:
    return base64.b64encode(REF_WAV.read_bytes()).decode()


def test_voice_lifecycle_and_health(server):
    port = server
    status, health = _get(port, "/health")
    assert status == 200 and health["ok"]

    status, created = _post(port, "/v1/voices", {
        "name": "test-narrator",
        "audio_b64": _ref_b64(),
        "text": REF_TEXT,
        "recipe": {"temperature": 0.7, "top_p": 0.85},
    })
    assert status == 201 and created["name"] == "test-narrator"
    assert created["recipe"]["temperature"] == 0.7

    status, listed = _get(port, "/v1/voices")
    assert any(v["name"] == "test-narrator" for v in listed["voices"])

    status, one = _get(port, "/v1/voices/test-narrator")
    assert status == 200 and one["references"] == 1

    status, err = _post(port, "/v1/voices", {
        "name": "test-narrator", "audio_b64": _ref_b64(), "text": REF_TEXT,
    })
    assert status == 400 and "already exists" in err["error"]

    status, deleted = _delete(port, "/v1/voices/test-narrator")
    assert status == 200
    status, _ = _get(port, "/v1/voices/test-narrator")
    assert status == 404


def test_tts_json_envelope_and_seed_determinism(server):
    port = server
    payload = {
        "text": "The quick brown fox jumps over the lazy dog.",
        "reference_audio_b64": _ref_b64(),
        "reference_text": REF_TEXT,
        "seed": 1234,
        "response_format": "json",
    }
    status, env = _post(port, "/v1/tts", payload)
    assert status == 200
    for key in ("audio_b64", "duration_s", "rms", "f0_hz_median", "finish_reason",
                "rtf", "seed", "request_echo", "attempts", "cache"):
        assert key in env, key
    assert env["seed"] == 1234
    assert 1.0 < env["duration_s"] < 35.0  # MakeAPod line gate
    assert env["rms"] > 0.02
    assert env["f0_hz_median"] is None or 60 < env["f0_hz_median"] < 400
    assert env["finish_reason"] == "eos"
    assert env["request_echo"]["text"] == payload["text"]
    assert "reference_audio_b64" in env["request_echo"]  # sanitized, not raw

    status2, env2 = _post(port, "/v1/tts", {**payload, "seed": 1234})
    assert status2 == 200
    # same request+seed: served from cache, bit-identical audio
    assert env2["cache"] == "hit"
    assert env2["audio_b64"] == env["audio_b64"]

    status3, env3 = _post(port, "/v1/tts", {
        **payload,
        "retry": False,  # different fingerprint -> cache miss -> fresh render
    })
    assert env3["cache"] == "miss"
    assert env3["audio_b64"] == env["audio_b64"], "same seed must reproduce bytes"


def test_line_mode_one_clean_segment(server):
    port = server
    status, env = _post(port, "/v1/tts", {
        "text": "This is one clean line for the assembler.",
        "reference_audio_b64": _ref_b64(),
        "reference_text": REF_TEXT,
        "seed": 77,
        "line": True,
        "response_format": "json",
    })
    assert status == 200
    assert env["chunks"] == 1
    assert env["finish_reason"] == "eos"
    assert 1.0 < env["duration_s"] < 35.0


def test_batch_renders_lines_with_cache_and_stats(server):
    port = server
    line = {
        "text": "Batch line for the podcast assembler.",
        "voice": None,
        "reference_audio_b64": _ref_b64(),
        "reference_text": REF_TEXT,
        "seed": 555,
        "line": True,
    }
    status, payload = _post(port, "/v1/batch", {"lines": [
        {**line, "line_id": "L01"},
        {**line, "line_id": "L02"},  # identical -> cache hit
        {"text": "", "line_id": "L03"},  # invalid -> per-line error
    ]})
    assert status == 200
    results = payload["results"]
    assert [r["line_id"] for r in results] == ["L01", "L02", "L03"]
    assert results[0]["ok"] and results[1]["ok"] and not results[2]["ok"]
    assert results[1]["cache"] == "hit"
    assert results[0]["audio_b64"] == results[1]["audio_b64"]
    assert "error" in results[2]

    status, stats = _get(port, "/v1/cache")
    assert status == 200
    assert stats["entries"] >= 2 and stats["hits"] >= 1


def test_multi_speaker_tagged_text_auto_chunk(server):
    """Blocker regression (2026-10-15): engine.tts with <|speaker:N|> tags and
    auto-chunking crashed validate_text (list-wrapped chunk list)."""
    port = server
    status, env = _post(port, "/v1/tts", {
        "text": (
            "<|speaker:0|> The keeper rose before dawn, as he had for thirty years.\n"
            "<|speaker:1|> The sea had not yet decided between grey and gold.\n"
            "<|speaker:0|> He logged the wind and watched the boats slip out "
            "of the harbour, one small black stitch at a time."
        ),
        "reference_audio_b64": _ref_b64(),
        "reference_text": REF_TEXT,
        "seed": 31,
        "response_format": "json",
    })
    assert status == 200, env.get("error")
    assert env["finish_reason"] == "eos"
    assert env["duration_s"] > 1.0


def test_wav_mode_still_default(server):
    port = server
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/tts",
        data=json.dumps({
            "text": "Wav mode remains the default.",
            "reference_audio_b64": _ref_b64(),
            "reference_text": REF_TEXT,
            "seed": 9,
            "line": True,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        assert r.headers["Content-Type"] == "audio/wav"
        wav_bytes = r.read()
        assert r.headers.get("X-Real-Time-Factor")
    wav, sr = sf.read(io.BytesIO(wav_bytes))
    assert sr == 44100 and len(wav) > sr  # >1 s of audio
