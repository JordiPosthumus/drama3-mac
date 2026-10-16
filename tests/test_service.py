"""Unit tests for the production-API building blocks: metrics, voices, service.

The service tests run against a fake engine (scripted TTSResult stubs), so the
envelope/cache/ladder/line/batch logic is covered without loading the model.
Real-model integration lives in tests/test_server_api.py.
"""

import base64
import io
import json

import numpy as np
import pytest
import soundfile as sf

from drama3_mac import metrics
from drama3_mac.guardrails import PreflightError
from drama3_mac.service import (
    LADDER_CHUNK_BYTES,
    LineCache,
    RenderError,
    TtsService,
    _clear_mlx_cache,
    aggregate_finish_reason,
    sanitize_request,
)
from drama3_mac.voices import VoiceError, VoiceRegistry

SR = 44100


def sine(freq: float, seconds: float, sr: int = SR, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_rms_of_sine_matches_amplitude_over_root2():
    assert metrics.rms(sine(180, 1.0, amplitude=0.3)) == pytest.approx(0.3 / np.sqrt(2), rel=0.01)


def test_duration_s():
    assert metrics.duration_s(sine(100, 2.0), SR) == pytest.approx(2.0)


def test_f0_median_of_180hz_sine():
    f0 = metrics.f0_median_hz(sine(180, 2.0), SR)
    assert f0 == pytest.approx(180, abs=4)


def test_f0_median_of_100hz_sine():
    f0 = metrics.f0_median_hz(sine(100, 2.0), SR)
    assert f0 == pytest.approx(100, abs=3)


def test_f0_silence_returns_none():
    assert metrics.f0_median_hz(np.zeros(SR, np.float32), SR) is None


def test_f0_too_short_returns_none():
    assert metrics.f0_median_hz(sine(180, 0.05), SR) is None


def test_f0_median_across_harmonics_is_fundamental():
    # 120 Hz fundamental + strong 2nd harmonic: median must stay ~120
    a = sine(120, 2.0) + 0.6 * sine(240, 2.0)
    f0 = metrics.f0_median_hz(a, SR)
    assert f0 == pytest.approx(120, abs=6)


# ---------------------------------------------------------------------------
# voices
# ---------------------------------------------------------------------------


def _wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


@pytest.fixture
def registry(tmp_path):
    return VoiceRegistry(tmp_path / "voices")


def test_voice_create_get_list_delete(registry):
    registry.create_from_bytes(
        "narrator", [_wav_bytes(sine(150, 1.5))], "transcript here", {"temperature": 0.7}
    )
    voice = registry.get("narrator")
    assert voice.recipe["temperature"] == 0.7
    assert voice.text == "transcript here"
    assert (registry.root / "narrator" / "reference-1.wav").exists()
    assert [v.name for v in registry.list()] == ["narrator"]
    registry.delete("narrator")
    assert not registry.exists("narrator")
    with pytest.raises(VoiceError):
        registry.get("narrator")


def test_voice_multi_clip(registry):
    registry.create_from_bytes(
        "multi", [_wav_bytes(sine(150, 1.5)), _wav_bytes(sine(160, 1.5))],
        "two takes", {},
    )
    voice = registry.get("multi")
    assert len(voice.files) == 2
    assert len(registry.load_references(voice)) == 2


def test_voice_name_validation(registry):
    for bad in ("Bad Name", "-lead", "x" * 65, "", "UPPER"):
        with pytest.raises(VoiceError):
            registry.create_from_bytes(bad, [_wav_bytes(sine(150, 1.5))], "t", {})


def test_voice_duplicate_rejected(registry):
    registry.create_from_bytes("v", [_wav_bytes(sine(150, 1.5))], "t", {})
    with pytest.raises(VoiceError, match="already exists"):
        registry.create_from_bytes("v", [_wav_bytes(sine(150, 1.5))], "t", {})


def test_voice_recipe_validation(registry):
    with pytest.raises(VoiceError):
        registry.create_from_bytes("v", [_wav_bytes(sine(150, 1.5))], "t", {"temperature": 9})
    with pytest.raises(VoiceError):
        registry.create_from_bytes("v2", [_wav_bytes(sine(150, 1.5))], "t", {"top_p": 0.0})
    with pytest.raises(VoiceError):
        registry.create_from_bytes("v3", [_wav_bytes(sine(150, 1.5))], "t", {"nonsense": 1})


def test_voice_too_short_clip_rejected(registry):
    with pytest.raises(VoiceError, match="under 1 s"):
        registry.create_from_bytes("v", [_wav_bytes(sine(150, 0.3))], "t", {})


def test_voice_undecodable_audio_rejected(registry):
    with pytest.raises(VoiceError, match="cannot decode"):
        registry.create_from_bytes("v", [b"not audio at all"], "t", {})


def test_voice_content_hash_changes_on_recreate(registry):
    registry.create_from_bytes("v", [_wav_bytes(sine(150, 1.5))], "t", {})
    h1 = registry.get("v").content_hash()
    registry.delete("v")
    registry.create_from_bytes("v", [_wav_bytes(sine(200, 1.5))], "t2", {})
    assert registry.get("v").content_hash() != h1


# ---------------------------------------------------------------------------
# service with a fake engine
# ---------------------------------------------------------------------------


class FakeResult:
    def __init__(self, seed_used: int, finish_reasons=("eos",), chunks=1, seconds=2.0,
                 warnings=()):
        freq = 150 + (seed_used % 50)  # audio depends on the seed
        self.audio = sine(freq, seconds)
        self.sample_rate = SR
        self.finish_reasons = tuple(finish_reasons)
        self.chunks = chunks
        self.warnings: tuple[str, ...] = tuple(warnings)
        self._elapsed = seconds / 0.9

    @property
    def real_time_factor(self) -> float:
        return self._elapsed / metrics.duration_s(self.audio, self.sample_rate)


class FakeEngine:
    """Scripted engine: records tts() kwargs, returns seeded results."""

    def __init__(self, finish_sequence=("eos",), warnings_sequence=()):
        self.calls: list[dict] = []
        self.finish_sequence = list(finish_sequence)
        self.warnings_sequence = list(warnings_sequence)
        self.encode_reference = lambda wav, sr=SR: np.zeros((10, 8), np.int32)

    def tts(self, **kwargs):
        self.calls.append(kwargs)
        attempt = len(self.calls) - 1
        finish = self.finish_sequence[min(attempt, len(self.finish_sequence) - 1)]
        warnings = self.warnings_sequence[min(attempt, len(self.warnings_sequence) - 1)] \
            if self.warnings_sequence else ()
        return FakeResult(kwargs.get("seed") or 0, finish_reasons=(finish,),
                          warnings=warnings)


@pytest.fixture
def registry_with_voice(tmp_path):
    reg = VoiceRegistry(tmp_path / "voices")
    reg.create_from_bytes("narrator", [_wav_bytes(sine(150, 1.5))], "ref text", {"temperature": 0.7})
    return reg


def _service(engine=None, registry=None, **kw):
    return TtsService(engine or FakeEngine(), registry or VoiceRegistry("__nonexistent__"), **kw)


def test_envelope_fields_complete(registry_with_voice):
    svc = _service(registry=registry_with_voice)
    env = svc.render({"text": "Hello there.", "voice": "narrator", "seed": 7})
    audio = base64.b64decode(env["audio_b64"])
    wav, sr = sf.read(io.BytesIO(audio))
    assert len(wav) / sr == pytest.approx(env["duration_s"], abs=0.02)
    assert 0 < env["rms"] < 1
    assert env["f0_hz_median"] == pytest.approx(157, abs=4)  # fake tone: 150 + seed%50
    assert env["finish_reason"] == "eos"
    assert env["seed"] == 7
    assert env["rtf"] > 0
    assert env["request_echo"]["text"] == "Hello there."
    assert env["request_echo"]["voice"] == "narrator"
    assert env["attempts"][-1]["finish_reason"] == "eos"
    assert env["cache"] == "miss"
    assert env["ok"] is True


def test_same_seed_bit_identical(registry_with_voice):
    svc = _service(registry=registry_with_voice)
    a = svc.render({"text": "Line one.", "seed": 42})
    b = svc.render({"text": "Line one.", "seed": 42})
    # second is served from cache and is bit-identical
    assert b["cache"] == "hit"
    assert a["audio_b64"] == b["audio_b64"]
    # bypass the cache: two separate services sharing the engine each render once
    engine = FakeEngine()
    svc_a = _service(engine, registry_with_voice)
    svc_b = _service(engine, registry_with_voice)
    a = svc_a.render({"text": "Line one.", "seed": 42})
    b = svc_b.render({"text": "Line one.", "seed": 42})
    assert engine.calls[0]["seed"] == engine.calls[1]["seed"] == 42
    assert a["audio_b64"] == b["audio_b64"]  # same seed -> bit-identical take


def test_cache_miss_on_different_text_and_stats():
    svc = _service()
    svc.render({"text": "a", "seed": 1})
    svc.render({"text": "b", "seed": 1})
    svc.render({"text": "a", "seed": 1})
    stats = svc.cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 2
    assert stats["entries"] == 2


def test_cache_key_includes_voice_content_hash(registry_with_voice, tmp_path):
    svc = _service(registry=registry_with_voice)
    svc.render({"text": "x", "voice": "narrator", "seed": 5})
    # recreate the voice with different audio -> different key -> miss
    registry_with_voice.delete("narrator")
    registry_with_voice.create_from_bytes(
        "narrator", [_wav_bytes(sine(190, 1.5))], "ref text", {}
    )
    env = svc.render({"text": "x", "voice": "narrator", "seed": 5})
    assert env["cache"] == "miss"


def test_voice_recipe_applies_and_request_overrides(registry_with_voice):
    engine = FakeEngine()
    svc = _service(engine, registry_with_voice)
    svc.render({"text": "x", "voice": "narrator", "seed": 1})
    assert engine.calls[0]["temperature"] == 0.7  # from the voice recipe
    svc.render({"text": "x", "voice": "narrator", "seed": 2, "temperature": 0.95})
    assert engine.calls[1]["temperature"] == 0.95  # explicit wins


def test_voice_xor_explicit_reference(registry_with_voice):
    svc = _service(registry=registry_with_voice)
    with pytest.raises(PreflightError, match="not both"):
        svc.render({
            "text": "x", "voice": "narrator", "seed": 1,
            "reference_audio_b64": "AAAA", "reference_text": "t",
        })


def test_unknown_voice_raises_voice_error():
    svc = _service()
    with pytest.raises(VoiceError):
        svc.render({"text": "x", "voice": "ghost"})


def test_line_mode_single_clean_segment(registry_with_voice):
    engine = FakeEngine()
    svc = _service(engine, registry=registry_with_voice)
    svc.render({"text": "One clean line.", "line": True, "seed": 3})
    kwargs = engine.calls[0]
    assert kwargs["single_chunk"] is True
    assert kwargs["edge_fades"] is False
    assert kwargs["chunk_gap"] == 0.0


def test_retry_ladder_resplits_until_eos():
    engine = FakeEngine(finish_sequence=("length", "eos"))
    svc = _service(engine)
    env = svc.render({"text": "x" * 200, "seed": 10})
    assert len(env["attempts"]) == 2
    assert env["finish_reason"] == "eos"
    assert env["attempts"][0]["finish_reason"] == "length"
    assert env["attempts"][1]["seed"] == 11
    assert engine.calls[1]["chunk_bytes"] == LADDER_CHUNK_BYTES[0]
    assert engine.calls[0].get("chunk_bytes") is None


def test_retry_ladder_exhaustion_raises():
    engine = FakeEngine(finish_sequence=("length",))
    svc = _service(engine)
    with pytest.raises(RenderError, match="no clean take \(EOS, no loop warnings\) after 3 attempts"):
        svc.render({"text": "x" * 200, "seed": 10})
    assert len(engine.calls) == 3


def test_retry_opt_out_returns_length_take_with_warning():
    engine = FakeEngine(finish_sequence=("length",))
    svc = _service(engine)
    env = svc.render({"text": "x" * 200, "seed": 10, "retry": False})
    assert len(env["attempts"]) == 1
    assert env["finish_reason"] == "length"
    assert any("finish-discipline" in w for w in env["warnings"])
    assert len(engine.calls) == 1


def test_ladder_skipped_for_long_text():
    engine = FakeEngine(finish_sequence=("length",))
    svc = _service(engine)
    env = svc.render({"text": "x" * 1500, "seed": 1})  # > LADDER_MAX_BYTES
    assert len(env["attempts"]) == 1
    assert env["finish_reason"] == "length"


def test_line_mode_ladder_reseeds_without_resplit():
    engine = FakeEngine(finish_sequence=("length", "eos"))
    svc = _service(engine)
    env = svc.render({"text": "x" * 200, "line": True, "seed": 10})
    assert len(env["attempts"]) == 2
    assert engine.calls[1]["single_chunk"] is True
    assert "chunk_bytes" not in engine.calls[1]


def test_batch_renders_lines_and_captures_errors(registry_with_voice):
    engine = FakeEngine()
    ran = []
    svc = _service(engine, registry_with_voice, run=lambda fn, *a, **kw: ran.append(fn) or fn(*a, **kw))
    payload = svc.batch({"lines": [
        {"text": "First line.", "voice": "narrator", "seed": 1, "line_id": "t01"},
        {"text": "   "},  # invalid -> per-line error, batch continues
        {"text": "Third line.", "seed": 2, "line_id": "t03"},
    ]})
    assert payload["count"] == 3
    assert payload["results"][0]["ok"] is True
    assert payload["results"][0]["line_id"] == "t01"
    assert payload["results"][1]["ok"] is False
    assert "error" in payload["results"][1]
    assert payload["results"][2]["ok"] is True
    assert payload["ok"] is False
    # one clear-cache per line (MakeAPod's law), through the runner
    assert sum(1 for fn in ran if fn is _clear_mlx_cache) == 3


def test_batch_duplicate_line_hits_cache():
    engine = FakeEngine()
    svc = _service(engine)
    payload = svc.batch({"lines": [
        {"text": "Same line.", "seed": 5},
        {"text": "Same line.", "seed": 5},
    ]})
    assert payload["results"][0]["cache"] == "miss"
    assert payload["results"][1]["cache"] == "hit"
    assert len(engine.calls) == 1  # second served from cache


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------


def test_aggregate_finish_reason():
    assert aggregate_finish_reason(("eos", "eos")) == "eos"
    assert aggregate_finish_reason(("eos", "length")) == "length"
    assert aggregate_finish_reason(()) == "length"


def test_sanitize_request_hides_audio_payloads():
    echo = sanitize_request({
        "text": "hi",
        "reference_audio_b64": "A" * 1000,
        "references": [{"audio_b64": "B" * 500, "text": "t"}],
    })
    assert echo["reference_audio_b64"] == "<1000 b64 chars>"
    assert echo["references"][0]["audio_b64"] == "<500 b64 chars>"
    assert echo["references"][0]["text"] == "t"


def test_line_cache_lru_eviction():
    cache = LineCache(max_entries=2)
    for i in range(3):
        cache.put(cache.key({"i": i}), {"n": i})
    assert cache.get(cache.key({"i": 0})) is None  # evicted
    assert cache.get(cache.key({"i": 2})) == {"n": 2}
    assert cache.stats()["entries"] == 2


# ---------------------------------------------------------------------------
# dialogue: reactive turn threading
# ---------------------------------------------------------------------------


def test_dialogue_threads_history(registry_with_voice):
    engine = FakeEngine()
    svc = _service(engine=engine, registry=registry_with_voice)
    env = svc.dialogue({"lines": [
        {"text": "Hello there.", "voice": "narrator", "seed": 1},
        {"text": "Hi, good morning.", "voice": "narrator", "seed": 2},
        {"text": "Sleep well.", "voice": "narrator", "seed": 3},
    ]})
    assert env["ok"] is True and env["count"] == 3
    # every line renders in line mode
    assert all(call.get("single_chunk") is True for call in engine.calls)
    # first line is deaf; later lines hear the previous turn's text + codes
    assert not engine.calls[0].get("history")
    second = engine.calls[1]["history"]
    assert [text for text, _ in second] == ["Hello there."]
    assert second[0][1].shape == (10, 8)  # FakeEngine.encode_reference codes
    third = engine.calls[2]["history"]
    assert [text for text, _ in third] == ["Hi, good morning."]
    # assembled conversation wav exists and covers all three lines
    wav, sr = sf.read(io.BytesIO(base64.b64decode(env["audio_b64"])))
    assert len(wav) / sr > 3 * 0.5


def test_dialogue_history_turns_zero_is_deaf(registry_with_voice):
    engine = FakeEngine()
    svc = _service(engine=engine, registry=registry_with_voice)
    env = svc.dialogue({"history_turns": 0, "lines": [
        {"text": "One.", "voice": "narrator"},
        {"text": "Two.", "voice": "narrator"},
    ]})
    assert env["ok"] is True
    assert env["history_turns"] == 0
    assert not engine.calls[1].get("history")


def test_dialogue_failed_line_keeps_threading(registry_with_voice):
    engine = FakeEngine(finish_sequence=("eos",))
    svc = _service(engine=engine, registry=registry_with_voice)
    env = svc.dialogue({"lines": [
        {"text": "Fine.", "voice": "narrator"},
        {"text": "Broken line.", "voice": "does-not-exist"},
        {"text": "Recovers.", "voice": "narrator"},
    ]})
    assert env["ok"] is False
    assert env["results"][1]["ok"] is False
    assert "VoiceError" in env["results"][1]["error"]
    # the third line still hears the last GOOD turn
    assert [text for text, _ in engine.calls[1]["history"]] == ["Fine."]


def test_history_splits_the_cache_key(registry_with_voice):
    svc = _service(registry=registry_with_voice)
    wav_b64 = base64.b64encode(_wav_bytes(sine(200, 1.0))).decode()
    deaf = svc.render({"text": "Same text.", "voice": "narrator", "seed": 5})
    reactive = svc.render({"text": "Same text.", "voice": "narrator", "seed": 5,
                           "history": [{"text": "Prior turn.", "audio_b64": wav_b64}]})
    assert deaf["cache"] == "miss" and reactive["cache"] == "miss"
    again = svc.render({"text": "Same text.", "voice": "narrator", "seed": 5,
                        "history": [{"text": "Prior turn.", "audio_b64": wav_b64}]})
    assert again["cache"] == "hit"
    # different history audio -> different key, no stale replay
    other = svc.render({"text": "Same text.", "voice": "narrator", "seed": 5,
                        "history": [{"text": "Prior turn.", "audio_b64": base64.b64encode(_wav_bytes(sine(300, 1.0))).decode()}]})
    assert other["cache"] == "miss"


def test_history_requires_text_and_audio(registry_with_voice):
    svc = _service(registry=registry_with_voice)
    with pytest.raises(PreflightError):
        svc.render({"text": "Hi.", "voice": "narrator", "history": [{"text": "no audio"}]})


def test_sanitize_redacts_history_audio(registry_with_voice):
    svc = _service(registry=registry_with_voice)
    wav_b64 = base64.b64encode(_wav_bytes(sine(200, 0.2))).decode()
    env = svc.render({"text": "Hi.", "voice": "narrator", "seed": 9,
                      "history": [{"text": "Prior.", "audio_b64": wav_b64}]})
    echoed = env["request_echo"]["history"][0]
    assert "audio_b64" in echoed and echoed["audio_b64"].startswith("<")


def test_dialogue_speaker_ids_tag_lines_and_history(registry_with_voice):
    engine = FakeEngine()
    svc = _service(engine=engine, registry=registry_with_voice)
    env = svc.dialogue({"lines": [
        {"text": "Hello there.", "voice": "narrator", "seed": 1, "speaker": 0},
        {"text": "Hi, good morning.", "voice": "narrator", "seed": 2, "speaker": 1},
        {"text": "Sleep well.", "voice": "narrator", "seed": 3, "speaker": 0},
    ]})
    assert env["ok"] is True
    # line texts carry their speaker tag; history text carries the speaker's tag
    assert engine.calls[0]["text"].startswith("<|speaker:0|> Hello there.")
    assert engine.calls[1]["text"].startswith("<|speaker:1|> Hi, good morning.")
    history = engine.calls[2]["history"]
    assert [text for text, _ in history] == ["<|speaker:1|> Hi, good morning."]


def test_dialogue_speaker_validation(registry_with_voice):
    svc = _service(registry=registry_with_voice)
    with pytest.raises(PreflightError):
        svc.dialogue({"lines": [{"text": "Hi.", "voice": "narrator", "speaker": 16}]})
    with pytest.raises(PreflightError):
        svc.dialogue({"lines": [{"text": "Hi.", "voice": "narrator", "speaker": "loud"}]})


# ---------------------------------------------------------------------------
# Phase 1: loop-aware ladder, degradation, voiced_ratio, speaker passthrough
# ---------------------------------------------------------------------------


def test_ladder_retries_on_loop_warnings(registry_with_voice):
    """eos + loop-detector warning is NOT clean: the ladder must re-roll."""
    engine = FakeEngine(finish_sequence=("eos", "eos"))
    engine.warnings_sequence = (
        ("[WARN] loop-detector: chunk 0: pattern [1] repeats 12x",),
        (),
    )
    svc = _service(engine=engine, registry=registry_with_voice)
    env = svc.render({"text": "Hello there.", "voice": "narrator", "seed": 7})
    assert env["ok"] is True
    assert env["finish_reason"] == "eos"
    assert env["degraded"] is False
    assert [a["loop"] for a in env["attempts"]] == [True, False]
    assert len(engine.calls) == 2
    assert env["seed"] == 8  # the clean attempt's seed, not the last one


def test_ladder_keeps_first_clean_take(registry_with_voice):
    """A later looped attempt must not displace an earlier clean one."""
    engine = FakeEngine(finish_sequence=("eos", "eos", "eos"))
    engine.warnings_sequence = ((), ("[WARN] loop-detector: x",), ())
    svc = _service(engine=engine, registry=registry_with_voice)
    env = svc.render({"text": "Hello there.", "voice": "narrator", "seed": 7})
    assert env["degraded"] is False
    assert env["seed"] == 7  # attempt 1 was clean; the ladder stopped there


def test_ladder_escalates_temperature_on_final_attempt(registry_with_voice):
    engine = FakeEngine(finish_sequence=("length", "length", "eos"))
    svc = _service(engine=engine, registry=registry_with_voice)
    env = svc.render({"text": "Hello there.", "voice": "narrator", "seed": 7,
                      "temperature": 0.8, "max_retries": 2})
    temps = [a["temperature"] for a in env["attempts"]]
    assert temps == [0.8, 0.8, 1.0]  # escalation only on the final attempt
    assert env["attempts"][-1]["finish_reason"] == "eos"


def test_no_temperature_escalation_with_sampling_plan(registry_with_voice):
    engine = FakeEngine(finish_sequence=("length", "eos"))
    svc = _service(engine=engine, registry=registry_with_voice)
    env = svc.render({"text": "Hello there.", "voice": "narrator", "seed": 7,
                      "temperature": 0.8, "max_retries": 1,
                      "sampling_plan": [[0.7, 0.85]]})
    assert env["attempts"][0]["temperature"] == 0.8


def test_exhaustion_with_accept_degraded_returns_best(registry_with_voice):
    """length (no loop) + accept_degraded -> degraded take instead of 422."""
    engine = FakeEngine(finish_sequence=("length",))
    svc = _service(engine=engine, registry=registry_with_voice)
    env = svc.render({"text": "Hello there.", "voice": "narrator", "seed": 7,
                      "accept_degraded": True})
    assert env["ok"] is True
    assert env["degraded"] is True
    assert any("degraded" in w for w in env["warnings"])


def test_exhaustion_looped_take_never_ships(registry_with_voice):
    """A looped take is not degraded-eligible: still raises."""
    engine = FakeEngine(finish_sequence=("length", "length"))
    engine.warnings_sequence = (
        ("[WARN] loop-detector: x",),
        ("[WARN] loop-detector: y",),
    )
    svc = _service(engine=engine, registry=registry_with_voice)
    with pytest.raises(RenderError):
        svc.render({"text": "Hello there.", "voice": "narrator", "seed": 7,
                    "accept_degraded": True})


def test_voiced_ratio_clean_vs_gapped(registry_with_voice):
    assert metrics.voiced_ratio(sine(150, 1.0), SR) > 0.9
    gapped = np.concatenate([np.zeros(int(SR * 0.5), np.float32), sine(150, 0.5)])
    ratio = metrics.voiced_ratio(gapped, SR)
    assert 0.2 < ratio < 0.8
    env = _service(registry=registry_with_voice).render(
        {"text": "Hello there.", "voice": "narrator", "seed": 3})
    assert 0.0 < env["voiced_ratio"] <= 1.0


def test_render_speaker_passthrough(registry_with_voice):
    engine = FakeEngine()
    svc = _service(engine=engine, registry=registry_with_voice)
    svc.render({"text": "Hello there.", "voice": "narrator", "seed": 1, "speaker": 1})
    assert engine.calls[0]["text"].startswith("<|speaker:1|> Hello there.")
    # explicit tag in text wins; no double tagging
    svc.render({"text": "<|speaker:2|> Explicit.", "voice": "narrator", "seed": 2,
                "speaker": 0})
    assert engine.calls[1]["text"].startswith("<|speaker:2|> Explicit.")
    with pytest.raises(PreflightError):
        svc.render({"text": "Hi.", "voice": "narrator", "speaker": 16})


def test_dialogue_gap_before_s(registry_with_voice):
    engine = FakeEngine()
    svc = _service(engine=engine, registry=registry_with_voice)
    env = svc.dialogue({"gap_s": 0.0, "lines": [
        {"text": "One.", "voice": "narrator", "seed": 1},
        {"text": "Two.", "voice": "narrator", "seed": 2, "gap_before_s": 0.5},
        {"text": "Three.", "voice": "narrator", "seed": 3},
    ]})
    assert env["ok"] is True
    assert env["results"][1]["start_s"] == pytest.approx(2.5, abs=0.01)  # 2.0 audio + 0.5 gap
    assert env["results"][2]["start_s"] == pytest.approx(4.5, abs=0.01)


def test_dialogue_aggregate_warnings(registry_with_voice):
    engine = FakeEngine(finish_sequence=("eos", "eos"))
    engine.warnings_sequence = ((), ("[WARN] loop-detector: x",))
    svc = _service(engine=engine, registry=registry_with_voice)
    env = svc.dialogue({"lines": [
        {"text": "One.", "voice": "narrator", "seed": 1},
        {"text": "Two.", "voice": "narrator", "seed": 2, "retry": False},
    ]})
    assert any("loop-detector" in w for w in env["warnings"])
