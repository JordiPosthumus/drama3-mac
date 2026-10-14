"""Unit tests for the guardrails (preflight refusal + postflight warnings)."""

import numpy as np
import pytest

from drama3_mac.guardrails import (
    LOOP_MIN_REPEATS,
    Finding,
    PreflightError,
    check_capacity,
    check_checkpoint,
    check_finish_discipline,
    check_loops,
    check_reference,
    check_tag_balance,
    normalize_words,
    word_f1,
)

SR = 44100


# ---------------------------------------------------------------------------
# tag balance
# ---------------------------------------------------------------------------


def test_balanced_known_tags_pass():
    assert check_tag_balance("<whisper>hide</whisper> and <soft>this</soft>") == []


def test_unclosed_tag_is_fail():
    findings = check_tag_balance("<whisper>never closed")
    assert [f.level for f in findings] == ["FAIL"]
    assert "never closed" in findings[0].message


def test_stray_closer_is_fail():
    findings = check_tag_balance("hello </fast> world")
    assert findings[0].level == "FAIL"
    assert "stray" in findings[0].message


def test_unknown_tag_warns_not_fails():
    findings = check_tag_balance("<shout>loud</shout>")
    assert [f.level for f in findings] == ["WARN"]
    assert "known delivery tag" in findings[0].message


def test_nested_mismatch_is_fail():
    findings = check_tag_balance("<whisper>a <soft>b</whisper></soft>")
    assert any(f.level == "FAIL" for f in findings)


# ---------------------------------------------------------------------------
# capacity
# ---------------------------------------------------------------------------


def _utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def test_capacity_ok_for_typical_job():
    # 5 s reference (~108 rows) + 300-byte chunk + 600 budget << 8192
    assert check_capacity("x" * 300, 108, utf8_len=_utf8_len, max_new_tokens=600) == []


def test_capacity_fails_when_reference_plus_text_overflow():
    # 30 s reference ≈ 645 rows; a 7000-byte text cannot fit with any budget
    findings = check_capacity("x" * 7000, 645, utf8_len=_utf8_len, max_new_tokens=2048)
    assert [f.level for f in findings] == ["FAIL"]
    assert "8192" in findings[0].message
    assert "645" in findings[0].message  # numbers in the message, not vibes


def test_capacity_warns_above_ninety_percent():
    # 645 + 6300 + 16 + 600 = 7561 > 0.9 * 8192 but < 8192
    findings = check_capacity("x" * 6300, 645, utf8_len=_utf8_len, max_new_tokens=600)
    assert [f.level for f in findings] == ["WARN"]


def test_capacity_warns_on_long_reference_even_when_room():
    findings = check_capacity("x" * 100, 645 + 700, utf8_len=_utf8_len, max_new_tokens=600)
    assert all(f.level == "WARN" for f in findings)
    assert any("reference is ~" in f.message for f in findings)


# ---------------------------------------------------------------------------
# reference checks
# ---------------------------------------------------------------------------


def test_reference_too_short_fails():
    findings = check_reference(np.zeros(SR // 2, np.float32), SR)
    assert [f.level for f in findings] == ["FAIL"]


def test_reference_nan_fails():
    audio = np.ones(SR, np.float32)
    audio[100] = np.nan
    assert check_reference(audio, SR)[0].level == "FAIL"


def test_reference_long_warns():
    findings = check_reference(np.random.default_rng(0).uniform(-1, 1, SR * 35).astype(np.float32), SR)
    assert [f.level for f in findings] == ["WARN"]


def test_reference_near_silence_warns():
    findings = check_reference(np.full(SR * 5, 1e-4, np.float32), SR)
    assert any(f.check == "reference" and "near-silence" in f.message for f in findings)


def test_reference_healthy_is_clean():
    rng = np.random.default_rng(1)
    audio = (rng.uniform(-0.5, 0.5, SR * 6)).astype(np.float32)
    assert check_reference(audio, SR) == []


# ---------------------------------------------------------------------------
# checkpoint presence
# ---------------------------------------------------------------------------


def test_checkpoint_missing_dir_fails(tmp_path):
    findings = check_checkpoint(tmp_path / "nope", None)
    assert findings and all(f.level == "FAIL" for f in findings)


def test_checkpoint_missing_codec_fails(tmp_path):
    (tmp_path / "m.safetensors").write_bytes(b"x")
    (tmp_path / "config.json").write_text("{}")
    findings = check_checkpoint(tmp_path, None)
    assert any("codec" in f.message for f in findings if f.level == "FAIL")


def test_checkpoint_complete_is_clean(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "m.safetensors").write_bytes(b"x")
    (d / "config.json").write_text("{}")
    (d / "codec.pth").write_bytes(b"x")
    assert check_checkpoint(d, None) == []


# ---------------------------------------------------------------------------
# postflight: finish discipline + loop detector
# ---------------------------------------------------------------------------


def test_finish_discipline_flags_short_length_cut():
    findings = check_finish_discipline(["length"], [50])  # ~2.3 s, budget 600
    assert [f.level for f in findings] == ["WARN"]
    assert "cut off" in findings[0].message


def test_finish_discipline_flags_immediate_eos():
    findings = check_finish_discipline(["eos"], [5])
    assert [f.level for f in findings] == ["WARN"]


def test_finish_discipline_accepts_full_chunks():
    assert check_finish_discipline(["eos", "length"], [580, 600]) == []


def test_loop_detector_flags_periodic_run():
    rng = np.random.default_rng(2)
    seq = rng.integers(100, 200, 300).astype(np.int64)
    seq[100:100 + 2 * LOOP_MIN_REPEATS] = np.tile([7, 9], LOOP_MIN_REPEATS)
    findings = check_loops([seq])
    assert [f.level for f in findings] == ["WARN"]
    assert "[7, 9]" in findings[0].message
    assert "frame 100" in findings[0].message


def test_loop_detector_accepts_natural_speech_codes():
    rng = np.random.default_rng(3)
    seq = rng.integers(100, 200, 2000).astype(np.int64)
    seq[500:520] = 42  # a 20-run of one code: periodic but below 6*? -> 20 >= 6! 
    findings = check_loops([seq])
    # a 20x run of a single code IS flagged (that is a real loop)
    assert findings and "42" in findings[0].message


def test_loop_detector_ignores_short_runs():
    rng = np.random.default_rng(4)
    seq = rng.integers(100, 200, 500).astype(np.int64)
    seq[50:55] = [7, 9, 7, 9, 7]  # 2.5 repeats: below the threshold
    assert check_loops([seq]) == []


def test_loop_detector_clean_sequence_is_silent():
    rng = np.random.default_rng(5)
    seq = rng.integers(0, 4096, 3000).astype(np.int64)
    assert check_loops([seq]) == []


# ---------------------------------------------------------------------------
# verify round-trip helpers (pure parts)
# ---------------------------------------------------------------------------


def test_word_f1_perfect_match():
    assert word_f1("The quick brown fox.", "the quick, brown fox!") == pytest.approx(1.0)


def test_word_f1_partial_match():
    f1 = word_f1("the quick brown fox jumps", "the quick brown fox")
    assert 0.7 < f1 < 1.0


def test_word_f1_disjoint_is_zero():
    assert word_f1("alpha beta", "gamma delta") == 0.0


def test_normalize_words_strips_punctuation_and_case():
    assert normalize_words("Hello, WORLD! it's") == ["hello", "world", "it", "s"]


def test_finding_str_format():
    assert str(Finding("WARN", "check", "msg")) == "[WARN] check: msg"
