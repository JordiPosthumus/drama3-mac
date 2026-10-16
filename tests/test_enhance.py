"""Validator tests for the script augmenter (pure — no network, no MLX)."""

import json

import pytest

from drama3_mac.enhance import EnhanceError, _extract_json_array, enhance_lines, sanitize_line


def test_strips_special_tokens():
    text, warns = sanitize_line("Hi.", "<|speaker:1|> Hi.")
    assert text == "Hi."
    assert any("special" in w for w in warns)


def test_drops_unknown_paired_tags_keeps_inner_text():
    text, warns = sanitize_line("Say it loudly.", "<loudly>Say it loudly.</loudly>")
    assert text == "Say it loudly."
    assert any("unknown" in w for w in warns)


def test_known_paired_tags_survive():
    text, _ = sanitize_line("Say it softly.", "<soft>Say it softly.</soft>")
    assert text == "<soft>Say it softly.</soft>"


def test_line_start_sound_event_removed():
    text, warns = sanitize_line("Well, hello.", "[sighs] Well, hello.")
    assert text == "Well, hello."
    assert any("line-start" in w for w in warns)


def test_overlong_direction_removed():
    cue = "[" + "x" * 90 + "]"
    text, warns = sanitize_line("Fine.", f"{cue} Fine.")
    assert text == "Fine."
    assert any("chars removed" in w for w in warns)


def test_unbalanced_brackets_contaminate_then_fall_back():
    text, warns = sanitize_line("Fine.", "[warm Fine.")
    assert text == "Fine."  # orphaned cue word breaks preservation → original
    assert any("unbalanced" in w for w in warns)


def test_unmatched_close_bracket_falls_back():
    text, warns = sanitize_line("Fine.", "warm] Fine.")
    assert text == "Fine."  # orphaned word breaks preservation → original
    assert any("unbalanced" in w for w in warns)


def test_paired_tag_budget():
    text, warns = sanitize_line(
        "A b c.", "<slow>A</slow> <emphasis>b</emphasis> <stress>c</stress>")
    assert "<" not in text and ">" not in text
    assert any("budget" in w for w in warns)
    # two pairs are fine (at budget, not over)
    text, warns = sanitize_line("A b c.", "<slow>A b <emphasis>c</emphasis>.</slow>")
    assert warns == []
    assert text == "<slow>A b <emphasis>c</emphasis>.</slow>"


def test_all_caps_downcased():
    text, warns = sanitize_line("This is BIG news.", "This is BIG news.")
    assert text == "This is big news."
    assert any("ALL-CAPS" in w for w in warns)


def test_word_change_falls_back_to_original():
    text, warns = sanitize_line("Give me the keys.", "Hand me the keys, please.")
    assert text == "Give me the keys."
    assert any("word content changed" in w for w in warns)


def test_word_addition_falls_back_to_original():
    text, warns = sanitize_line("Yes.", "Yes, absolutely yes.")
    assert text == "Yes."


def test_words_preserved_with_direction_added():
    text, warns = sanitize_line(
        "When your whole life is a routine, that matters.",
        "[warm] When your whole life is a routine... that <slow>matters</slow>.",
    )
    assert warns == []
    assert "routine... that" in text


def test_empty_proposal_keeps_original():
    text, warns = sanitize_line("Hello.", "   ")
    assert text == "Hello."
    assert any("empty" in w for w in warns)


def test_extract_json_array_tolerates_fences():
    reply = 'Sure!\n```json\n[{"line_id": 1, "text": "Hi."}]\n```'
    assert _extract_json_array(reply)[0]["text"] == "Hi."


def test_extract_json_array_no_array_raises():
    with pytest.raises(EnhanceError):
        _extract_json_array("no json here")


def test_enhance_lines_end_to_end(monkeypatch):
    import drama3_mac.enhance as enh

    def fake_chat(messages, base_url, model, api_key, timeout):
        return ('[{"line_id": 0, "text": "[warm] Well... hello there.",'
                ' "rationale": "warm greeting"},'
                ' {"line_id": 1, "text": "Actually CHANGED WORDS.",'
                ' "rationale": "bad"}]')

    monkeypatch.setattr(enh, "_chat", fake_chat)
    out = enh.enhance_lines([{"line_id": 0, "text": "Well, hello there."},
                            {"line_id": 1, "text": "Original words."}])
    assert out[0]["text"] == "[warm] Well... hello there."
    assert out[0]["rationale"] == "warm greeting"
    assert out[1]["text"] == "Original words."  # fell back
    assert any("word content changed" in w for w in out[1]["warnings"])


def test_enhance_lines_missing_proposal_keeps_original(monkeypatch):
    import drama3_mac.enhance as enh

    monkeypatch.setattr(enh, "_chat", lambda *a, **k: '[{"line_id": 0, "text": "Hi."}]')
    out = enh.enhance_lines([{"line_id": 0, "text": "Hi."},
                             {"line_id": 1, "text": "Uncovered."}])
    assert out[1]["text"] == "Uncovered."
    assert any("no proposal" in w for w in out[1]["warnings"])


def test_enhance_lines_rejects_bad_input():
    with pytest.raises(ValueError):
        enhance_lines([{"no_text": True}])
    with pytest.raises(ValueError):
        enhance_lines([])
    with pytest.raises(ValueError):
        enhance_lines([{"text": "x"}] * 65)


def test_empty_content_raises_enhance_error(monkeypatch):
    import drama3_mac.enhance as enh

    def fake_urlopen(req, timeout):
        import io

        class R(io.BytesIO):
            def __enter__(self):
                return self

        payload = {"choices": [{"message": {"role": "assistant",
                                            "content": None,
                                            "reasoning_content": "thought forever"}}]}
        return R(json.dumps(payload).encode())

    monkeypatch.setattr(enh.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(EnhanceError, match="empty content"):
        enh.enhance_lines([{"line_id": 0, "text": "Hi."}])


def test_long_break_normalized():
    text, warns = sanitize_line("Wait. Go on.", "Wait. [long-break] Go on.")
    assert text == "Wait. ... Go on."
    assert any("normalized" in w for w in warns)


def test_break_normalized_to_ellipsis():
    text, warns = sanitize_line("Wait. Go on.", "Wait. [break] Go on.")
    assert text == "Wait. ... Go on."
    assert any("normalized" in w for w in warns)
