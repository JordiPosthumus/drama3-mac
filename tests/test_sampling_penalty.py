"""Unit tests for the opt-in repetition penalty (CTRL-style, window of 64).

The penalty must only push down logits of ids present in the recent window:
positive logit / penalty, negative logit * penalty, untouched otherwise.
"""

import numpy as np
import pytest

import mlx.core as mx

from drama3_mac.tower import _compiled_penalty


VOCAB = 32
PENALTY = 1.2


def test_penalty_pushes_down_repeated_positive_logits():
    logits = mx.array(np.arange(VOCAB, dtype=np.float32) - 10.0)[None]  # -10..21
    window = mx.array([3, 7, 20], mx.int32)
    out = np.array(_compiled_penalty(logits.shape[-1])(logits, window, mx.array(PENALTY, mx.float32)))[0]
    expected = np.arange(VOCAB, dtype=np.float32) - 10.0
    for tid in (3, 7, 20):
        expected[tid] = (
            expected[tid] / PENALTY if expected[tid] > 0 else expected[tid] * PENALTY
        )
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_penalty_multiplies_negative_logits():
    logits = mx.array(np.full(VOCAB, -2.0, np.float32))[None]
    window = mx.array([5], mx.int32)
    out = np.array(_compiled_penalty(logits.shape[-1])(logits, window, mx.array(PENALTY, mx.float32)))[0]
    assert out[5] == pytest.approx(-2.0 * PENALTY, abs=1e-5)
    assert out[6] == pytest.approx(-2.0, abs=1e-6)


def test_penalty_leaves_unseen_ids_and_rank_order_alone():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(VOCAB).astype(np.float32)
    logits = mx.array(base)[None]
    window = mx.array([0, 31, 15], mx.int32)
    out = np.array(_compiled_penalty(logits.shape[-1])(logits, window, mx.array(PENALTY, mx.float32)))[0]
    # untouched ids identical
    mask = np.ones(VOCAB, bool)
    mask[[0, 31, 15]] = False
    np.testing.assert_allclose(out[mask], base[mask], rtol=1e-6)
    # penalized ids moved toward the tail but never above their unpenalized value
    for tid in (0, 31, 15):
        if base[tid] > 0:
            assert out[tid] < base[tid]
        else:
            assert out[tid] < base[tid]  # negative: multiplied further down
    # no new argmax from outside the window's top: penalizing cannot promote
    unseen = np.argsort(base)[::-1]
    for tid in unseen:
        if tid in (0, 31, 15):
            continue
        for pid in (0, 31, 15):
            assert out[pid] <= base[pid] + 1e-6


def test_penalty_empty_window_is_identity():
    logits = mx.array(np.arange(VOCAB, dtype=np.float32))[None]
    window = mx.full((64,), -1, mx.int32)
    out = np.array(_compiled_penalty(logits.shape[-1])(logits, window, mx.array(PENALTY, mx.float32)))[0]
    np.testing.assert_allclose(out, np.arange(VOCAB, dtype=np.float32), rtol=1e-6)
