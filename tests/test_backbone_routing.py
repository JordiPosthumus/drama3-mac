"""Regression test for the 1-row prefill chunk bug.

A prompt whose row count is 1 mod prefill_chunk_size (128) makes the final
prefill chunk exactly one row; that call must take the eager prefill path,
not the compiled decode path (which requires a position).  The routing
predicate is tested directly; a real-model end-to-end check lives in
scripts/regress_1row_prefill.py (needs a converted checkpoint).
"""

from __future__ import annotations

import mlx.core as mx

from drama3_mac.backbone import is_decode_step


def test_decode_call_has_position():
    assert is_decode_step(1, True, True, mx.array(5, mx.int32))


def test_one_row_prefill_chunk_is_not_decode():
    # the exact crash shape: 1-row continue_state call with no position
    assert not is_decode_step(1, True, True, None)


def test_multi_row_prefill_is_not_decode():
    assert not is_decode_step(128, False, True, None)
    assert not is_decode_step(31, False, False, None)
    assert not is_decode_step(1, False, True, mx.array(3, mx.int32))
