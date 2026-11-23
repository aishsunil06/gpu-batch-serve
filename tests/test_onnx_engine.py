"""Tests for the ONNX engine and the exported graph.

These tests are skipped when the graph has not been exported, so the suite
stays runnable on a clean checkout. When it has been exported, they are the
check that all three arms really are serving the same model: an engine that
silently dropped the pooling, or that rejected a batch size Triton will
certainly form, would still pass every other test in the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = (
    ROOT / "serving" / "triton" / "model_repository" / "minilm_pooled" / "1" / "model.onnx"
)
FIXTURE_PATH = ROOT / "bench" / "fixtures" / "workload_tokens.json"

pytestmark = pytest.mark.skipif(
    not ONNX_PATH.exists(),
    reason="run scripts/export_onnx.py to exercise the ONNX engine",
)

SEQ = 64


@pytest.fixture(scope="module")
def engine():
    from gpu_batch_serve.onnx_engine import OnnxPooledEngine

    return OnnxPooledEngine(model_path=str(ONNX_PATH), sequence_length=SEQ)


def _tokens(count: int, real: int = 10):
    ids = [[101] * real + [0] * (SEQ - real) for _ in range(count)]
    mask = [[1] * real + [0] * (SEQ - real) for _ in range(count)]
    return ids, mask


def test_returns_one_vector_per_request(engine):
    ids, mask = _tokens(5)

    vectors = engine.embed_tokens(ids, mask)

    assert len(vectors) == 5
    assert all(len(v) == 384 for v in vectors)


def test_vectors_are_l2_normalised(engine):
    """Normalisation is fused into the graph; if it were lost, cosine
    similarity downstream would silently change meaning."""
    ids, mask = _tokens(3)

    vectors = engine.embed_tokens(ids, mask)

    for vector in vectors:
        norm = sum(value * value for value in vector) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-4)


def test_batching_does_not_change_an_embedding(engine):
    """The property the whole experiment rests on.

    If padding were not masked out of the pooling, an embedding would depend
    on the length of whatever else shared its batch, and every throughput
    comparison would be between arms computing different things.
    """
    ids_short, mask_short = _tokens(1, real=8)
    ids_long, mask_long = _tokens(1, real=30)

    alone = engine.embed_tokens(ids_short, mask_short)[0]
    batched = engine.embed_tokens(
        ids_short + ids_long, mask_short + mask_long
    )[0]

    for a, b in zip(alone, batched):
        assert a == pytest.approx(b, abs=1e-5)


def test_accepts_a_batch_size_the_export_never_saw(engine):
    """Triton forms batches up to 32; the graph was exported at 2."""
    ids, mask = _tokens(32)

    vectors = engine.embed_tokens(ids, mask)

    assert len(vectors) == 32


def test_rejects_the_wrong_sequence_length(engine):
    """A fixed-shape graph must fail loudly, not pad silently."""
    with pytest.raises(ValueError, match="sequence length"):
        engine.embed_tokens([[101] * 32], [[1] * 32])


def test_empty_batch_returns_empty(engine):
    assert engine.embed_tokens([], []) == []


def test_warmup_covers_requested_shapes(engine):
    engine.warmup([1, 8])  # must not raise


@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="run scripts/prepare_fixture.py")
def test_serves_the_real_fixture(engine):
    """The fixture every arm sends must actually be servable by the graph."""
    import json

    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert payload["meta"]["sequence_length"] == SEQ

    records = payload["requests"][:8]
    vectors = engine.embed_tokens(
        [r["input_ids"] for r in records],
        [r["attention_mask"] for r in records],
        [r["token_type_ids"] for r in records],
    )

    assert len(vectors) == 8
    # Distinct queries must not collapse to one vector, which would mean the
    # graph is ignoring its input.
    assert len({tuple(round(x, 6) for x in v[:8]) for v in vectors}) > 1
