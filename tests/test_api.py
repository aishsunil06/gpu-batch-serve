"""Tests for the HTTP surface, driven against a real ASGI transport."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from gpu_batch_serve.api import create_app
from gpu_batch_serve.config import ServingConfig

from conftest import FakeEngine


def _config(**overrides) -> ServingConfig:
    base = {
        "model_id": "fake",
        "precision": "fp32",
        "device": "cpu",
        "max_batch_size": 8,
        "max_queue_delay_ms": 5.0,
    }
    base.update(overrides)
    return ServingConfig(**base)


async def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_embed_returns_a_vector():
    engine = FakeEngine(dimensions=5)
    app = create_app(config=_config(), engine=engine)

    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            response = await client.post("/embed", json={"text": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["dimensions"] == 5
    assert len(body["embedding"]) == 5


async def test_concurrent_requests_share_batches():
    """The HTTP layer must not serialise what the batcher would coalesce."""
    engine = FakeEngine(latency_seconds=0.02)
    app = create_app(config=_config(max_batch_size=16, max_queue_delay_ms=20), engine=engine)

    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            responses = await asyncio.gather(
                *[client.post("/embed", json={"text": f"q{i}"}) for i in range(12)]
            )

    assert all(r.status_code == 200 for r in responses)
    assert max(engine.observed_batches) > 1


async def test_warmup_runs_before_traffic_is_accepted():
    """Every shape the server can produce must be warmed at startup."""
    engine = FakeEngine()
    app = create_app(config=_config(max_batch_size=32), engine=engine)

    async with app.router.lifespan_context(app):
        assert 1 in engine.warmed_shapes
        assert 32 in engine.warmed_shapes


async def test_warmup_is_excluded_from_reported_stats():
    """Warmup traffic must not inflate the batching counters."""
    engine = FakeEngine()
    app = create_app(config=_config(max_batch_size=4), engine=engine)

    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            stats = (await client.get("/stats")).json()

    assert stats["batching"]["requests"] == 0


async def test_stats_reports_the_configuration_under_test():
    app = create_app(
        config=_config(max_batch_size=32, max_queue_delay_ms=7.5), engine=FakeEngine()
    )

    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            stats = (await client.get("/stats")).json()

    assert stats["config"]["max_batch_size"] == 32
    assert stats["config"]["max_queue_delay_ms"] == 7.5


async def test_healthz_reports_not_ready_before_lifespan():
    app = create_app(config=_config(), engine=FakeEngine())

    async with await _client(app) as client:
        body = (await client.get("/healthz")).json()

    assert body["ready"] is False


async def test_empty_text_is_rejected():
    app = create_app(config=_config(), engine=FakeEngine())

    async with app.router.lifespan_context(app):
        async with await _client(app) as client:
            response = await client.post("/embed", json={"text": ""})

    assert response.status_code == 422


async def test_requests_before_readiness_get_503():
    app = create_app(config=_config(), engine=FakeEngine())

    async with await _client(app) as client:
        response = await client.post("/embed", json={"text": "hello"})

    assert response.status_code == 503


def test_fp16_on_cpu_is_rejected():
    with pytest.raises(ValueError):
        _config(precision="fp16", device="cpu").validate()
