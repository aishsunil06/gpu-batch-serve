"""Tests for the dynamic batcher.

These are the load-bearing tests of the project. The benchmark's whole
claim is that arm B coalesced requests and arm A did not, and that claim
is only meaningful if the batcher provably does what its configuration
says. A batcher that silently ran everything one at a time would produce a
perfectly plausible-looking results table.
"""

from __future__ import annotations

import asyncio

import pytest

from gpu_batch_serve.batcher import DynamicBatcher

from conftest import FakeEngine


@pytest.fixture
async def started_batcher():
    created = []

    async def _make(engine, **kwargs):
        batcher = DynamicBatcher(engine=engine, **kwargs)
        await batcher.start()
        created.append(batcher)
        return batcher

    yield _make

    for batcher in created:
        await batcher.stop()


async def test_single_request_returns_its_vector(started_batcher):
    engine = FakeEngine(dimensions=3)
    batcher = await started_batcher(engine, max_batch_size=8, max_queue_delay_ms=5)

    vector = await batcher.submit("hello")

    assert len(vector) == 3
    assert engine.observed_batches == [1]


async def test_batch_size_one_never_coalesces(started_batcher):
    """With the cap at 1, concurrent arrivals must still run separately."""
    engine = FakeEngine(latency_seconds=0.01)
    batcher = await started_batcher(engine, max_batch_size=1, max_queue_delay_ms=50)

    await asyncio.gather(*[batcher.submit(f"text {i}") for i in range(10)])

    assert engine.observed_batches == [1] * 10
    assert batcher.stats.max_observed_batch_size == 1


async def test_concurrent_arrivals_are_coalesced(started_batcher):
    """Ten simultaneous requests under a cap of 8 must not run as ten passes."""
    engine = FakeEngine(latency_seconds=0.02)
    batcher = await started_batcher(engine, max_batch_size=8, max_queue_delay_ms=20)

    results = await asyncio.gather(*[batcher.submit(f"text {i}") for i in range(10)])

    assert len(results) == 10
    assert batcher.stats.requests == 10
    assert batcher.stats.batches < 10
    assert batcher.stats.max_observed_batch_size > 1


async def test_batch_never_exceeds_the_cap(started_batcher):
    engine = FakeEngine(latency_seconds=0.02)
    batcher = await started_batcher(engine, max_batch_size=4, max_queue_delay_ms=25)

    await asyncio.gather(*[batcher.submit(f"text {i}") for i in range(40)])

    assert engine.observed_batches
    assert max(engine.observed_batches) <= 4


async def test_zero_delay_takes_only_what_is_already_queued(started_batcher):
    """A lone request must not wait when there is no grace period."""
    engine = FakeEngine()
    batcher = await started_batcher(engine, max_batch_size=32, max_queue_delay_ms=0)

    started = asyncio.get_running_loop().time()
    await batcher.submit("solo")
    elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000

    assert engine.observed_batches == [1]
    assert elapsed_ms < 50


async def test_grace_period_bounds_how_long_a_lone_request_waits(started_batcher):
    """A lone request pays the delay once, not repeatedly."""
    engine = FakeEngine()
    batcher = await started_batcher(engine, max_batch_size=32, max_queue_delay_ms=30)

    started = asyncio.get_running_loop().time()
    await batcher.submit("solo")
    elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000

    # It must wait at least a little, and must not wait several multiples
    # of the configured budget.
    assert elapsed_ms >= 20
    assert elapsed_ms < 300


async def test_engine_failure_reaches_every_caller_in_the_batch(started_batcher):
    """One bad forward pass must fail its whole batch, not hang it."""

    class ExplodingEngine(FakeEngine):
        def embed(self, texts):
            raise RuntimeError("cuda out of memory")

    batcher = await started_batcher(
        ExplodingEngine(), max_batch_size=8, max_queue_delay_ms=10
    )

    results = await asyncio.gather(
        *[batcher.submit(f"text {i}") for i in range(5)],
        return_exceptions=True,
    )

    assert len(results) == 5
    assert all(isinstance(r, RuntimeError) for r in results)


async def test_stats_reset_clears_counters(started_batcher):
    engine = FakeEngine()
    batcher = await started_batcher(engine, max_batch_size=8, max_queue_delay_ms=5)

    await batcher.submit("one")
    assert batcher.stats.requests == 1

    batcher.reset_stats()

    assert batcher.stats.requests == 0
    assert batcher.stats.batches == 0
    assert batcher.stats.mean_batch_size == 0.0


def test_rejects_invalid_configuration():
    engine = FakeEngine()
    with pytest.raises(ValueError):
        DynamicBatcher(engine=engine, max_batch_size=0)
    with pytest.raises(ValueError):
        DynamicBatcher(engine=engine, max_queue_delay_ms=-1)
