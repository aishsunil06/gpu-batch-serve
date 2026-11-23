"""Closed-loop load generator.

Closed loop, not open loop: a fixed number of virtual clients each send a
request, wait for its response, and immediately send the next. That is the
right model here because it keeps exactly ``concurrency`` requests in
flight, which is the independent variable the batching experiment needs to
control. An open-loop generator at a fixed arrival rate would let a slow
arm accumulate an unbounded queue and report latency that says more about
the backlog than the server.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import List, Sequence

import httpx


@dataclass
class LoadResult:
    """Outcome of one measurement window."""

    concurrency: int
    completed: int
    failed: int
    wall_seconds: float
    latencies_ms: List[float]

    @property
    def throughput_rps(self) -> float:
        if self.wall_seconds <= 0:
            return 0.0
        return self.completed / self.wall_seconds

    def percentile(self, fraction: float) -> float:
        return percentile(self.latencies_ms, fraction)

    def as_dict(self) -> dict:
        return {
            "concurrency": self.concurrency,
            "completed": self.completed,
            "failed": self.failed,
            "wall_seconds": round(self.wall_seconds, 3),
            "throughput_rps": round(self.throughput_rps, 2),
            "latency_ms": {
                "p50": round(self.percentile(0.50), 2),
                "p95": round(self.percentile(0.95), 2),
                "p99": round(self.percentile(0.99), 2),
                "mean": round(statistics.fmean(self.latencies_ms), 2)
                if self.latencies_ms
                else 0.0,
            },
        }


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Nearest-rank rather than interpolated, because an interpolated p99 over
    a few hundred samples invents a value that no request experienced.
    """
    if not values:
        return 0.0
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(values)
    rank = max(1, int(round(fraction * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


async def _client_loop(
    client: httpx.AsyncClient,
    url: str,
    texts: Sequence[str],
    start_index: int,
    stride: int,
    deadline: float,
    latencies: List[float],
    failures: List[int],
) -> None:
    index = start_index
    while time.perf_counter() < deadline:
        text = texts[index % len(texts)]
        index += stride
        started = time.perf_counter()
        try:
            response = await client.post(url, json={"text": text})
            response.raise_for_status()
        except Exception:  # noqa: BLE001 - counted, not raised
            failures.append(1)
            continue
        latencies.append((time.perf_counter() - started) * 1000.0)


async def run_load(
    base_url: str,
    texts: Sequence[str],
    concurrency: int,
    duration_seconds: float,
    warmup_seconds: float = 2.0,
) -> LoadResult:
    """Drive the server at a fixed concurrency and measure the result.

    Args:
        base_url: Root URL of a running server.
        texts: Workload to cycle through. Each client walks the list with a
            different offset so clients do not send identical text in
            lockstep, which would be an unrealistically batch-friendly and
            cache-friendly pattern.
        concurrency: Number of virtual clients held in flight.
        duration_seconds: Length of the measured window.
        warmup_seconds: Unmeasured window run first, then discarded.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    url = f"{base_url.rstrip('/')}/embed"
    limits = httpx.Limits(
        max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8
    )
    timeout = httpx.Timeout(60.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        if warmup_seconds > 0:
            await _run_window(client, url, texts, concurrency, warmup_seconds)
            await client.post(f"{base_url.rstrip('/')}/reset-stats")

        return await _run_window(client, url, texts, concurrency, duration_seconds)


async def _run_window(
    client: httpx.AsyncClient,
    url: str,
    texts: Sequence[str],
    concurrency: int,
    duration_seconds: float,
) -> LoadResult:
    latencies: List[float] = []
    failures: List[int] = []
    started = time.perf_counter()
    deadline = started + duration_seconds

    await asyncio.gather(
        *[
            _client_loop(
                client,
                url,
                texts,
                start_index=worker,
                stride=max(1, concurrency),
                deadline=deadline,
                latencies=latencies,
                failures=failures,
            )
            for worker in range(concurrency)
        ]
    )

    wall = time.perf_counter() - started
    return LoadResult(
        concurrency=concurrency,
        completed=len(latencies),
        failed=len(failures),
        wall_seconds=wall,
        latencies_ms=latencies,
    )


async def wait_for_ready(base_url: str, timeout_seconds: float = 300.0) -> bool:
    """Poll ``/healthz`` until the model is loaded and warmed."""
    url = f"{base_url.rstrip('/')}/healthz"
    deadline = time.perf_counter() + timeout_seconds
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.perf_counter() < deadline:
            try:
                response = await client.get(url)
                if response.status_code == 200 and response.json().get("ready"):
                    return True
            except Exception:  # noqa: BLE001 - server not up yet
                pass
            await asyncio.sleep(1.0)
    return False
