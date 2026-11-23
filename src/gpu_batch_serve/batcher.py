"""Dynamic batching: coalesce concurrent requests into one forward pass.

The whole experiment lives here. A request does not run on arrival; it
joins a queue, and a single collector loop decides when enough has
accumulated to be worth a forward pass. Two knobs govern that decision:

  * ``max_batch_size`` caps how many requests one pass may carry.
  * ``max_queue_delay_ms`` caps how long the first waiting request is made
    to wait for company.

The second knob is the interesting one, because it is a latency cost paid
unconditionally in exchange for a throughput gain that only materialises
when traffic is concurrent enough to fill batches.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Sequence

from .engine import Engine


@dataclass
class BatchStats:
    """Observed batching behaviour over the life of the batcher.

    Recorded so that a throughput result can be explained rather than just
    reported: a batching arm that never actually filled a batch looks
    identical to a no-batching arm in the latency table, and only these
    counters distinguish "batching did not help" from "batching did not
    happen".
    """

    batches: int = 0
    requests: int = 0
    batch_sizes: List[int] = field(default_factory=list)
    inference_seconds: float = 0.0

    @property
    def mean_batch_size(self) -> float:
        if self.batches == 0:
            return 0.0
        return self.requests / self.batches

    @property
    def max_observed_batch_size(self) -> int:
        return max(self.batch_sizes) if self.batch_sizes else 0

    def as_dict(self) -> dict:
        return {
            "batches": self.batches,
            "requests": self.requests,
            "mean_batch_size": round(self.mean_batch_size, 3),
            "max_observed_batch_size": self.max_observed_batch_size,
            "inference_seconds": round(self.inference_seconds, 4),
        }


@dataclass
class _Waiter:
    """One queued request: its payload and the future awaiting its result.

    The payload is typed loosely because the collector is deliberately
    agnostic to it. It is a string for the text service and a tokenized
    record for the server comparison.
    """

    text: object
    future: asyncio.Future


class DynamicBatcher:
    """Queue requests, run them in coalesced batches.

    Args:
        engine: The thing that actually runs inference.
        max_batch_size: Largest batch the collector will build. ``1``
            disables coalescing entirely.
        max_queue_delay_ms: Grace period the collector grants a partially
            filled batch before running it anyway.
        batch_fn: Optional callable taking the list of queued items and
            returning one result per item, used instead of ``engine.embed``.
            This is what lets the same collector queue pre-tokenized arrays
            for the server comparison and plain strings for the original
            service, without the queueing logic knowing which it holds.
    """

    def __init__(
        self,
        engine: Engine,
        max_batch_size: int = 1,
        max_queue_delay_ms: float = 0.0,
        batch_fn=None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_queue_delay_ms < 0:
            raise ValueError("max_queue_delay_ms must be >= 0")

        self._engine = engine
        self._batch_fn = batch_fn
        self._max_batch_size = max_batch_size
        self._max_queue_delay = max_queue_delay_ms / 1000.0
        self._queue: asyncio.Queue[_Waiter] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._stats = BatchStats()

    @property
    def stats(self) -> BatchStats:
        return self._stats

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._collect_forever())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def reset_stats(self) -> None:
        self._stats = BatchStats()

    async def submit(self, text: object) -> List[float]:
        """Enqueue one item and wait for its result."""
        loop = asyncio.get_running_loop()
        waiter = _Waiter(text=text, future=loop.create_future())
        await self._queue.put(waiter)
        return await waiter.future

    async def _collect_forever(self) -> None:
        while True:
            batch = await self._gather_batch()
            if batch:
                await self._run_batch(batch)

    async def _gather_batch(self) -> List[_Waiter]:
        """Block for the first request, then try to accumulate company.

        The delay budget is measured from the arrival of the *first*
        request, not reset per arrival, so a steady trickle cannot hold a
        batch open indefinitely and starve the request that opened it.
        """
        first = await self._queue.get()
        batch = [first]

        if self._max_batch_size == 1:
            return batch

        deadline = time.perf_counter() + self._max_queue_delay
        while len(batch) < self._max_batch_size:
            if self._max_queue_delay == 0:
                # No grace period: take only what is already queued.
                try:
                    batch.append(self._queue.get_nowait())
                    continue
                except asyncio.QueueEmpty:
                    break

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                waiter = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            batch.append(waiter)

        return batch

    async def _run_batch(self, batch: Sequence[_Waiter]) -> None:
        texts = [w.text for w in batch]
        loop = asyncio.get_running_loop()
        runner = self._batch_fn if self._batch_fn is not None else self._engine.embed
        started = time.perf_counter()
        try:
            # Torch and ONNX Runtime both release the GIL inside the forward
            # pass, so running it in the default executor keeps the event
            # loop free to accept the arrivals that form the next batch.
            vectors = await loop.run_in_executor(None, runner, texts)
        except Exception as exc:  # noqa: BLE001 - propagated to every caller
            for waiter in batch:
                if not waiter.future.done():
                    waiter.future.set_exception(exc)
            return
        elapsed = time.perf_counter() - started

        self._stats.batches += 1
        self._stats.requests += len(batch)
        self._stats.batch_sizes.append(len(batch))
        self._stats.inference_seconds += elapsed

        for waiter, vector in zip(batch, vectors):
            if not waiter.future.done():
                waiter.future.set_result(vector)
