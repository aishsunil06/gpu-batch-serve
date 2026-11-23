"""Shared fixtures.

A fake engine stands in for the real encoder so the batching and HTTP
layers can be tested on any machine, with no GPU and no model download.
The fake records the exact batch shapes it was handed, which is what the
batching tests assert on.
"""

from __future__ import annotations

import threading
import time
from typing import List, Sequence

import pytest


class FakeEngine:
    """Deterministic stand-in for the real encoder.

    Args:
        dimensions: Width of the vectors it returns.
        latency_seconds: Simulated time for one forward pass, charged per
            batch rather than per item, which is the property that makes
            batching worth anything.
    """

    def __init__(self, dimensions: int = 4, latency_seconds: float = 0.0) -> None:
        self.dimensions = dimensions
        self.latency_seconds = latency_seconds
        self.observed_batches: List[int] = []
        self.warmed_shapes: List[int] = []
        self._lock = threading.Lock()

    @property
    def description(self) -> str:
        return "fake engine"

    def warmup(self, batch_sizes: Sequence[int]) -> None:
        self.warmed_shapes.extend(int(size) for size in batch_sizes)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        with self._lock:
            self.observed_batches.append(len(texts))
        if self.latency_seconds:
            time.sleep(self.latency_seconds)
        return [
            [float(len(text))] * self.dimensions
            for text in texts
        ]


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()
