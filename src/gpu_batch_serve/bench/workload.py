"""The texts the benchmark sends.

Length matters to a batching experiment, because a padded batch runs at
the length of its longest member. A workload of uniform-length strings
would flatter batching in a way real traffic does not, so the default
workload is deliberately mixed-length and drawn deterministically from a
seed.
"""

from __future__ import annotations

import random
from typing import List

_FRAGMENTS = [
    "how do i",
    "what is the best way to",
    "can someone explain",
    "why does",
    "is it true that",
    "what happens when",
    "how long does it take to",
    "which is better for",
]

_TOPICS = [
    "serve a machine learning model on a gpu",
    "reduce tail latency in a web service",
    "choose between batch and streaming inference",
    "quantize a transformer without losing accuracy",
    "scale a kubernetes deployment under load",
    "debug a slow database query",
    "design a recommendation system for a large feed",
    "measure throughput of an inference server",
    "pick an embedding model for semantic search",
    "handle backpressure in a request queue",
]


def build_workload(size: int, seed: int = 17) -> List[str]:
    """Return ``size`` deterministic mixed-length query strings.

    Args:
        size: How many strings to produce.
        seed: Seed for the generator. Arms of the experiment share a seed
            so every arm meets exactly the same input.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    rng = random.Random(seed)
    texts: List[str] = []
    for index in range(size):
        fragment = rng.choice(_FRAGMENTS)
        topic = rng.choice(_TOPICS)
        # A third of the workload is a long query, which is what makes
        # padding cost visible in the batched arms.
        if index % 3 == 0:
            extra = rng.choice(_TOPICS)
            texts.append(f"{fragment} {topic}, and also {extra}?")
        else:
            texts.append(f"{fragment} {topic}?")
    return texts
