"""Configuration for the serving process.

Every knob the benchmark sweeps is read from the environment so that one
arm of the experiment differs from another only by environment, never by
code path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class ServingConfig:
    """Resolved serving configuration.

    Attributes:
        model_id: Hugging Face id of the encoder to serve.
        precision: ``fp32`` or ``fp16``. ``fp16`` is GPU-only.
        device: ``cuda`` or ``cpu``.
        max_batch_size: Upper bound on requests coalesced into one forward
            pass. ``1`` disables batching.
        max_queue_delay_ms: How long the batcher waits for a batch to fill
            before running a short one. ``0`` means run as soon as anything
            is available.
        max_sequence_length: Tokenizer truncation length.
    """

    model_id: str = DEFAULT_MODEL
    precision: str = "fp32"
    device: str = "cuda"
    max_batch_size: int = 1
    max_queue_delay_ms: float = 0.0
    max_sequence_length: int = 128

    @property
    def batching_enabled(self) -> bool:
        return self.max_batch_size > 1

    @classmethod
    def from_env(cls) -> "ServingConfig":
        return cls(
            model_id=os.environ.get("GBS_MODEL_ID", DEFAULT_MODEL),
            precision=os.environ.get("GBS_PRECISION", "fp32").lower(),
            device=os.environ.get("GBS_DEVICE", "cuda").lower(),
            max_batch_size=_int("GBS_MAX_BATCH_SIZE", 1),
            max_queue_delay_ms=_float("GBS_MAX_QUEUE_DELAY_MS", 0.0),
            max_sequence_length=_int("GBS_MAX_SEQUENCE_LENGTH", 128),
        )

    def validate(self) -> None:
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError(f"unknown precision {self.precision!r}")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError(f"unknown device {self.device!r}")
        if self.precision == "fp16" and self.device == "cpu":
            raise ValueError("fp16 is not supported on cpu in this project")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_queue_delay_ms < 0:
            raise ValueError("max_queue_delay_ms must be >= 0")
