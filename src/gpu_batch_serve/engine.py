"""The embedding engine: a batch of strings in, a batch of vectors out.

The engine is deliberately synchronous and batch-shaped. It knows nothing
about queueing, concurrency or HTTP; deciding *what* goes into a batch is
the batcher's job, and this module only runs the batch it is handed. That
split is what lets the benchmark change batching policy without touching
the inference path.
"""

from __future__ import annotations

import threading
from typing import List, Protocol, Sequence

from .config import ServingConfig


class Engine(Protocol):
    """Anything that can turn texts into embedding vectors."""

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def warmup(self, batch_sizes: Sequence[int]) -> None:
        ...

    @property
    def description(self) -> str:
        ...


class TorchEncoderEngine:
    """Mean-pooled sentence embeddings from a Hugging Face encoder.

    A single CUDA stream is shared by every caller, so a lock serialises
    forward passes. That is intentional: it mirrors how one model instance
    on one GPU actually behaves, and it is the thing concurrent-instance
    experiments are measured against.
    """

    def __init__(self, config: ServingConfig) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        config.validate()
        self._config = config
        self._torch = torch

        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "GBS_DEVICE=cuda but no CUDA device is visible. On Colab, "
                "set Runtime > Change runtime type > T4 GPU."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        model = AutoModel.from_pretrained(config.model_id)
        model.eval()

        self._dtype = torch.float16 if config.precision == "fp16" else torch.float32
        model = model.to(device=config.device, dtype=self._dtype)
        self._model = model
        self._lock = threading.Lock()

    @property
    def description(self) -> str:
        return (
            f"{self._config.model_id} on {self._config.device} "
            f"in {self._config.precision}"
        )

    def warmup(self, batch_sizes: Sequence[int]) -> None:
        """Run one forward pass per distinct batch shape.

        cuDNN picks algorithms per shape on first sight and caches them, so
        an unwarmed first request pays a cost that has nothing to do with
        steady-state serving. Warming every shape the run will use keeps
        that cost out of the measurement.
        """
        for size in sorted({max(1, int(s)) for s in batch_sizes}):
            self.embed(["warmup text"] * size)
        if self._config.device == "cuda":
            self._torch.cuda.synchronize()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        torch = self._torch
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._config.max_sequence_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._config.device) for k, v in encoded.items()}

        with self._lock:
            with torch.inference_mode():
                output = self._model(**encoded)
                pooled = _mean_pool(
                    torch, output.last_hidden_state, encoded["attention_mask"]
                )
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                # Synchronise inside the lock so the measured duration is the
                # GPU's, not just the launch queue's.
                if self._config.device == "cuda":
                    torch.cuda.synchronize()
                return pooled.float().cpu().tolist()


def _mean_pool(torch, hidden_state, attention_mask):
    """Average token vectors, ignoring padding.

    Padding tokens carry no meaning, so including them would make an
    embedding depend on the length of whatever else shared its batch. That
    would make results batch-size dependent and the experiment invalid.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def build_engine(config: ServingConfig) -> Engine:
    return TorchEncoderEngine(config)
