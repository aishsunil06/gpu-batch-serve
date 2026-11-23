"""ONNX Runtime engine over pre-tokenized input.

This engine exists so the local service can execute byte-identical weights
to the ones Triton and Ray Serve load. Comparing a Torch service against a
Triton service would confound runtime with serving architecture, and the
question here is only about serving architecture.

It takes token arrays rather than strings. Tokenization is CPU work that
happens before any batching decision, so leaving it in the request path
would add a per-request cost to every arm and shrink the measured
difference between them for a reason unrelated to serving.
"""

from __future__ import annotations

import threading
from typing import List, Sequence

import numpy as np

TokenizedRequest = Sequence[Sequence[int]]


class OnnxPooledEngine:
    """Runs the pooled encoder graph on ONNX Runtime.

    Args:
        model_path: Path to the exported ``model.onnx``.
        sequence_length: Fixed sequence length the graph was exported with.
        intra_op_threads: Threads ONNX Runtime may use inside one operator.
            Pinned rather than left to default so a run is reproducible and
            so the local arm does not quietly get a different thread budget
            than the one Triton is configured with.
    """

    def __init__(
        self,
        model_path: str,
        sequence_length: int = 64,
        intra_op_threads: int = 0,
    ) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        if intra_op_threads > 0:
            options.intra_op_num_threads = intra_op_threads
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self._session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._sequence_length = sequence_length
        self._lock = threading.Lock()
        self._model_path = model_path

        names = {i.name for i in self._session.get_inputs()}
        expected = {"input_ids", "attention_mask", "token_type_ids"}
        if names != expected:
            raise ValueError(
                f"graph inputs {sorted(names)} do not match expected {sorted(expected)}"
            )

    @property
    def description(self) -> str:
        return f"onnxruntime cpu, seq={self._sequence_length}, {self._model_path}"

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    def warmup(self, batch_sizes: Sequence[int]) -> None:
        """Run one pass per distinct batch shape.

        ONNX Runtime allocates and plans per shape on first sight, so an
        unwarmed first request at a new batch size pays a cost that has
        nothing to do with steady-state serving.
        """
        for size in sorted({max(1, int(s)) for s in batch_sizes}):
            ids = [[101] + [0] * (self._sequence_length - 1)] * size
            mask = [[1] + [0] * (self._sequence_length - 1)] * size
            self.embed_tokens(ids, mask)

    def embed_tokens(
        self,
        input_ids: TokenizedRequest,
        attention_mask: TokenizedRequest,
        token_type_ids: TokenizedRequest | None = None,
    ) -> List[List[float]]:
        """Embed a batch of already-tokenized requests."""
        if not input_ids:
            return []

        ids = np.asarray(input_ids, dtype=np.int64)
        mask = np.asarray(attention_mask, dtype=np.int64)
        if token_type_ids is None:
            types = np.zeros_like(ids)
        else:
            types = np.asarray(token_type_ids, dtype=np.int64)

        if ids.shape[1] != self._sequence_length:
            raise ValueError(
                f"expected sequence length {self._sequence_length}, got {ids.shape[1]}"
            )

        with self._lock:
            output = self._session.run(
                ["embedding"],
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "token_type_ids": types,
                },
            )[0]
        return output.tolist()
