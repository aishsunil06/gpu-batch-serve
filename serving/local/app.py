"""Arm 1: the hand-rolled batcher, serving pre-tokenized requests.

This is the control. It runs the same ONNX graph Triton and Ray Serve load,
behind the ~60-line collector in ``gpu_batch_serve.batcher``, so the only
thing that distinguishes it from the other two arms is the implementation
of the batching policy rather than the model, the runtime, or the work.

Run:
    GBS_ONNX_PATH=serving/triton/model_repository/minilm_pooled/1/model.onnx \
    python -m uvicorn serving.local.app:get_app --factory --port 8100
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from gpu_batch_serve.batcher import DynamicBatcher
from gpu_batch_serve.onnx_engine import OnnxPooledEngine

DEFAULT_ONNX = str(
    Path(__file__).resolve().parents[2]
    / "serving"
    / "triton"
    / "model_repository"
    / "minilm_pooled"
    / "1"
    / "model.onnx"
)


class TokenRequest(BaseModel):
    input_ids: List[int] = Field(..., min_length=1)
    attention_mask: List[int] = Field(..., min_length=1)
    token_type_ids: List[int] | None = None


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimensions: int


def create_app(
    onnx_path: str | None = None,
    max_batch_size: int | None = None,
    max_queue_delay_ms: float | None = None,
    sequence_length: int | None = None,
) -> FastAPI:
    resolved_path = onnx_path or os.environ.get("GBS_ONNX_PATH", DEFAULT_ONNX)
    resolved_batch = (
        max_batch_size
        if max_batch_size is not None
        else int(os.environ.get("GBS_MAX_BATCH_SIZE", "32"))
    )
    resolved_delay = (
        max_queue_delay_ms
        if max_queue_delay_ms is not None
        else float(os.environ.get("GBS_MAX_QUEUE_DELAY_MS", "5"))
    )
    resolved_seq = (
        sequence_length
        if sequence_length is not None
        else int(os.environ.get("GBS_SEQUENCE_LENGTH", "64"))
    )

    state: dict = {}

    def _run_batch(items: List[TokenRequest]) -> List[List[float]]:
        engine: OnnxPooledEngine = state["engine"]
        return engine.embed_tokens(
            [item.input_ids for item in items],
            [item.attention_mask for item in items],
            [
                item.token_type_ids or [0] * resolved_seq
                for item in items
            ],
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = OnnxPooledEngine(
            model_path=resolved_path, sequence_length=resolved_seq
        )
        state["engine"] = engine
        batcher = DynamicBatcher(
            engine=engine,
            max_batch_size=resolved_batch,
            max_queue_delay_ms=resolved_delay,
            batch_fn=_run_batch,
        )
        await batcher.start()
        engine.warmup([1, resolved_batch])
        batcher.reset_stats()
        state["batcher"] = batcher
        try:
            yield
        finally:
            await batcher.stop()

    app = FastAPI(title="gpu-batch-serve local arm", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok" if "batcher" in state else "starting", "ready": "batcher" in state}

    @app.get("/stats")
    async def stats() -> dict:
        batcher: DynamicBatcher | None = state.get("batcher")
        if batcher is None:
            raise HTTPException(status_code=503, detail="not ready")
        return {
            "config": {
                "arm": "local",
                "max_batch_size": resolved_batch,
                "max_queue_delay_ms": resolved_delay,
                "sequence_length": resolved_seq,
            },
            "batching": batcher.stats.as_dict(),
        }

    @app.post("/reset-stats")
    async def reset_stats() -> dict:
        batcher: DynamicBatcher | None = state.get("batcher")
        if batcher is None:
            raise HTTPException(status_code=503, detail="not ready")
        batcher.reset_stats()
        return {"status": "reset"}

    @app.post("/embed-tokens", response_model=EmbedResponse)
    async def embed_tokens(request: TokenRequest) -> EmbedResponse:
        batcher: DynamicBatcher | None = state.get("batcher")
        if batcher is None:
            raise HTTPException(status_code=503, detail="not ready")
        if len(request.input_ids) != resolved_seq:
            raise HTTPException(
                status_code=422,
                detail=f"expected {resolved_seq} tokens, got {len(request.input_ids)}",
            )
        vector = await batcher.submit(request)
        return EmbedResponse(embedding=vector, dimensions=len(vector))

    return app


def get_app() -> FastAPI:
    return create_app()
