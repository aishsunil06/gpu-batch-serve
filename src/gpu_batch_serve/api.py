"""HTTP surface for the embedding service.

One request carries one text. That is deliberate and is the point of the
project: a client that could batch its own work would not need a batching
server, so the interesting case is many independent clients each asking
for one thing, which is what a recommendation or ranking service actually
sees.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .batcher import DynamicBatcher
from .config import ServingConfig
from .engine import build_engine


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192)


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimensions: int


class StatsResponse(BaseModel):
    config: dict
    batching: dict


def create_app(config: ServingConfig | None = None, engine=None) -> FastAPI:
    """Build the ASGI app.

    Args:
        config: Serving configuration. Read from the environment when
            omitted, which is how the benchmark varies arms.
        engine: Injected for tests, so the HTTP and batching layers can be
            exercised without a GPU or a model download.
    """
    resolved = config or ServingConfig.from_env()
    resolved.validate()

    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        built = engine if engine is not None else build_engine(resolved)
        batcher = DynamicBatcher(
            engine=built,
            max_batch_size=resolved.max_batch_size,
            max_queue_delay_ms=resolved.max_queue_delay_ms,
        )
        await batcher.start()
        # Warm every shape before the port accepts traffic, so no client
        # ever measures first-call algorithm selection.
        built.warmup([1, resolved.max_batch_size])
        batcher.reset_stats()
        state["engine"] = built
        state["batcher"] = batcher
        try:
            yield
        finally:
            await batcher.stop()

    app = FastAPI(title="gpu-batch-serve", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict:
        ready = "batcher" in state
        return {"status": "ok" if ready else "starting", "ready": ready}

    @app.get("/stats", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        batcher: DynamicBatcher | None = state.get("batcher")
        if batcher is None:
            raise HTTPException(status_code=503, detail="not ready")
        return StatsResponse(
            config={
                "model_id": resolved.model_id,
                "precision": resolved.precision,
                "device": resolved.device,
                "max_batch_size": resolved.max_batch_size,
                "max_queue_delay_ms": resolved.max_queue_delay_ms,
            },
            batching=batcher.stats.as_dict(),
        )

    @app.post("/reset-stats")
    async def reset_stats() -> dict:
        batcher: DynamicBatcher | None = state.get("batcher")
        if batcher is None:
            raise HTTPException(status_code=503, detail="not ready")
        batcher.reset_stats()
        return {"status": "reset"}

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: EmbedRequest) -> EmbedResponse:
        batcher: DynamicBatcher | None = state.get("batcher")
        if batcher is None:
            raise HTTPException(status_code=503, detail="not ready")
        vector = await batcher.submit(request.text)
        return EmbedResponse(embedding=vector, dimensions=len(vector))

    return app


app = None


def get_app() -> FastAPI:
    """Entry point for ``uvicorn gpu_batch_serve.api:get_app --factory``."""
    return create_app()
