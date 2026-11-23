"""Arm 2: Ray Serve, using its own dynamic batching.

Ray Serve implements the same policy as the hand-rolled collector, through
``@serve.batch``, with two knobs that correspond directly:

    max_batch_size        <-> DynamicBatcher.max_batch_size
    batch_wait_timeout_s  <-> DynamicBatcher.max_queue_delay_ms

Set identically, the comparison isolates implementation quality from
policy. Ray additionally brings replica autoscaling, which the hand-rolled
arm has no equivalent for at all, so replica count is held at 1 for the
headline comparison and swept separately.

Run:
    python serving/ray/app.py --max-batch-size 32 --batch-wait-ms 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_ONNX = str(
    ROOT / "serving" / "triton" / "model_repository" / "minilm_pooled" / "1" / "model.onnx"
)


def build_deployment(
    onnx_path: str,
    max_batch_size: int,
    batch_wait_ms: float,
    sequence_length: int,
    num_replicas: int,
):
    from ray import serve
    from starlette.requests import Request

    from gpu_batch_serve.onnx_engine import OnnxPooledEngine

    @serve.deployment(
        num_replicas=num_replicas,
        ray_actor_options={"num_cpus": 1},
    )
    class EmbeddingDeployment:
        def __init__(self) -> None:
            self.engine = OnnxPooledEngine(
                model_path=onnx_path, sequence_length=sequence_length
            )
            # Warm every shape before the replica reports ready, matching
            # the other two arms. Without this, Ray's first measured
            # request pays ONNX Runtime's per-shape planning cost and the
            # arm looks worse for a reason unrelated to Ray.
            self.engine.warmup([1, max_batch_size])
            self.batches = 0
            self.requests = 0

        @serve.batch(
            max_batch_size=max_batch_size,
            batch_wait_timeout_s=batch_wait_ms / 1000.0,
        )
        async def _embed_batch(self, requests: List[dict]) -> List[List[float]]:
            self.batches += 1
            self.requests += len(requests)
            return self.engine.embed_tokens(
                [r["input_ids"] for r in requests],
                [r["attention_mask"] for r in requests],
                [r.get("token_type_ids") or [0] * sequence_length for r in requests],
            )

        async def __call__(self, http_request: Request):
            path = http_request.url.path.rstrip("/")

            if path.endswith("/healthz"):
                return {"status": "ok", "ready": True}

            if path.endswith("/stats"):
                mean = self.requests / self.batches if self.batches else 0.0
                return {
                    "config": {
                        "arm": "ray",
                        "max_batch_size": max_batch_size,
                        "max_queue_delay_ms": batch_wait_ms,
                        "num_replicas": num_replicas,
                        "sequence_length": sequence_length,
                    },
                    "batching": {
                        "batches": self.batches,
                        "requests": self.requests,
                        "mean_batch_size": round(mean, 3),
                    },
                }

            if path.endswith("/reset-stats"):
                self.batches = 0
                self.requests = 0
                return {"status": "reset"}

            body = await http_request.json()
            vector = await self._embed_batch(body)
            return {"embedding": vector, "dimensions": len(vector)}

    return EmbeddingDeployment.bind()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", default=os.environ.get("GBS_ONNX_PATH", DEFAULT_ONNX))
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=float, default=5.0)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--num-replicas", type=int, default=1)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--block", action="store_true", help="serve until interrupted")
    args = parser.parse_args(argv)

    import ray
    from ray import serve

    ray.init(ignore_reinit_error=True, logging_level="warning")
    serve.start(http_options={"host": "127.0.0.1", "port": args.port})
    serve.run(
        build_deployment(
            onnx_path=args.onnx_path,
            max_batch_size=args.max_batch_size,
            batch_wait_ms=args.batch_wait_ms,
            sequence_length=args.sequence_length,
            num_replicas=args.num_replicas,
        ),
        route_prefix="/",
    )
    print(
        f"ray serve listening on http://127.0.0.1:{args.port} "
        f"(batch={args.max_batch_size}, wait={args.batch_wait_ms}ms, "
        f"replicas={args.num_replicas})",
        flush=True,
    )

    if args.block:
        import time

        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
