#!/usr/bin/env python
"""Run the batching experiment and write results to bench/results/.

Each arm runs in its own server process. That costs a model load per arm,
but it guarantees an arm cannot inherit warm state, a cached allocator
pool, or accumulated statistics from the arm before it, which an
in-process sweep cannot promise.

Usage:
    python scripts/run_bench.py                    # full grid
    python scripts/run_bench.py --quick            # one repeat, short windows
    python scripts/run_bench.py --device cpu       # no GPU present
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from gpu_batch_serve.bench.load import run_load, wait_for_ready  # noqa: E402
from gpu_batch_serve.bench.report import (  # noqa: E402
    aggregate_runs,
    render_markdown_table,
)
from gpu_batch_serve.bench.workload import build_workload  # noqa: E402
from gpu_batch_serve.gpu import GpuSampler  # noqa: E402


@dataclass(frozen=True)
class Arm:
    """One configuration of the server under test."""

    name: str
    max_batch_size: int
    max_queue_delay_ms: float
    precision: str = "fp32"
    note: str = ""

    @property
    def letter(self) -> str:
        return self.name.split(":")[0].strip()


ARMS: List[Arm] = [
    Arm("A: no batching", 1, 0.0, "fp32", "one request, one forward pass"),
    Arm("B: batch 8, no wait", 8, 0.0, "fp32", "coalesce only what is already queued"),
    Arm("C: batch 8, 5ms wait", 8, 5.0, "fp32", "grant a partial batch a grace period"),
    Arm("D: batch 32, 5ms wait", 32, 5.0, "fp32", "wider cap, same grace period"),
    Arm("E: batch 32, 5ms, fp16", 32, 5.0, "fp16", "half precision on top of D"),
]

CONCURRENCIES = [1, 8, 32, 64]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def serve(arm: Arm, device: str, model_id: str, log_dir: Path):
    """Start a server process for one arm and tear it down afterwards."""
    port = find_free_port()
    env = dict(os.environ)
    env.update(
        {
            "GBS_MODEL_ID": model_id,
            "GBS_DEVICE": device,
            "GBS_PRECISION": arm.precision if device == "cuda" else "fp32",
            "GBS_MAX_BATCH_SIZE": str(arm.max_batch_size),
            "GBS_MAX_QUEUE_DELAY_MS": str(arm.max_queue_delay_ms),
            "PYTHONPATH": str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{arm.letter}.log"
    log_file = log_path.open("w", encoding="utf-8")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "gpu_batch_serve.api:get_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url, process, log_path
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        log_file.close()


async def fetch_stats(base_url: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{base_url.rstrip('/')}/stats")
        response.raise_for_status()
        return response.json()


async def run_cell(
    base_url: str,
    texts: List[str],
    concurrency: int,
    duration: float,
    warmup: float,
) -> dict:
    """One (arm, concurrency) measurement, with GPU sampled throughout."""
    with GpuSampler() as sampler:
        result = await run_load(
            base_url=base_url,
            texts=texts,
            concurrency=concurrency,
            duration_seconds=duration,
            warmup_seconds=warmup,
        )
    stats = await fetch_stats(base_url)
    record = result.as_dict()
    record["batching"] = stats.get("batching", {})
    record["gpu"] = sampler.summary().as_dict()
    return record


async def main_async(args: argparse.Namespace) -> int:
    texts = build_workload(args.workload_size, seed=args.seed)
    results_dir = ROOT / "bench" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "bench" / "logs"

    arms = ARMS if args.device == "cuda" else [a for a in ARMS if a.precision == "fp32"]
    if args.arms:
        wanted = {a.strip().upper() for a in args.arms.split(",")}
        arms = [a for a in arms if a.letter.upper() in wanted]
    if not arms:
        print("no arms selected", file=sys.stderr)
        return 2

    cells: List[dict] = []
    started_at = time.time()

    for arm in arms:
        print(f"\n=== {arm.name} ({arm.note}) ===", flush=True)
        with serve(arm, args.device, args.model_id, log_dir) as (url, process, log_path):
            ready = await wait_for_ready(url, timeout_seconds=args.startup_timeout)
            if not ready:
                print(
                    f"  server failed to become ready; see {log_path}",
                    file=sys.stderr,
                    flush=True,
                )
                if process.poll() is not None:
                    print(log_path.read_text(encoding="utf-8")[-2000:], file=sys.stderr)
                return 1

            for concurrency in args.concurrencies:
                runs: List[dict] = []
                for repeat in range(args.repeats):
                    record = await run_cell(
                        url, texts, concurrency, args.duration, args.warmup
                    )
                    runs.append(record)
                    mean_batch = record["batching"].get("mean_batch_size", 0)
                    gpu_util = record["gpu"].get("mean_utilization_pct")
                    print(
                        f"  conc={concurrency:<3} run={repeat + 1}/{args.repeats}  "
                        f"rps={record['throughput_rps']:<8.1f} "
                        f"p50={record['latency_ms']['p50']:<7.1f} "
                        f"p95={record['latency_ms']['p95']:<7.1f} "
                        f"batch={mean_batch:<5} "
                        f"gpu={gpu_util}",
                        flush=True,
                    )
                cells.append(
                    {
                        "arm": arm.name,
                        "note": arm.note,
                        "max_batch_size": arm.max_batch_size,
                        "max_queue_delay_ms": arm.max_queue_delay_ms,
                        "precision": arm.precision,
                        "concurrency": concurrency,
                        "runs": runs,
                        "summary": aggregate_runs(runs),
                    }
                )

    payload = {
        "meta": {
            "model_id": args.model_id,
            "device": args.device,
            "workload_size": args.workload_size,
            "seed": args.seed,
            "duration_seconds": args.duration,
            "warmup_seconds": args.warmup,
            "repeats": args.repeats,
            "concurrencies": args.concurrencies,
            "started_at": started_at,
            "elapsed_seconds": round(time.time() - started_at, 1),
            "gpu_name": detect_gpu_name(),
        },
        "cells": cells,
    }

    out_path = results_dir / "latest.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    table = render_markdown_table(cells)
    (results_dir / "latest.md").write_text(table + "\n", encoding="utf-8")

    print("\n" + table)
    print(f"\nwrote {out_path}")
    return 0


def detect_gpu_name() -> Optional[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    name = result.stdout.strip().splitlines()
    return name[0].strip() if name else None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workload-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument(
        "--arms", default="", help="comma-separated arm letters, e.g. A,C,E"
    )
    parser.add_argument(
        "--concurrencies",
        default=",".join(str(c) for c in CONCURRENCIES),
        help="comma-separated concurrency levels",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one repeat and short windows, for checking the harness runs",
    )
    args = parser.parse_args(argv)

    args.concurrencies = [int(c) for c in str(args.concurrencies).split(",") if c]
    if args.quick:
        args.repeats = 1
        args.duration = 6.0
        args.warmup = 2.0
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
