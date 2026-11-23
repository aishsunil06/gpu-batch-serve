#!/usr/bin/env python
"""Compare three implementations of one batching policy.

The arms serve byte-identical ONNX weights, with batch cap and queue delay
set to the same values, over the same pre-tokenized fixture. Anything that
differs between them is therefore attributable to the serving
implementation rather than to the model, the runtime, or the work.

    local        the ~60-line collector in gpu_batch_serve.batcher
    ray          Ray Serve's @serve.batch
    triton       NVIDIA Triton's dynamic_batching, over HTTP
    triton-grpc  the same Triton, over gRPC

The gRPC arm is included because Triton is the only arm offering a second
transport, and separating transport from batching is the only way to know
whether a Triton win is its batcher or its protocol.

Servers are NOT started by this script. Start them first, since each has a
different lifecycle (uvicorn, a Ray cluster, a Docker container), and a
script that owned all three would obscure which one failed. See README.

Usage:
    python scripts/compare_servers.py --arms local,triton
    python scripts/compare_servers.py --concurrencies 1,8,32 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

DEFAULT_FIXTURE = ROOT / "bench" / "fixtures" / "workload_tokens.json"


@dataclass(frozen=True)
class ArmSpec:
    """How to reach one arm and read its own view of batching."""

    name: str
    base_url: str
    endpoint: str
    protocol: str = "http"
    stats_url: Optional[str] = None
    metrics_url: Optional[str] = None
    health_url: Optional[str] = None


ARMS: Dict[str, ArmSpec] = {
    "local": ArmSpec(
        name="local",
        base_url="http://127.0.0.1:8100",
        endpoint="http://127.0.0.1:8100/embed-tokens",
        stats_url="http://127.0.0.1:8100/stats",
        health_url="http://127.0.0.1:8100/healthz",
    ),
    "ray": ArmSpec(
        name="ray",
        base_url="http://127.0.0.1:8200",
        endpoint="http://127.0.0.1:8200/embed-tokens",
        stats_url="http://127.0.0.1:8200/stats",
        health_url="http://127.0.0.1:8200/healthz",
    ),
    "triton": ArmSpec(
        name="triton",
        base_url="http://127.0.0.1:8000",
        endpoint="http://127.0.0.1:8000/v2/models/minilm_pooled/infer",
        metrics_url="http://127.0.0.1:8002/metrics",
        health_url="http://127.0.0.1:8000/v2/health/ready",
    ),
    "triton-grpc": ArmSpec(
        name="triton-grpc",
        base_url="127.0.0.1:8001",
        endpoint="127.0.0.1:8001",
        protocol="grpc",
        metrics_url="http://127.0.0.1:8002/metrics",
        health_url="http://127.0.0.1:8000/v2/health/ready",
    ),
}


def check_health(spec: ArmSpec, timeout: float = 5.0) -> bool:
    if not spec.health_url:
        return True
    try:
        response = httpx.get(spec.health_url, timeout=timeout)
    except Exception:  # noqa: BLE001 - absence is the answer
        return False
    if response.status_code != 200:
        return False
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - Triton returns an empty 200
        return True
    if isinstance(body, dict) and "ready" in body:
        return bool(body["ready"])
    return True


def parse_prometheus(text: str, metric: str) -> Optional[float]:
    """Sum every labelled series of one Triton counter.

    Triton labels its counters by model and version, so a bare name match
    would miss them and a first-match would undercount a repository holding
    more than one model.
    """
    total = None
    pattern = re.compile(rf"^{re.escape(metric)}\{{[^}}]*\}}\s+([0-9.eE+-]+)$")
    bare = re.compile(rf"^{re.escape(metric)}\s+([0-9.eE+-]+)$")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        match = pattern.match(line) or bare.match(line)
        if match:
            total = (total or 0.0) + float(match.group(1))
    return total


def read_triton_batching(metrics_url: str) -> Dict[str, Optional[float]]:
    """Derive achieved batch size from Triton's own counters.

    nv_inference_request_success counts requests and nv_inference_exec_count
    counts forward passes, so their ratio is the batch size Triton actually
    formed. This is deliberately the same quantity the local arm reports as
    mean_batch_size, which is what makes the two comparable on the metric
    that distinguishes "batching did not help" from "batching did not
    happen".
    """
    try:
        text = httpx.get(metrics_url, timeout=10.0).text
    except Exception:  # noqa: BLE001
        return {"mean_batch_size": None, "requests": None, "batches": None}

    requests = parse_prometheus(text, "nv_inference_request_success")
    execs = parse_prometheus(text, "nv_inference_exec_count")
    mean = None
    if requests is not None and execs:
        mean = requests / execs
    return {
        "mean_batch_size": round(mean, 3) if mean is not None else None,
        "requests": requests,
        "batches": execs,
        "queue_us": parse_prometheus(text, "nv_inference_queue_duration_us"),
        "compute_us": parse_prometheus(text, "nv_inference_compute_infer_duration_us"),
    }


def run_go_loadgen(
    spec: ArmSpec,
    fixture: Path,
    concurrency: int,
    duration: float,
    warmup: float,
) -> dict:
    """Drive an HTTP arm with the Go generator."""
    if shutil.which("go") is None:
        raise RuntimeError("the go toolchain is not on PATH")

    arm_flag = "triton" if spec.name.startswith("triton") else spec.name
    result = subprocess.run(
        [
            "go", "run", ".",
            "-url", spec.endpoint,
            "-arm", arm_flag,
            "-fixture", str(fixture),
            "-concurrency", str(concurrency),
            "-duration", str(duration),
            "-warmup", str(warmup),
            "-json",
        ],
        cwd=str(ROOT / "loadgen"),
        capture_output=True,
        text=True,
        timeout=duration + warmup + 300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"go loadgen failed: {result.stderr[-800:]}")
    return json.loads(result.stdout)


def run_grpc_loadgen(
    spec: ArmSpec,
    fixture: Path,
    concurrency: int,
    duration: float,
    warmup: float,
) -> dict:
    """Drive Triton over gRPC with its official Python client.

    Hand-rolling Triton's protobuf service in Go would be a large amount of
    generated code for one table row, so the gRPC arm uses the official
    client. That makes the gRPC row's client-side overhead different from
    the Go rows', which is stated in the findings rather than hidden: the
    comparison that row supports is Triton-gRPC against Triton-HTTP driven
    by the same Python client, not against the Go rows.
    """
    try:
        import tritonclient.grpc as grpcclient
    except ImportError as exc:
        raise RuntimeError(
            "pip install tritonclient[grpc] to measure the gRPC arm"
        ) from exc

    import threading

    import numpy as np

    payload = json.loads(fixture.read_text(encoding="utf-8"))
    records = payload["requests"]
    seq_len = payload["meta"]["sequence_length"]

    def one_client(offset: int, deadline: float, out: List[float], fails: List[int]):
        client = grpcclient.InferenceServerClient(url=spec.endpoint, verbose=False)
        index = offset
        while time.perf_counter() < deadline:
            record = records[index % len(records)]
            index += concurrency
            inputs = []
            for name in ("input_ids", "attention_mask", "token_type_ids"):
                arr = np.asarray([record[name]], dtype=np.int64)
                inp = grpcclient.InferInput(name, [1, seq_len], "INT64")
                inp.set_data_from_numpy(arr)
                inputs.append(inp)
            started = time.perf_counter()
            try:
                client.infer(model_name="minilm_pooled", inputs=inputs)
            except Exception:  # noqa: BLE001 - counted, not raised
                fails.append(1)
                continue
            out.append((time.perf_counter() - started) * 1000.0)
        client.close()

    def window(seconds: float) -> tuple[List[float], List[int], float]:
        latencies: List[float] = []
        fails: List[int] = []
        started = time.perf_counter()
        deadline = started + seconds
        threads = [
            threading.Thread(
                target=one_client, args=(i, deadline, latencies, fails), daemon=True
            )
            for i in range(concurrency)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return latencies, fails, time.perf_counter() - started

    if warmup > 0:
        window(warmup)
    latencies, fails, wall = window(duration)
    latencies.sort()

    def pct(fraction: float) -> float:
        if not latencies:
            return 0.0
        rank = max(1, int(round(fraction * len(latencies))))
        return latencies[min(rank, len(latencies)) - 1]

    return {
        "arm": spec.name,
        "concurrency": concurrency,
        "completed": len(latencies),
        "failed": len(fails),
        "wall_seconds": wall,
        "throughput_rps": len(latencies) / wall if wall > 0 else 0.0,
        "latency_ms": {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)},
        "generator": "python-grpc",
    }


def reset_arm_stats(spec: ArmSpec) -> Optional[dict]:
    """Zero an arm's counters, or snapshot Triton's before a window.

    Triton's counters are monotonic and cannot be reset, so its per-window
    figures are computed by differencing a before and after snapshot.
    """
    if spec.stats_url:
        try:
            httpx.post(f"{spec.base_url}/reset-stats", timeout=10.0)
        except Exception:  # noqa: BLE001
            pass
        return None
    if spec.metrics_url:
        return read_triton_batching(spec.metrics_url)
    return None


def read_arm_batching(spec: ArmSpec, before: Optional[dict]) -> dict:
    if spec.stats_url:
        try:
            body = httpx.get(spec.stats_url, timeout=10.0).json()
            return body.get("batching", {})
        except Exception:  # noqa: BLE001
            return {}
    if spec.metrics_url:
        after = read_triton_batching(spec.metrics_url)
        if not before:
            return after
        requests = (after.get("requests") or 0) - (before.get("requests") or 0)
        batches = (after.get("batches") or 0) - (before.get("batches") or 0)
        mean = requests / batches if batches else None
        return {
            "requests": requests,
            "batches": batches,
            "mean_batch_size": round(mean, 3) if mean is not None else None,
        }
    return {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="local,ray,triton,triton-grpc")
    parser.add_argument("--concurrencies", default="1,8,32,64")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "servers.json")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    if args.quick:
        args.repeats = 1
        args.duration = 6.0
        args.warmup = 2.0

    if not args.fixture.exists():
        print(
            f"fixture {args.fixture} not found; run scripts/prepare_fixture.py first",
            file=sys.stderr,
        )
        return 2

    concurrencies = [int(c) for c in args.concurrencies.split(",") if c]
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in wanted if a not in ARMS]
    if unknown:
        print(f"unknown arms: {unknown}; known: {sorted(ARMS)}", file=sys.stderr)
        return 2

    reachable, skipped = [], []
    for name in wanted:
        (reachable if check_health(ARMS[name]) else skipped).append(name)

    if skipped:
        print(f"skipping unreachable arms: {', '.join(skipped)}")
        print("  start them per the README, then re-run")
    if not reachable:
        print("no arms reachable", file=sys.stderr)
        return 1

    print(f"comparing: {', '.join(reachable)}")
    cells: List[dict] = []

    for name in reachable:
        spec = ARMS[name]
        print(f"\n=== {name} ===", flush=True)
        for concurrency in concurrencies:
            runs: List[dict] = []
            for repeat in range(args.repeats):
                before = reset_arm_stats(spec)
                try:
                    if spec.protocol == "grpc":
                        record = run_grpc_loadgen(
                            spec, args.fixture, concurrency, args.duration, args.warmup
                        )
                    else:
                        record = run_go_loadgen(
                            spec, args.fixture, concurrency, args.duration, args.warmup
                        )
                except RuntimeError as exc:
                    print(f"  conc={concurrency}: {exc}", file=sys.stderr)
                    break
                record["batching"] = read_arm_batching(spec, before)
                runs.append(record)
                print(
                    f"  conc={concurrency:<3} run={repeat + 1}/{args.repeats}  "
                    f"rps={record['throughput_rps']:<8.1f} "
                    f"p50={record['latency_ms']['p50']:<8.1f} "
                    f"p95={record['latency_ms']['p95']:<8.1f} "
                    f"batch={record['batching'].get('mean_batch_size')} "
                    f"failed={record.get('failed', 0)}",
                    flush=True,
                )
            if runs:
                cells.append(
                    {
                        "arm": name,
                        "concurrency": concurrency,
                        "runs": runs,
                        "summary": _summarize(runs),
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "meta": {
                    "arms": reachable,
                    "skipped": skipped,
                    "concurrencies": concurrencies,
                    "repeats": args.repeats,
                    "duration_seconds": args.duration,
                    "warmup_seconds": args.warmup,
                    "fixture": str(args.fixture),
                },
                "cells": cells,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    table = _render(cells)
    (args.out.parent / "servers.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nwrote {args.out}")
    return 0


def _summarize(runs: List[dict]) -> dict:
    def agg(getter) -> dict:
        values = []
        for run in runs:
            value = getter(run)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            return {"mean": None, "stderr": None, "n": 0}
        return {
            "mean": round(statistics.fmean(values), 3),
            "stderr": round(
                statistics.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0.0,
                3,
            ),
            "n": len(values),
        }

    return {
        "throughput_rps": agg(lambda r: r.get("throughput_rps")),
        "p50_ms": agg(lambda r: r.get("latency_ms", {}).get("p50")),
        "p95_ms": agg(lambda r: r.get("latency_ms", {}).get("p95")),
        "p99_ms": agg(lambda r: r.get("latency_ms", {}).get("p99")),
        "mean_batch_size": agg(lambda r: r.get("batching", {}).get("mean_batch_size")),
    }


def _render(cells: List[dict]) -> str:
    headers = ["Arm", "Conc.", "RPS", "p50 ms", "p95 ms", "p99 ms", "Mean batch"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for cell in cells:
        s = cell["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    cell["arm"],
                    str(cell["concurrency"]),
                    _fmt(s["throughput_rps"]),
                    _fmt(s["p50_ms"]),
                    _fmt(s["p95_ms"]),
                    _fmt(s["p99_ms"]),
                    _fmt(s["mean_batch_size"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _fmt(entry: dict) -> str:
    if not entry or entry.get("mean") is None:
        return "n/a"
    if entry.get("n", 0) > 1 and entry.get("stderr"):
        return f"{entry['mean']:.1f} (SE {entry['stderr']:.1f})"
    return f"{entry['mean']:.1f}"


if __name__ == "__main__":
    sys.exit(main())
