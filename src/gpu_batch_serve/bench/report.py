"""Turn raw benchmark records into a readable table.

Kept separate from the load generator so that a saved results file can be
re-rendered without re-running an experiment, and so the rendering logic
is testable without a GPU.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Sequence

_HEADERS = [
    "Arm",
    "Batch",
    "Delay ms",
    "Conc.",
    "RPS",
    "p50 ms",
    "p95 ms",
    "p99 ms",
    "Mean batch",
    "GPU %",
]


def aggregate_runs(runs: Sequence[dict]) -> dict:
    """Collapse repeated runs of one cell into mean and standard error.

    Reported as standard error rather than standard deviation because the
    question a reader has is whether two arms differ, and the error on the
    mean is what answers it.
    """
    if not runs:
        raise ValueError("no runs to aggregate")

    def _values(path: Sequence[str]) -> List[float]:
        out: List[float] = []
        for run in runs:
            node = run
            for key in path:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            if isinstance(node, (int, float)):
                out.append(float(node))
        return out

    summary: Dict[str, dict] = {}
    for label, path in {
        "throughput_rps": ("throughput_rps",),
        "p50_ms": ("latency_ms", "p50"),
        "p95_ms": ("latency_ms", "p95"),
        "p99_ms": ("latency_ms", "p99"),
        "mean_batch_size": ("batching", "mean_batch_size"),
        "gpu_mean_utilization_pct": ("gpu", "mean_utilization_pct"),
    }.items():
        values = _values(path)
        if not values:
            summary[label] = {"mean": None, "stderr": None, "n": 0}
            continue
        mean = statistics.fmean(values)
        stderr = (
            statistics.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0.0
        )
        summary[label] = {
            "mean": round(mean, 3),
            "stderr": round(stderr, 3),
            "n": len(values),
        }
    return summary


def render_markdown_table(cells: Sequence[dict]) -> str:
    """Render aggregated cells as a GitHub-flavoured markdown table."""
    lines = [
        "| " + " | ".join(_HEADERS) + " |",
        "|" + "|".join(["---"] * len(_HEADERS)) + "|",
    ]
    for cell in cells:
        summary = cell["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(cell.get("arm", "")),
                    str(cell.get("max_batch_size", "")),
                    str(cell.get("max_queue_delay_ms", "")),
                    str(cell.get("concurrency", "")),
                    _fmt(summary.get("throughput_rps")),
                    _fmt(summary.get("p50_ms")),
                    _fmt(summary.get("p95_ms")),
                    _fmt(summary.get("p99_ms")),
                    _fmt(summary.get("mean_batch_size")),
                    _fmt(summary.get("gpu_mean_utilization_pct")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _fmt(entry: dict | None) -> str:
    if not entry or entry.get("mean") is None:
        return "n/a"
    mean = entry["mean"]
    stderr = entry.get("stderr") or 0.0
    if entry.get("n", 0) > 1 and stderr > 0:
        return f"{mean:.1f} (SE {stderr:.1f})"
    return f"{mean:.1f}"
