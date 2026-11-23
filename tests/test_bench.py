"""Tests for the measurement code itself.

A benchmark whose statistics are wrong is worse than no benchmark, because
it produces numbers that look defensible. These tests pin the percentile
definition, the aggregation, and the workload's determinism.
"""

from __future__ import annotations

import pytest

from gpu_batch_serve.bench.load import LoadResult, percentile
from gpu_batch_serve.bench.report import aggregate_runs, render_markdown_table
from gpu_batch_serve.bench.workload import build_workload


def test_percentile_is_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    assert percentile(values, 0.5) == 5
    assert percentile(values, 0.95) == 10
    assert percentile(values, 1.0) == 10


def test_percentile_returns_an_observed_value():
    """Nearest-rank must never invent a value between two samples."""
    values = [10.0, 20.0, 30.0]

    for fraction in (0.1, 0.5, 0.9, 0.99):
        assert percentile(values, fraction) in values


def test_percentile_of_empty_is_zero():
    assert percentile([], 0.95) == 0.0


def test_percentile_rejects_out_of_range_fraction():
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)


def test_throughput_counts_only_completed_requests():
    result = LoadResult(
        concurrency=4,
        completed=100,
        failed=7,
        wall_seconds=10.0,
        latencies_ms=[1.0] * 100,
    )

    assert result.throughput_rps == pytest.approx(10.0)
    assert result.as_dict()["failed"] == 7


def test_aggregate_reports_zero_error_for_a_single_run():
    runs = [
        {
            "throughput_rps": 100.0,
            "latency_ms": {"p50": 10.0, "p95": 20.0, "p99": 30.0},
            "batching": {"mean_batch_size": 4.0},
            "gpu": {"mean_utilization_pct": 50.0},
        }
    ]

    summary = aggregate_runs(runs)

    assert summary["throughput_rps"]["mean"] == 100.0
    assert summary["throughput_rps"]["stderr"] == 0.0
    assert summary["throughput_rps"]["n"] == 1


def test_aggregate_computes_standard_error_across_runs():
    runs = [
        {
            "throughput_rps": value,
            "latency_ms": {"p50": 1.0, "p95": 1.0, "p99": 1.0},
            "batching": {"mean_batch_size": 1.0},
            "gpu": {"mean_utilization_pct": 1.0},
        }
        for value in (90.0, 100.0, 110.0)
    ]

    summary = aggregate_runs(runs)

    assert summary["throughput_rps"]["mean"] == pytest.approx(100.0)
    assert summary["throughput_rps"]["stderr"] > 0
    assert summary["throughput_rps"]["n"] == 3


def test_aggregate_tolerates_a_missing_gpu_column():
    """A CPU run has no utilisation data and must still aggregate."""
    runs = [
        {
            "throughput_rps": 10.0,
            "latency_ms": {"p50": 1.0, "p95": 1.0, "p99": 1.0},
            "batching": {"mean_batch_size": 1.0},
            "gpu": {"mean_utilization_pct": None},
        }
    ]

    summary = aggregate_runs(runs)

    assert summary["gpu_mean_utilization_pct"]["mean"] is None


def test_render_marks_missing_values_rather_than_dropping_columns():
    cells = [
        {
            "arm": "A",
            "max_batch_size": 1,
            "max_queue_delay_ms": 0.0,
            "concurrency": 8,
            "summary": aggregate_runs(
                [
                    {
                        "throughput_rps": 10.0,
                        "latency_ms": {"p50": 1.0, "p95": 2.0, "p99": 3.0},
                        "batching": {"mean_batch_size": 1.0},
                        "gpu": {"mean_utilization_pct": None},
                    }
                ]
            ),
        }
    ]

    table = render_markdown_table(cells)

    assert "n/a" in table
    assert table.count("|") > 10


def test_workload_is_deterministic_for_a_seed():
    assert build_workload(32, seed=17) == build_workload(32, seed=17)


def test_workload_differs_across_seeds():
    assert build_workload(32, seed=17) != build_workload(32, seed=18)


def test_workload_has_mixed_lengths():
    """Uniform-length input would flatter batching unrealistically."""
    lengths = {len(text) for text in build_workload(64)}

    assert len(lengths) > 3
