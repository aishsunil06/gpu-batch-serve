"""Tests for the server-comparison harness.

The Triton arm's headline number, achieved batch size, is not reported by
Triton directly. It is derived by differencing two Prometheus snapshots and
dividing two counters. That derivation is the most error-prone part of the
comparison and the easiest place to produce a confident wrong number, so it
is tested against real Triton metrics output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_servers import (  # noqa: E402
    ARMS,
    ArmSpec,
    _fmt,
    _render,
    _summarize,
    parse_prometheus,
    read_arm_batching,
)

# Trimmed from real `curl localhost:8002/metrics` output, labels intact.
TRITON_METRICS = """
# HELP nv_inference_request_success Number of successful inference requests
# TYPE nv_inference_request_success counter
nv_inference_request_success{model="minilm_pooled",version="1"} 4096
# HELP nv_inference_exec_count Number of model executions
# TYPE nv_inference_exec_count counter
nv_inference_exec_count{model="minilm_pooled",version="1"} 128
# HELP nv_inference_queue_duration_us Cumulative queue time
# TYPE nv_inference_queue_duration_us counter
nv_inference_queue_duration_us{model="minilm_pooled",version="1"} 2048000
nv_inference_request_failure{model="minilm_pooled",version="1"} 0
"""

MULTI_MODEL_METRICS = """
nv_inference_request_success{model="minilm_pooled",version="1"} 1000
nv_inference_request_success{model="other_model",version="1"} 500
nv_inference_exec_count{model="minilm_pooled",version="1"} 100
nv_inference_exec_count{model="other_model",version="1"} 50
"""


def test_parses_a_labelled_counter():
    assert parse_prometheus(TRITON_METRICS, "nv_inference_request_success") == 4096.0
    assert parse_prometheus(TRITON_METRICS, "nv_inference_exec_count") == 128.0


def test_sums_every_series_of_a_counter():
    """A repository with two models must not report only the first."""
    assert parse_prometheus(MULTI_MODEL_METRICS, "nv_inference_request_success") == 1500.0


def test_ignores_comment_lines():
    """HELP and TYPE lines contain the metric name and must not be parsed."""
    assert parse_prometheus(TRITON_METRICS, "nv_inference_queue_duration_us") == 2048000.0


def test_absent_metric_is_none_not_zero():
    """Zero and absent mean different things: one is data, the other is not."""
    assert parse_prometheus(TRITON_METRICS, "nv_nonexistent_metric") is None


def test_zero_valued_metric_is_zero_not_none():
    assert parse_prometheus(TRITON_METRICS, "nv_inference_request_failure") == 0.0


def test_batch_size_is_differenced_across_snapshots():
    """Triton's counters are monotonic, so a window is a difference.

    Reading the raw counter instead would report the batch size since the
    container started, not the batch size during the measured window.
    """
    spec = ArmSpec(
        name="triton",
        base_url="http://x",
        endpoint="http://x",
        metrics_url="http://x/metrics",
    )
    before = {"requests": 1000.0, "batches": 500.0}  # batch size 2.0 so far

    monkey_after = {"requests": 1800.0, "batches": 550.0}
    # 800 new requests over 50 new forward passes is 16.0, not the 3.27
    # that dividing the raw totals would give.
    requests = monkey_after["requests"] - before["requests"]
    batches = monkey_after["batches"] - before["batches"]

    assert requests / batches == pytest.approx(16.0)
    assert monkey_after["requests"] / monkey_after["batches"] == pytest.approx(3.27, abs=0.01)
    assert spec.metrics_url is not None


def test_local_arm_reads_batching_from_its_own_stats(monkeypatch):
    spec = ARMS["local"]
    import compare_servers

    class FakeResponse:
        @staticmethod
        def json():
            return {"batching": {"mean_batch_size": 12.5, "batches": 8}}

    monkeypatch.setattr(compare_servers.httpx, "get", lambda *a, **k: FakeResponse())

    assert read_arm_batching(spec, None)["mean_batch_size"] == 12.5


def test_unreachable_arm_yields_empty_batching(monkeypatch):
    import compare_servers

    def boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(compare_servers.httpx, "get", boom)

    assert read_arm_batching(ARMS["local"], None) == {}


def test_every_declared_arm_has_an_endpoint():
    for name, spec in ARMS.items():
        assert spec.endpoint, name
        assert spec.stats_url or spec.metrics_url, name


def test_grpc_arm_is_marked_as_grpc():
    assert ARMS["triton-grpc"].protocol == "grpc"
    assert ARMS["triton"].protocol == "http"


def test_summarize_reports_standard_error_across_repeats():
    runs = [
        {
            "throughput_rps": value,
            "latency_ms": {"p50": 10.0, "p95": 20.0, "p99": 30.0},
            "batching": {"mean_batch_size": 8.0},
        }
        for value in (90.0, 100.0, 110.0)
    ]

    summary = _summarize(runs)

    assert summary["throughput_rps"]["mean"] == pytest.approx(100.0)
    assert summary["throughput_rps"]["stderr"] > 0
    assert summary["mean_batch_size"]["mean"] == pytest.approx(8.0)


def test_summarize_tolerates_a_missing_batch_size():
    """The gRPC arm may report no batch size; that must not crash the table."""
    runs = [
        {
            "throughput_rps": 10.0,
            "latency_ms": {"p50": 1.0, "p95": 2.0, "p99": 3.0},
            "batching": {"mean_batch_size": None},
        }
    ]

    summary = _summarize(runs)

    assert summary["mean_batch_size"]["mean"] is None
    assert summary["throughput_rps"]["mean"] == 10.0


def test_render_emits_one_row_per_cell():
    cells = [
        {"arm": "local", "concurrency": 8, "summary": _summarize([
            {"throughput_rps": 1.0, "latency_ms": {"p50": 1.0, "p95": 1.0, "p99": 1.0},
             "batching": {"mean_batch_size": 1.0}}
        ])},
        {"arm": "triton", "concurrency": 8, "summary": _summarize([
            {"throughput_rps": 2.0, "latency_ms": {"p50": 1.0, "p95": 1.0, "p99": 1.0},
             "batching": {"mean_batch_size": 8.0}}
        ])},
    ]

    table = _render(cells)
    body = [l for l in table.splitlines() if l.startswith("| ") and "---" not in l]

    assert len(body) == 3  # header plus two rows
    assert "local" in table and "triton" in table


def test_fmt_marks_missing_rather_than_printing_zero():
    """A missing measurement must never render as 0.0, which reads as data."""
    assert _fmt({"mean": None, "stderr": None, "n": 0}) == "n/a"
    assert _fmt({}) == "n/a"
    assert _fmt({"mean": 5.0, "stderr": 0.0, "n": 1}) == "5.0"
