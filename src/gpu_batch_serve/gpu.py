"""Sample GPU utilisation and memory while a benchmark runs.

Throughput alone cannot say whether batching helped because it filled an
idle GPU or because it merely queued work differently. Utilisation can,
which is why every arm carries a utilisation column.

Sampling shells out to ``nvidia-smi`` rather than using NVML bindings so
the project has no dependency beyond the driver that is already present on
any machine with a GPU.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

_QUERY = "utilization.gpu,memory.used"


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _sample_once() -> Optional[tuple[float, float]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    line = result.stdout.strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


@dataclass
class GpuSummary:
    """Aggregated utilisation over a measurement window."""

    samples: int = 0
    mean_utilization_pct: Optional[float] = None
    peak_utilization_pct: Optional[float] = None
    peak_memory_mib: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "samples": self.samples,
            "mean_utilization_pct": _round(self.mean_utilization_pct),
            "peak_utilization_pct": _round(self.peak_utilization_pct),
            "peak_memory_mib": _round(self.peak_memory_mib),
        }


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 1)


class GpuSampler:
    """Poll ``nvidia-smi`` on a background thread.

    Degrades to a no-op when no GPU is present, so the same benchmark
    script runs unchanged on a CPU-only machine and simply reports no
    utilisation data.
    """

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._utilization: List[float] = []
        self._memory: List[float] = []
        self.enabled = nvidia_smi_available()

    def __enter__(self) -> "GpuSampler":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = _sample_once()
            if sample is not None:
                util, mem = sample
                self._utilization.append(util)
                self._memory.append(mem)
            self._stop.wait(self._interval)

    def summary(self) -> GpuSummary:
        if not self._utilization:
            return GpuSummary(samples=0)
        return GpuSummary(
            samples=len(self._utilization),
            mean_utilization_pct=statistics.fmean(self._utilization),
            peak_utilization_pct=max(self._utilization),
            peak_memory_mib=max(self._memory) if self._memory else None,
        )
