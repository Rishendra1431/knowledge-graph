"""Resource measurement: wall-clock time, CPU, RAM and throughput.

Instrumentation notes that belong in any write-up using these numbers:

* ``time.perf_counter`` gives wall-clock durations, so values include operating
  system scheduling, disk I/O and (unavoidably) other load on the machine.
* Process CPU percent comes from ``psutil`` and is expressed relative to **one
  core**; a value above 100% means more than one core was busy. It is reported
  alongside ``cpu_seconds`` (from ``cpu_times`` deltas), which is the more
  reliable quantity when comparing runs.
* Peak process RAM is taken from the OS-reported peak working set when the
  platform exposes it, otherwise from periodic ``memory_info().rss`` sampling.
  Sampling can miss a short-lived allocation spike, so treat it as a lower bound.
* ``tracemalloc`` tracks **Python-level allocations only**. It does not see
  memory allocated inside C extensions (numpy buffers, pandas blocks, BLAS
  workspaces) and it only counts allocations made after it was started. It is
  reported as ``tracemalloc_peak_mb`` and explicitly must not be read as total
  process RAM.
* Measurements are taken on the machine described in the ``environment`` block
  stored next to them. They are not transferable to other hardware.
"""
from __future__ import annotations

import gc
import os
import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.config import Settings, get_logger

logger = get_logger("eval.resources")

CAVEATS = [
    "Wall-clock times include OS scheduling, I/O and other load on the same machine.",
    "Process CPU percent is relative to one core and can exceed 100% when several cores are used.",
    "Peak RAM comes from OS peak working set when available, else from periodic RSS sampling (a lower bound).",
    "tracemalloc measures Python-level allocations only and misses buffers allocated in C extensions; "
    "it is NOT total process RAM.",
    "Values are specific to the recorded hardware/Python/package versions and are not portable.",
]


@dataclass
class Measurement:
    name: str
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    cpu_percent_avg: float | None = None
    cpu_percent_peak: float | None = None
    system_cpu_percent_avg: float | None = None
    rss_before_mb: float | None = None
    rss_after_mb: float | None = None
    rss_peak_mb: float | None = None
    rss_delta_mb: float | None = None
    rss_peak_source: str = "unavailable"
    tracemalloc_peak_mb: float | None = None
    tracemalloc_current_mb: float | None = None
    tracemalloc_available: bool = False
    n_cpu_samples: int = 0
    sample_interval_seconds: float = 0.1
    n_items: int | None = None
    items_per_second: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wall_seconds": round(self.wall_seconds, 6),
            "cpu_seconds": round(self.cpu_seconds, 6),
            "cpu_percent_avg": None if self.cpu_percent_avg is None else round(self.cpu_percent_avg, 4),
            "cpu_percent_peak": None if self.cpu_percent_peak is None else round(self.cpu_percent_peak, 4),
            "cpu_definition": "percent of a single logical core; >100% indicates multiple cores busy",
            "system_cpu_percent_avg": None if self.system_cpu_percent_avg is None else round(self.system_cpu_percent_avg, 4),
            "rss_before_mb": None if self.rss_before_mb is None else round(self.rss_before_mb, 3),
            "rss_after_mb": None if self.rss_after_mb is None else round(self.rss_after_mb, 3),
            "rss_peak_mb": None if self.rss_peak_mb is None else round(self.rss_peak_mb, 3),
            "rss_delta_mb": None if self.rss_delta_mb is None else round(self.rss_delta_mb, 3),
            "rss_peak_source": self.rss_peak_source,
            "tracemalloc_peak_mb": None if self.tracemalloc_peak_mb is None else round(self.tracemalloc_peak_mb, 3),
            "tracemalloc_current_mb": None if self.tracemalloc_current_mb is None else round(self.tracemalloc_current_mb, 3),
            "tracemalloc_available": self.tracemalloc_available,
            "n_cpu_samples": self.n_cpu_samples,
            "sample_interval_seconds": self.sample_interval_seconds,
            "n_items": self.n_items,
            "items_per_second": None if self.items_per_second is None else round(self.items_per_second, 4),
            "metadata": self.metadata,
        }


class ResourceMonitor:
    """Context manager that records time/CPU/RAM for one pipeline stage.

    Usage::

        monitor = ResourceMonitor("model_training", settings)
        with monitor:
            model.fit(X, y)
        print(monitor.measurement.to_dict())

    ``n_items`` marks throughput (e.g. rows processed); wall time is divided by it.
    """

    def __init__(self, name: str, settings: Settings | None = None, n_items: int | None = None) -> None:
        self.name = name
        self.settings = settings
        self.n_items = n_items
        interval = 0.1 if settings is None else float(settings.resources.sample_interval_seconds)
        self.interval = max(interval, 0.01)
        self.use_tracemalloc = True if settings is None else bool(settings.resources.use_tracemalloc)
        self.measurement = Measurement(name=name, sample_interval_seconds=self.interval)
        self._psutil_process = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cpu_samples: list[float] = []
        self._system_samples: list[float] = []
        self._rss_samples: list[float] = []
        self._started_here = False

    # -- setup -------------------------------------------------------------
    def _process(self):
        if self._psutil_process is None:
            try:
                import psutil

                self._psutil_process = psutil.Process(os.getpid())
            except Exception:
                self._psutil_process = False
        return self._psutil_process or None

    def _prime(self) -> None:
        process = self._process()
        if process is None:
            return
        try:
            process.cpu_percent(interval=None)
            import psutil

            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def _cpu_seconds(self) -> float:
        process = self._process()
        if process is None:
            return 0.0
        try:
            times = process.cpu_times()
            return float(times.user + times.system)
        except Exception:
            return 0.0

    def _rss_mb(self) -> float | None:
        process = self._process()
        if process is None:
            return None
        try:
            return float(process.memory_info().rss) / (1024**2)
        except Exception:
            return None

    def _os_peak_mb(self) -> tuple[float | None, str]:
        process = self._process()
        if process is None:
            return None, "unavailable"
        try:
            info = process.memory_info()
            peak = getattr(info, "peak_wset", None)  # Windows
            if peak:
                return float(peak) / (1024**2), "psutil memory_info().peak_wset (OS peak working set)"
        except Exception:
            pass
        return None, "unavailable"

    # -- sampling ----------------------------------------------------------
    def _sample_loop(self) -> None:
        process = self._process()
        try:
            import psutil
        except Exception:
            return
        while not self._stop.is_set():
            try:
                if process is not None:
                    cpu = process.cpu_percent(interval=None)
                    if cpu is not None:
                        self._cpu_samples.append(float(cpu))
                self._system_samples.append(float(psutil.cpu_percent(interval=None)))
                rss = self._rss_mb()
                if rss is not None:
                    self._rss_samples.append(rss)
            except Exception:
                pass
            self._stop.wait(self.interval)

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "ResourceMonitor":
        gc.collect()
        self._prime()
        self.measurement.rss_before_mb = self._rss_mb()
        self._cpu_start = self._cpu_seconds()
        if self.use_tracemalloc:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._started_here = True
            tracemalloc.reset_peak()
            self.measurement.tracemalloc_available = True
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, name=f"resmon-{self.name}", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._t1 = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 4))
        self._cpu_end = self._cpu_seconds()

        measurement = self.measurement
        measurement.wall_seconds = self._t1 - self._t0
        measurement.cpu_seconds = max(self._cpu_end - self._cpu_start, 0.0)
        measurement.n_cpu_samples = len(self._cpu_samples)
        if self._cpu_samples:
            measurement.cpu_percent_avg = float(np.mean(self._cpu_samples))
            measurement.cpu_percent_peak = float(np.max(self._cpu_samples))
        if measurement.wall_seconds > 0 and measurement.cpu_seconds > 0:
            derived = 100.0 * measurement.cpu_seconds / measurement.wall_seconds
            measurement.metadata["cpu_percent_from_cpu_times"] = round(derived, 4)
        if self._system_samples:
            measurement.system_cpu_percent_avg = float(np.mean(self._system_samples))

        measurement.rss_after_mb = self._rss_mb()
        if measurement.rss_before_mb is not None and measurement.rss_after_mb is not None:
            measurement.rss_delta_mb = measurement.rss_after_mb - measurement.rss_before_mb
        os_peak, source = self._os_peak_mb()
        sampled_peak = max(self._rss_samples) if self._rss_samples else None
        candidates = [v for v in (os_peak, sampled_peak) if v is not None]
        if candidates:
            measurement.rss_peak_mb = max(candidates)
            measurement.rss_peak_source = (
                source if os_peak is not None and os_peak >= (sampled_peak or 0)
                else "sampled psutil memory_info().rss (lower bound)"
            )

        if self.use_tracemalloc and tracemalloc.is_tracing():
            try:
                current, peak = tracemalloc.get_traced_memory()
                measurement.tracemalloc_current_mb = float(current) / (1024**2)
                measurement.tracemalloc_peak_mb = float(peak) / (1024**2)
            finally:
                if self._started_here:
                    tracemalloc.stop()

        measurement.n_items = self.n_items
        if self.n_items and measurement.wall_seconds > 0:
            measurement.items_per_second = float(self.n_items) / measurement.wall_seconds
        measurement.metadata.setdefault("caveats", CAVEATS)
        measurement.metadata.setdefault("exceptions_raised", None if exc_type is None else exc_type.__name__)

        logger.info(
            "stage %-28s wall=%.3fs cpu_avg=%.1f%% cpu_peak=%.1f%% rss_peak=%s MB tracemalloc_peak=%s MB",
            self.name,
            measurement.wall_seconds,
            measurement.cpu_percent_avg or 0.0,
            measurement.cpu_percent_peak or 0.0,
            f"{measurement.rss_peak_mb:.1f}" if measurement.rss_peak_mb is not None else "n/a",
            f"{measurement.tracemalloc_peak_mb:.1f}" if measurement.tracemalloc_peak_mb is not None else "n/a",
        )
        return False  # never swallow exceptions


@contextmanager
def measure(name: str, settings: Settings | None = None, n_items: int | None = None):
    """Yield a :class:`ResourceMonitor`; its measurement is available after the block."""
    monitor = ResourceMonitor(name, settings, n_items=n_items)
    with monitor:
        yield monitor


def mean_std(values: list[float | None]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    array = np.asarray(clean, dtype="float64")
    return {
        "mean": round(float(array.mean()), 6),
        "std": round(float(array.std(ddof=1)), 6) if len(array) > 1 else 0.0,
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        "n": int(len(array)),
    }


class ResourceTracker:
    """Collects stage measurements across repeated runs and summarises them."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stages: dict[str, list[Measurement]] = {}
        self.throughput: dict[str, list[float]] = {}

    def add(self, monitor: ResourceMonitor, run_index: int | None = None) -> None:
        measurement = monitor.measurement
        measurement.metadata.setdefault("run_index", run_index)
        self.stages.setdefault(measurement.name, []).append(measurement)
        if measurement.items_per_second is not None:
            self.throughput.setdefault(measurement.name, []).append(measurement.items_per_second)

    def stage_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for name, measurements in self.stages.items():
            summary[name] = {
                "n_runs": len(measurements),
                "wall_seconds": mean_std([m.wall_seconds for m in measurements]),
                "cpu_seconds": mean_std([m.cpu_seconds for m in measurements]),
                "cpu_percent_avg": mean_std([m.cpu_percent_avg for m in measurements]),
                "cpu_percent_peak": mean_std([m.cpu_percent_peak for m in measurements]),
                "rss_peak_mb": mean_std([m.rss_peak_mb for m in measurements]),
                "rss_delta_mb": mean_std([m.rss_delta_mb for m in measurements]),
                "tracemalloc_peak_mb": mean_std([m.tracemalloc_peak_mb for m in measurements]),
                "items_per_second": mean_std([m.items_per_second for m in measurements]),
                "n_items": measurements[0].n_items,
                "rss_peak_source": measurements[0].rss_peak_source,
                "per_run": [m.to_dict() for m in measurements],
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured": True,
            "n_runs": self.settings.evaluation.n_runs,
            "sample_interval_seconds": self.settings.resources.sample_interval_seconds,
            "tracemalloc_enabled": self.settings.resources.use_tracemalloc,
            "stages": self.stage_summary(),
            "caveats": CAVEATS,
            "one_time_vs_per_event": (
                "Training/graph-construction stages run once; inference and explanation stages are "
                "per-event costs and are reported per event and per second."
            ),
        }


def overhead_table(
    baseline: dict[str, Any],
    graph_aware: dict[str, Any],
    baseline_name: str = "baseline_rf",
    graph_name: str = "graph_aware_rf",
) -> dict[str, Any]:
    """Resource overhead of the graph-aware detector relative to the baseline."""
    rows: list[dict[str, Any]] = []
    metrics = (
        ("wall_seconds", "runtime (s)", "mean"),
        ("cpu_seconds", "cpu time (s)", "mean"),
        ("cpu_percent_avg", "cpu percent (avg)", "mean"),
        ("cpu_percent_peak", "cpu percent (peak)", "mean"),
        ("rss_peak_mb", "peak RAM (MB)", "mean"),
        ("items_per_second", "throughput (events/s)", "mean"),
    )
    for key, label, statistic in metrics:
        base = ((baseline.get(key) or {}) if isinstance(baseline.get(key), dict) else {}).get(statistic)
        graph = ((graph_aware.get(key) or {}) if isinstance(graph_aware.get(key), dict) else {}).get(statistic)
        if base is None and isinstance(baseline.get(key), (int, float)):
            base = float(baseline[key])
        if graph is None and isinstance(graph_aware.get(key), (int, float)):
            graph = float(graph_aware[key])
        row = {
            "metric": label,
            "key": key,
            f"{baseline_name}": base,
            f"{graph_name}": graph,
            "absolute_difference": None if base is None or graph is None else round(graph - base, 6),
            "relative_change_percent": (
                None if not base or graph is None else round(100.0 * (graph - base) / base, 4)
            ),
        }
        rows.append(row)
    return {
        "baseline": baseline_name,
        "graph_aware": graph_name,
        "rows": rows,
        "note": (
            "Relative change is (graph_aware - baseline) / baseline. Inference throughput below the "
            "baseline means the graph features cost extra time per event."
        ),
    }
