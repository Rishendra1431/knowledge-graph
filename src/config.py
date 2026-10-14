"""Central configuration, path handling, environment capture and logging.

Every setting has a code default, so the project runs even if ``config.yaml``
is missing. Values from ``config.yaml`` override the defaults.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
import sysconfig
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Package versions worth recording next to any published number.
_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "networkx",
    "psutil",
    "streamlit",
    "plotly",
    "matplotlib",
    "shap",
)


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DatasetSettings:
    name: str = "CICIDS2017"
    raw_dir: str = "data/raw"
    max_rows_per_file: int | None = 25000
    sample_strategy: str = "head"
    allow_synthetic_fallback: bool = True
    synthetic_rows: int = 60000
    download_urls: list[str] = field(default_factory=list)
    download_timeout_seconds: int = 60


@dataclass
class PreprocessSettings:
    label_column: str = "Label"
    benign_label: str = "BENIGN"
    timestamp_column: str = "Timestamp"
    drop_columns: list[str] = field(default_factory=lambda: ["Flow ID", "Fwd Header Length.1"])
    entity_columns: list[str] = field(
        default_factory=lambda: [
            "Source IP",
            "Destination IP",
            "Source Port",
            "Destination Port",
            "Protocol",
        ]
    )
    target_column: str = "is_attack"
    replace_infinite: bool = True
    drop_constant_features: bool = True
    variance_threshold: float = 0.0
    clip_outliers: bool = True
    clip_quantiles: list[float] = field(default_factory=lambda: [0.001, 0.999])
    max_missing_fraction: float = 0.5
    one_hot_max_cardinality: int = 12
    split_strategy: str = "temporal"
    test_size: float = 0.3
    temporal: dict[str, list[str]] = field(
        default_factory=lambda: {
            "train_days": ["Monday", "Tuesday", "Wednesday", "Thursday"],
            "test_days": ["Friday"],
        }
    )
    group: dict[str, list[str]] = field(
        default_factory=lambda: {"key_columns": ["Source IP", "Destination IP"]}
    )


@dataclass
class GraphSettings:
    structural_scope: str = "full"
    include_event_nodes: bool = True
    max_event_nodes: int = 250000
    derived_confidence_k: float = 10.0
    max_hops_to_intel: int = 4
    max_common_neighbor_pairs: int = 200000
    community_detection: bool = True
    # Persist a degree-ranked subgraph for the dashboard, and optionally the
    # whole graph as GraphML (can be large when event nodes are included).
    persist_subgraph_nodes: int = 2000
    export_graphml: bool = False


@dataclass
class ModelSettings:
    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_leaf: int = 2
    class_weight: str | None = "balanced"
    n_jobs: int = -1
    decision_threshold: float = 0.5


@dataclass
class EvaluationSettings:
    n_runs: int = 3
    bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95
    mcnemar_alpha: float = 0.05
    inference_batch_size: int = 2048
    save_predictions: bool = True
    run_leakage_sensitivity: bool = True


@dataclass
class ResourceSettings:
    sample_interval_seconds: float = 0.1
    use_tracemalloc: bool = True


@dataclass
class ExplainSettings:
    top_k_features: int = 6
    n_alerts: int = 25
    method: str = "auto"
    shap_max_samples: int = 200


@dataclass
class DashboardSettings:
    results_path: str = "results/metrics.json"
    max_graph_nodes: int = 120
    max_alerts_shown: int = 100


@dataclass
class PathSettings:
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    predictions_dir: str = "results/predictions"
    reports_dir: str = "results/reports"
    models_dir: str = "models"
    logs_dir: str = "logs"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"


@dataclass
class Settings:
    seed: int = 42
    dataset: DatasetSettings = field(default_factory=DatasetSettings)
    preprocess: PreprocessSettings = field(default_factory=PreprocessSettings)
    graph: GraphSettings = field(default_factory=GraphSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    resources: ResourceSettings = field(default_factory=ResourceSettings)
    explain: ExplainSettings = field(default_factory=ExplainSettings)
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)
    paths: PathSettings = field(default_factory=PathSettings)

    # -- convenience -------------------------------------------------------
    def path(self, name: str) -> Path:
        """Absolute path for a directory key in ``paths``."""
        return (PROJECT_ROOT / getattr(self.paths, name)).resolve()

    def ensure_directories(self) -> dict[str, Path]:
        dirs = {
            name: self.path(name)
            for name in (
                "results_dir",
                "figures_dir",
                "predictions_dir",
                "reports_dir",
                "models_dir",
                "logs_dir",
                "interim_dir",
                "processed_dir",
            )
        }
        dirs["raw_dir"] = (PROJECT_ROOT / self.dataset.raw_dir).resolve()
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        return dirs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _merge_section(section_cls, raw: dict[str, Any] | None, name: str):
    section = section_cls()
    if not raw:
        return section
    valid = {f for f in section.__dataclass_fields__}
    for key, value in raw.items():
        if key not in valid:
            logging.getLogger("kgids.config").warning(
                "ignoring unknown config key %s.%s", name, key
            )
            continue
        setattr(section, key, value)
    return section


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    """Load settings from ``config.yaml`` (if present) with code defaults."""
    if config_path is None:
        candidate = PROJECT_ROOT / "config.yaml"
        config_path = candidate if candidate.exists() else None

    raw: dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            try:
                import yaml  # imported lazily so PyYAML stays optional-ish
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "PyYAML is required to read config.yaml; install requirements.txt "
                    "or delete config.yaml to use code defaults."
                ) from exc
            with path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}

    settings = Settings(seed=int(raw.get("seed", 42)))
    settings.dataset = _merge_section(DatasetSettings, raw.get("dataset"), "dataset")
    settings.preprocess = _merge_section(PreprocessSettings, raw.get("preprocess"), "preprocess")
    settings.graph = _merge_section(GraphSettings, raw.get("graph"), "graph")
    settings.model = _merge_section(ModelSettings, raw.get("model"), "model")
    settings.evaluation = _merge_section(EvaluationSettings, raw.get("evaluation"), "evaluation")
    settings.resources = _merge_section(ResourceSettings, raw.get("resources"), "resources")
    settings.explain = _merge_section(ExplainSettings, raw.get("explain"), "explain")
    settings.dashboard = _merge_section(DashboardSettings, raw.get("dashboard"), "dashboard")
    settings.paths = _merge_section(PathSettings, raw.get("paths"), "paths")

    if settings.dataset.max_rows_per_file in (0, "0"):
        settings.dataset.max_rows_per_file = None
    settings.ensure_directories()
    return settings


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(settings: Settings | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure console + file logging and return the project logger."""
    logger = logging.getLogger("kgids")
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    try:
        log_dir = settings.path("logs_dir") if settings else (PROJECT_ROOT / "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"pipeline_{stamp}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:  # read-only filesystem: console logging is enough
        pass

    logger._configured = True  # type: ignore[attr-defined]
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"kgids.{name}")


# ---------------------------------------------------------------------------
# Environment capture (recorded next to every measured number)
# ---------------------------------------------------------------------------
def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            module = __import__(name.replace("-", "_"))
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    return versions


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "logical_cores": os.cpu_count(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_brand": "unknown",
        "physical_cores": None,
        "cpu_max_mhz": None,
    }
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["cpu_max_mhz"] = round(psutil.cpu_freq().max, 1) if psutil.cpu_freq() else None
        info["total_ram_mb"] = round(psutil.virtual_memory().total / (1024**2), 1)
        info["available_ram_mb"] = round(psutil.virtual_memory().available / (1024**2), 1)
    except Exception:
        info["total_ram_mb"] = None
        info["available_ram_mb"] = None

    if platform.system() == "Windows":
        try:
            import winreg  # type: ignore

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            info["cpu_brand"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    info["cpu_brand"] = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            import subprocess

            info["cpu_brand"] = (
                subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                .decode()
                .strip()
            )
        except Exception:
            pass
    return info


def capture_environment(settings: Settings | None = None) -> dict[str, Any]:
    """Hardware/software fingerprint. Stored with every result artifact."""
    env: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "pointer_bits": sysconfig.get_config_var("pointer_size") or (64 if sys.maxsize > 2**32 else 32),
        "cpu": _cpu_info(),
        "packages": _package_versions(),
    }
    try:
        import psutil

        env["logical_cores_psutil"] = psutil.cpu_count(logical=True)
    except Exception:
        pass
    if settings is not None:
        env["config_snapshot"] = settings.to_dict()
    return env


def dump_json(payload: Any, path: str | os.PathLike[str], indent: int = 2) -> Path:
    """Write JSON with ``numpy``-safe conversion, atomically."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, default=_json_default, ensure_ascii=False)
    tmp.replace(out)
    return out


def _json_default(obj: Any) -> Any:
    import numpy as np
    import pandas as pd

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")
