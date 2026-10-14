"""pytest configuration: make the project root importable and set test-friendly defaults."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from src.config import Settings  # noqa: E402


@pytest.fixture()
def test_settings(tmp_path) -> Settings:
    """Small, fast settings that keep every test off the real dataset."""
    settings = Settings()
    settings.seed = 7
    settings.dataset.allow_synthetic_fallback = True
    settings.dataset.synthetic_rows = 1200
    settings.dataset.max_rows_per_file = None
    settings.dataset.raw_dir = str(tmp_path / "raw")
    settings.model.n_estimators = 12
    settings.model.n_jobs = 1
    settings.evaluation.n_runs = 1
    settings.evaluation.bootstrap_samples = 60
    settings.evaluation.save_predictions = False
    settings.evaluation.run_leakage_sensitivity = False
    settings.resources.sample_interval_seconds = 0.02
    settings.explain.n_alerts = 3
    settings.explain.method = "auto"
    settings.graph.persist_subgraph_nodes = 200
    settings.paths.results_dir = str(tmp_path / "results")
    settings.paths.figures_dir = str(tmp_path / "results" / "figures")
    settings.paths.predictions_dir = str(tmp_path / "results" / "predictions")
    settings.paths.reports_dir = str(tmp_path / "results" / "reports")
    settings.paths.models_dir = str(tmp_path / "models")
    settings.paths.logs_dir = str(tmp_path / "logs")
    settings.paths.interim_dir = str(tmp_path / "data" / "interim")
    settings.paths.processed_dir = str(tmp_path / "data" / "processed")
    settings.ensure_directories()
    return settings


@pytest.fixture(scope="session")
def quick_dataset():
    """A small synthetic frame reused across tests (generation is deterministic)."""
    from src.config import Settings as S
    from src.data.acquire import generate_synthetic_dataset

    settings = S()
    settings.seed = 11
    settings.dataset.synthetic_rows = 1500
    frame, provenance = generate_synthetic_dataset(settings)
    return frame, provenance, settings
