"""Results-schema, metric and resource-instrumentation tests.

The validation tests are the important ones: they assert that fabricated or
inconsistent numbers are *rejected*, which is what makes the published metrics
trustworthy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn import metrics as sk_metrics

from src.eval.metrics import (
    auto_claim,
    classification_metrics,
    mcnemar_exact,
    paired_bootstrap_difference,
)
from src.eval.resources import ResourceMonitor, ResourceTracker, overhead_table
from src.eval.results import (
    SCHEMA_VERSION,
    blank_results,
    finalize,
    validate_results,
    write_artifact_bundle,
)


def _labels(n: int = 200, seed: int = 2):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < 0.4).astype(int)
    y_pred = y_true.copy()
    flip = rng.choice(n, size=n // 10, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    y_score = np.clip(0.2 + 0.6 * y_pred + rng.normal(0, 0.05, n), 0, 1)
    return y_true, y_pred, y_score


def _measured_payload(settings):
    y_true, y_pred, y_score = _labels()
    payload = blank_results(settings, "unit-test")
    payload["dataset"] = {
        "name": "unit-dataset", "origin": "synthetic", "is_synthetic": True,
        "rows_loaded": int(len(y_true) * 3), "files": ["unit.csv"],
    }
    payload["leakage_audit"] = {"passed_all": True, "checks": [], "n_failed": 0}
    payload["models"]["baseline_rf"] = {
        "metrics_status": "measured",
        "metrics": classification_metrics(y_true, y_pred, y_score, threshold=0.5),
        "predictor": {"name": "baseline_rf", "decision_threshold": 0.5},
    }
    other = y_pred.copy()
    other[:20] = 1 - other[:20]
    payload["models"]["graph_aware_rf"] = {
        "metrics_status": "measured",
        "metrics": classification_metrics(y_true, other, y_score, threshold=0.5),
        "predictor": {"name": "graph_aware_rf", "decision_threshold": 0.5},
    }
    return payload, y_true, y_pred, other


def test_blank_payload_is_explicitly_not_measured(tmp_path):
    from src.config import Settings

    settings = Settings()
    settings.paths.results_dir = str(tmp_path / "results")
    settings.ensure_directories()
    payload = finalize(blank_results(settings, "unit"))

    assert payload["status"] == "failed"
    assert payload["measured"] is False
    assert payload["metrics_published"] is False
    assert payload["validation"]["valid"] is False
    assert "at_least_one_model_measured" in payload["validation"]["failed_checks"]


def test_valid_payload_passes_validation(test_settings):
    payload, y_true, _, _ = _measured_payload(test_settings)
    payload = finalize(payload)
    assert payload["validation"]["valid"], payload["validation"]["failed_checks"]
    assert payload["status"] == "measured"
    assert payload["metrics_published"] is True


def test_tampered_metric_is_rejected(test_settings):
    payload, _, _, _ = _measured_payload(test_settings)
    payload["models"]["baseline_rf"]["metrics"]["f1"] = 0.9999  # fabricated
    report = validate_results(payload)
    assert report["valid"] is False
    assert "metrics_reproducible_from_confusion_matrix" in report["failed_checks"]
    finalized = finalize(payload)
    assert finalized["metrics_published"] is False


def test_inconsistent_confusion_matrix_is_rejected(test_settings):
    payload, _, _, _ = _measured_payload(test_settings)
    payload["models"]["baseline_rf"]["metrics"]["confusion_matrix"] = [[10, 5], [3, 400]]
    report = validate_results(payload)
    assert report["valid"] is False
    assert "metrics_reproducible_from_confusion_matrix" in report["failed_checks"]


def test_confusion_matrix_must_sum_to_n_samples(test_settings):
    payload, _, _, _ = _measured_payload(test_settings)
    payload["models"]["baseline_rf"]["metrics"]["confusion_matrix"] = [[1, 1], [1, 1]]
    payload["models"]["baseline_rf"]["metrics"]["n_samples"] = 999
    report = validate_results(payload)
    assert "model[baseline_rf].confusion_matrix_sums_to_n_samples" in report["failed_checks"]


def test_unmeasured_model_must_have_null_metrics(test_settings):
    payload, _, _, _ = _measured_payload(test_settings)
    payload["models"]["graph_aware_rf"] = {
        "metrics_status": "not_executed",
        "metrics": {"f1": 0.9},  # a metric for a stage that did not run
        "reason": "no graph features",
    }
    report = validate_results(payload)
    assert "model[graph_aware_rf].not_measured_is_explicit" in report["failed_checks"]


def test_failed_stage_downgrades_status_to_partial(test_settings):
    from src.eval.results import record_stage

    payload, _, _, _ = _measured_payload(test_settings)
    record_stage(payload, "explanation_generation", "failed", "boom")
    payload = finalize(payload)
    assert payload["status"] == "partial"
    assert payload["measured"] is True
    assert "explanation_generation" in payload["failed_stages"]


def test_schema_version_is_recorded(test_settings):
    payload, _, _, _ = _measured_payload(test_settings)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["environment"]["python_version"]


def test_metrics_match_sklearn(test_settings):
    y_true, y_pred, _ = _labels()
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["precision"] == pytest.approx(
        sk_metrics.precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    assert metrics["recall"] == pytest.approx(
        sk_metrics.recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    assert metrics["f1"] == pytest.approx(
        sk_metrics.f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    assert metrics["accuracy"] == pytest.approx(sk_metrics.accuracy_score(y_true, y_pred))
    cm = metrics["confusion_matrix"]
    assert sum(sum(row) for row in cm) == metrics["n_samples"]


def test_metrics_report_none_for_undefined_roc_auc():
    y_true = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    metrics = classification_metrics(y_true, y_pred, np.zeros(10))
    assert metrics["roc_auc"] is None
    assert any("undefined" in note for note in metrics["notes"])


def test_bootstrap_and_mcnemar_on_identical_predictions():
    y_true, y_pred, _ = _labels()
    bootstrap = paired_bootstrap_difference(y_true, y_pred, y_pred, n_boot=50, seed=1)
    assert bootstrap["observed_difference_b_minus_a"] == 0.0
    assert bootstrap["ci_lower"] == 0.0 and bootstrap["ci_upper"] == 0.0
    assert bootstrap["p_value_two_sided"] == pytest.approx(1.0)

    mcnemar = mcnemar_exact(y_true, y_pred, y_pred)
    assert mcnemar["n_discordant"] == 0
    assert mcnemar["p_value"] is None
    assert "agree" in mcnemar["interpretation"]


def test_auto_claim_only_asserts_a_difference_when_the_ci_excludes_zero():
    better = {"available": True, "ci_lower": 0.01, "ci_upper": 0.07,
              "observed_difference_b_minus_a": 0.04, "confidence": 0.95}
    claim = auto_claim(better, "baseline_rf", "graph_aware_rf", "F1", is_synthetic=False)
    assert claim["supported"] is True
    assert "outperforms" in claim["claim"]

    inconclusive = {"available": True, "ci_lower": -0.02, "ci_upper": 0.05,
                    "observed_difference_b_minus_a": 0.01, "confidence": 0.95}
    claim = auto_claim(inconclusive, "baseline_rf", "graph_aware_rf", "F1", is_synthetic=False)
    assert claim["supported"] is False
    assert "no statistically detectable difference" in claim["claim"]

    worse = {"available": True, "ci_lower": -0.09, "ci_upper": -0.01,
             "observed_difference_b_minus_a": -0.05, "confidence": 0.95}
    claim = auto_claim(worse, "baseline_rf", "graph_aware_rf", "F1", is_synthetic=False)
    assert "baseline_rf outperforms" in claim["claim"]

    synthetic = auto_claim(better, "baseline_rf", "graph_aware_rf", "F1", is_synthetic=True)
    assert "synthetic" in synthetic["claim"].lower()


def test_resource_monitor_records_measured_values(test_settings):
    monitor = ResourceMonitor("unit_stage", test_settings, n_items=1000)
    with monitor:
        time.sleep(0.25)
        _ = [index * index for index in range(20000)]

    measurement = monitor.measurement.to_dict()
    assert measurement["wall_seconds"] > 0.05
    assert measurement["cpu_seconds"] >= 0.0
    assert measurement["rss_peak_mb"] is not None and measurement["rss_peak_mb"] > 0
    assert measurement["n_cpu_samples"] >= 1, "the sampler thread must collect samples"
    assert measurement["items_per_second"] is not None
    assert measurement["tracemalloc_available"] in (True, False)
    assert any("tracemalloc" in caveat for caveat in measurement["metadata"]["caveats"])


def test_resource_tracker_summary_and_overhead(test_settings):
    tracker = ResourceTracker(test_settings)
    for name in ("baseline_rf.inference", "graph_aware_rf.inference"):
        for run in range(2):
            monitor = ResourceMonitor(name, test_settings, n_items=500)
            with monitor:
                time.sleep(0.05)
            tracker.add(monitor, run)

    summary = tracker.stage_summary()
    assert set(summary) == {"baseline_rf.inference", "graph_aware_rf.inference"}
    assert summary["baseline_rf.inference"]["n_runs"] == 2
    assert len(summary["baseline_rf.inference"]["per_run"]) == 2
    assert summary["baseline_rf.inference"]["wall_seconds"]["min"] is not None

    table = overhead_table(summary["baseline_rf.inference"], summary["graph_aware_rf.inference"])
    assert table["rows"], "the overhead table must contain rows"
    assert all("relative_change_percent" in row for row in table["rows"])


def test_artifact_bundle_writes_readable_files(test_settings):
    payload, _, _, _ = _measured_payload(test_settings)
    payload = finalize(payload)
    written = write_artifact_bundle(payload, test_settings, alerts=None)

    assert "metrics_json" in written
    path = Path(written["metrics_json"])
    assert path.exists()
    from_disk = json.loads(path.read_text(encoding="utf-8"))
    assert from_disk["schema_version"] == SCHEMA_VERSION
    assert validate_results(from_disk)["valid"], validate_results(from_disk)["failed_checks"]
    assert "comparison_csv" in written and Path(written["comparison_csv"]).exists()
    assert "metrics_headline_csv" in written
