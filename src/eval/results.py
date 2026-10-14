"""Result schema, self-consistency validation and artifact writing.

The project's honesty rule - "never report a number that was not measured" - is
implemented here in three ways:

1. **No literal metrics in the code base.** Every number in an artifact comes out
   of a computation over real arrays.
2. **Internal consistency checks.** Precision, recall, F1 and accuracy are
   recomputed from the reported confusion matrix and must agree with the
   reported values. A hand-edited or fabricated metric block fails validation.
3. **Failure is representable.** A stage that did not run is written as
   ``status: failed`` with ``metrics: null`` and listed in ``failed_stages``,
   rather than being silently replaced by a plausible-looking default.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings, capture_environment, dump_json, get_logger

logger = get_logger("eval.results")

SCHEMA_VERSION = "1.0"

REQUIRED_METRIC_FIELDS = (
    "n_samples", "n_positive", "n_negative", "confusion_matrix",
    "accuracy", "precision", "recall", "f1",
)

STATUS_MEASURED = "measured"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_NOT_EXECUTED = "not_executed"


def blank_results(settings: Settings, mode: str, notes: list[str] | None = None) -> dict[str, Any]:
    """An empty, explicitly-not-measured results payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS_NOT_EXECUTED,
        "measured": False,
        "metrics_published": False,
        "mode": mode,
        "environment": capture_environment(settings),
        "config": settings.to_dict(),
        "dataset": None,
        "preprocessing": None,
        "split": None,
        "leakage_audit": None,
        "models": {},
        "comparison": {},
        "leakage_sensitivity": {},
        "resources": None,
        "resource_overhead": None,
        "explanations": None,
        "stages": [],
        "failed_stages": [],
        "notes": notes or [],
        "validation": None,
    }


def record_stage(payload: dict[str, Any], name: str, status: str, detail: str = "") -> None:
    payload.setdefault("stages", []).append(
        {"stage": name, "status": status, "detail": detail,
         "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    )
    if status not in ("success", "skipped"):
        payload.setdefault("failed_stages", []).append(name)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _is_finite(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return True


def _recompute_from_confusion(cm: list[list[int]]) -> dict[str, float]:
    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def validate_results(payload: dict[str, Any], tolerance: float = 1e-6) -> dict[str, Any]:
    """Independent checks on a results payload. Returns a report, never raises."""
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("schema_version_match", payload.get("schema_version") == SCHEMA_VERSION,
          f"schema_version={payload.get('schema_version')!r}, expected {SCHEMA_VERSION!r}")
    status = payload.get("status")
    check("status_is_declared", status in (STATUS_MEASURED, STATUS_PARTIAL, STATUS_FAILED, STATUS_NOT_EXECUTED),
          f"status={status!r}")

    environment = payload.get("environment") or {}
    check("environment_recorded",
          bool(environment.get("cpu")) and bool(environment.get("python_version")),
          f"python={environment.get('python_version')}, cpu_cores={((environment.get('cpu') or {}).get('logical_cores'))}")

    dataset = payload.get("dataset") or {}
    rows_loaded = dataset.get("rows_loaded")
    check("dataset_recorded", bool(dataset) and bool(rows_loaded),
          f"origin={dataset.get('origin')!r}, rows_loaded={rows_loaded!r}")

    models = payload.get("models") or {}
    executed_models = [name for name, block in models.items() if (block or {}).get("metrics_status") == STATUS_MEASURED]
    check("at_least_one_model_measured", bool(executed_models),
          f"measured models={executed_models or 'none'}")

    consistency_failures: list[str] = []
    for name, block in models.items():
        metrics = (block or {}).get("metrics")
        if (block or {}).get("metrics_status") != STATUS_MEASURED:
            check(f"model[{name}].not_measured_is_explicit", metrics is None,
                  f"metrics_status={block.get('metrics_status')!r}, metrics={'null' if metrics is None else 'present'}")
            continue

        missing = [field for field in REQUIRED_METRIC_FIELDS if field not in (metrics or {})]
        check(f"model[{name}].required_fields_present", not missing, f"missing={missing}")
        if missing or metrics is None:
            continue

        check(f"model[{name}].n_samples_positive", int(metrics.get("n_samples") or 0) > 0,
              f"n_samples={metrics.get('n_samples')}")

        cm = metrics.get("confusion_matrix")
        shape_ok = isinstance(cm, list) and len(cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cm)
        check(f"model[{name}].confusion_matrix_shape", shape_ok, f"confusion_matrix={cm}")
        if not shape_ok:
            continue

        total = sum(int(value) for row in cm for value in row)
        check(f"model[{name}].confusion_matrix_sums_to_n_samples", total == int(metrics.get("n_samples") or -1),
              f"sum(cm)={total}, n_samples={metrics.get('n_samples')}")

        recomputed = _recompute_from_confusion(cm)
        for metric_name, expected in recomputed.items():
            reported = metrics.get(metric_name)
            if reported is None:
                consistency_failures.append(f"{name}.{metric_name} is None")
                continue
            if abs(float(reported) - expected) > tolerance:
                consistency_failures.append(
                    f"{name}.{metric_name}: reported {float(reported):.8f} vs recomputed {expected:.8f}"
                )
        for metric_name in ("accuracy", "precision", "recall", "f1"):
            value = metrics.get(metric_name)
            if value is None or not (-tolerance <= float(value) <= 1 + tolerance):
                consistency_failures.append(f"{name}.{metric_name} out of [0,1]: {value}")

    check("metrics_reproducible_from_confusion_matrix", not consistency_failures,
          "; ".join(consistency_failures) if consistency_failures else
          "precision, recall, F1 and accuracy all match the confusion matrix")

    nan_fields: list[str] = []
    for name, block in models.items():
        for key, value in ((block or {}).get("metrics") or {}).items():
            if isinstance(value, float) and not math.isfinite(value):
                nan_fields.append(f"{name}.{key}")
    check("no_nan_or_inf_in_metrics", not nan_fields, f"non-finite: {nan_fields}" if nan_fields else "all finite")

    resources = payload.get("resources")
    if status in (STATUS_MEASURED, STATUS_PARTIAL) and resources:
        stages = (resources.get("stages") or {})
        with_time = [name for name, block in stages.items() if (block.get("wall_seconds") or {}).get("mean")]
        check("resource_stages_have_measurements", bool(with_time),
              f"{len(with_time)} stage(s) with wall-clock measurements: {with_time[:6]}")
        check("resource_per_run_values_present",
              all((block.get("per_run") or []) for block in stages.values()) if stages else False,
              "per-run measurements are retained alongside the summary")
    else:
        check("resource_block_present_or_not_claimed", not resources,
              f"resources={'present' if resources else 'absent'}, status={status}")

    audit = payload.get("leakage_audit")
    check("leakage_audit_recorded", bool(audit) and "passed_all" in (audit or {}),
          f"passed_all={(audit or {}).get('passed_all')!r}")

    comparison = payload.get("comparison") or {}
    if comparison.get("primary"):
        primary = comparison["primary"]
        claim = primary.get("claim") or {}
        check("comparison_claim_is_rule_based", bool(claim.get("rule")),
              "the comparison claim is derived from the bootstrap CI by a recorded rule")

    failed_stages = payload.get("failed_stages") or []
    check("failed_stages_reflected_in_status",
          (not failed_stages and status != STATUS_FAILED) or bool(failed_stages),
          f"failed_stages={failed_stages}")

    errors = [c for c in checks if not c["passed"]]
    return {
        "valid": not errors,
        "n_checks": len(checks),
        "n_failed": len(errors),
        "failed_checks": [c["name"] for c in errors],
        "checks": checks,
        "tolerance": tolerance,
        "note": (
            "Validation is a consistency guard, not a correctness proof. It confirms that reported "
            "metrics are reproducible from the confusion matrix and that unmeasured stages are marked "
            "as such; it cannot confirm that a measurement is representative of other hardware or data."
        ),
    }


def finalize(payload: dict[str, Any]) -> dict[str, Any]:
    """Set the evidence status, then validate and publish only consistent results."""
    models = payload.get("models") or {}
    measured_models = [n for n, b in models.items() if (b or {}).get("metrics_status") == STATUS_MEASURED]
    failed = payload.get("failed_stages") or []

    if not measured_models:
        payload["status"] = STATUS_FAILED
    elif failed:
        payload["status"] = STATUS_PARTIAL
    else:
        payload["status"] = STATUS_MEASURED

    validation = validate_results(payload)
    payload["validation"] = validation
    payload["measured"] = payload["status"] in (STATUS_MEASURED, STATUS_PARTIAL) and bool(measured_models)
    payload["metrics_published"] = payload["measured"] and validation["valid"] and bool(measured_models)
    payload["measured_models"] = measured_models
    payload["unmeasured_models"] = [n for n in models if n not in measured_models]

    if not validation["valid"]:
        payload["notes"] = list(payload.get("notes") or []) + [
            "VALIDATION FAILED: reported metrics are internally inconsistent. Treat every number in this "
            "file as invalid until the failed checks are resolved."
        ]
        payload["metrics_published"] = False
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


# ---------------------------------------------------------------------------
# Writing artifacts
# ---------------------------------------------------------------------------
def write_results(payload: dict[str, Any], path: str | Path) -> Path:
    return dump_json(payload, path)


def write_headline_csv(payload: dict[str, Any], path: str | Path) -> Path | None:
    """Long-format headline metrics: one row per (model, metric)."""
    rows: list[dict[str, Any]] = []
    for name, block in (payload.get("models") or {}).items():
        metrics = (block or {}).get("metrics")
        if not metrics:
            continue
        for key in ("accuracy", "precision", "recall", "f1", "f1_macro", "specificity",
                    "balanced_accuracy", "mcc", "roc_auc", "average_precision",
                    "false_positive_rate", "false_negative_rate", "n_samples"):
            if key in metrics:
                rows.append({"model": name, "metric": key, "value": metrics[key]})
        for stage_name, stage_block in ((payload.get("resources") or {}).get("stages") or {}).items():
            if stage_name.startswith(name) or name in stage_name:
                for stat in ("wall_seconds", "cpu_percent_avg", "cpu_percent_peak", "rss_peak_mb", "items_per_second"):
                    value = (stage_block.get(stat) or {}).get("mean")
                    if value is not None:
                        rows.append({"model": name, "metric": f"resource.{stage_name}.{stat}", "value": value})
    if not rows:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_comparison_csv(payload: dict[str, Any], path: str | Path) -> Path | None:
    """Baseline vs graph-aware table with absolute and relative differences."""
    models = payload.get("models") or {}
    names = list(models.keys())
    if len(names) < 2:
        return None
    comparison = (payload.get("comparison") or {}).get("primary") or {}
    metric_keys = [
        "accuracy", "precision", "recall", "f1", "f1_macro", "specificity", "mcc",
        "roc_auc", "average_precision", "false_positive_rate", "false_negative_rate",
        "true_positives", "false_positives", "false_negatives", "true_negatives", "n_samples",
    ]
    rows: list[dict[str, Any]] = []
    for key in metric_keys:
        left = ((models.get(names[0]) or {}).get("metrics") or {}).get(key)
        right = ((models.get(names[1]) or {}).get("metrics") or {}).get(key)
        row = {
            "metric": key,
            names[0]: left,
            names[1]: right,
            "absolute_difference": (
                None if left is None or right is None else round(float(right) - float(left), 8)
            ),
            "relative_change_percent": (
                None if not left or right is None else round(100.0 * (float(right) - float(left)) / float(left), 6)
            ),
        }
        bootstrap = comparison.get("bootstrap") or {}
        if key == comparison.get("primary_metric"):
            row["bootstrap_ci_lower"] = bootstrap.get("ci_lower")
            row["bootstrap_ci_upper"] = bootstrap.get("ci_upper")
            row["bootstrap_p_value"] = bootstrap.get("p_value_two_sided")
        row["significance"] = comparison.get("mcnemar", {}).get("p_value") if key == "f1" else None
        rows.append(row)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_resources_csv(payload: dict[str, Any], path: str | Path) -> Path | None:
    stages = ((payload.get("resources") or {}).get("stages")) or {}
    rows: list[dict[str, Any]] = []
    for stage, block in stages.items():
        for run in block.get("per_run") or []:
            rows.append(
                {
                    "stage": stage,
                    "run_index": (run.get("metadata") or {}).get("run_index"),
                    "wall_seconds": run.get("wall_seconds"),
                    "cpu_seconds": run.get("cpu_seconds"),
                    "cpu_percent_avg": run.get("cpu_percent_avg"),
                    "cpu_percent_peak": run.get("cpu_percent_peak"),
                    "rss_peak_mb": run.get("rss_peak_mb"),
                    "rss_delta_mb": run.get("rss_delta_mb"),
                    "tracemalloc_peak_mb": run.get("tracemalloc_peak_mb"),
                    "n_items": run.get("n_items"),
                    "items_per_second": run.get("items_per_second"),
                    "rss_peak_source": run.get("rss_peak_source"),
                }
            )
    if not rows:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_alerts_csv(alerts: list[dict[str, Any]], path: str | Path) -> Path | None:
    if not alerts:
        return None
    rows: list[dict[str, Any]] = []
    for alert in alerts:
        entities = {entity.get("role", "entity"): entity.get("value") for entity in alert.get("connected_entities", [])}
        features = alert.get("important_features", [])
        rows.append(
            {
                "alert_id": alert.get("alert_id"),
                "record_id": alert.get("record_id"),
                "threat_score": alert.get("threat_score"),
                "predicted_class": alert.get("predicted_class"),
                "ground_truth_label": alert.get("ground_truth_label"),
                "ground_truth_category": alert.get("ground_truth_category"),
                "source_ip": entities.get("source"),
                "destination_ip": entities.get("destination"),
                "destination_service": entities.get("destination_service"),
                "timestamp": (alert.get("evidence_source") or {}).get("timestamp"),
                "source_file": (alert.get("evidence_source") or {}).get("file"),
                "top_feature_1": features[0]["feature"] if len(features) > 0 else None,
                "top_feature_1_value": features[0]["value"] if len(features) > 0 else None,
                "top_feature_2": features[1]["feature"] if len(features) > 1 else None,
                "top_feature_3": features[2]["feature"] if len(features) > 2 else None,
                "attribution_method": (alert.get("attribution_method") or {}).get("name"),
                "n_graph_paths": len([p for p in alert.get("supporting_graph_paths", []) if p.get("available")]),
                "min_hops_to_intel": min(
                    [p.get("n_hops") for p in alert.get("supporting_graph_paths", []) if p.get("available") and p.get("n_hops", 0) > 0],
                    default=None,
                ),
                "summary": alert.get("analyst_explanation"),
            }
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_feature_importance_csv(payload: dict[str, Any], path: str | Path) -> Path | None:
    rows: list[dict[str, Any]] = []
    for name, block in (payload.get("models") or {}).items():
        for item in (block or {}).get("top_features") or []:
            rows.append({"model": name, "feature": item.get("feature"), "importance": item.get("importance")})
    if not rows:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_artifact_bundle(
    payload: dict[str, Any],
    settings: Settings,
    alerts: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Write every result artifact, keeping raw JSON for reproducibility."""
    results_dir = settings.path("results_dir")
    written: dict[str, str] = {}

    written["metrics_json"] = str(write_results(payload, results_dir / "metrics.json"))
    headline = write_headline_csv(payload, results_dir / "metrics_headline.csv")
    if headline:
        written["metrics_headline_csv"] = str(headline)
    comparison = write_comparison_csv(payload, results_dir / "comparison_table.csv")
    if comparison:
        written["comparison_csv"] = str(comparison)
    resources = write_resources_csv(payload, results_dir / "resources.csv")
    if resources:
        written["resources_csv"] = str(resources)
    importance = write_feature_importance_csv(payload, results_dir / "feature_importance.csv")
    if importance:
        written["feature_importance_csv"] = str(importance)
    if alerts:
        alert_csv = write_alerts_csv(alerts, results_dir / "alerts.csv")
        if alert_csv:
            written["alerts_csv"] = str(alert_csv)
        written["alerts_json"] = str(dump_json(alerts, results_dir / "alerts.json"))
    if payload.get("leakage_audit"):
        written["leakage_audit_json"] = str(dump_json(payload["leakage_audit"], results_dir / "leakage_audit.json"))
    return written
