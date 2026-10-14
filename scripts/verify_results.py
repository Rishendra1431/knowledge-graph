"""Independently verify ``results/metrics.json`` against the saved predictions.

This is the anti-fabrication check you run before quoting a number in a paper.
It does not trust the runner: it re-reads ``results/predictions/test_predictions.npz``,
recomputes precision/recall/F1/accuracy and the confusion matrix from scratch,
and compares them with the published values. It also checks that every alert's
threat score matches the model score for that record.

Exit codes: 0 = all checks passed, 1 = verification failed, 2 = results not measured.

Usage::

    python scripts/verify_results.py
    python scripts/verify_results.py --results results/metrics.json --strict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.results import validate_results  # noqa: E402

TOLERANCE = 1e-6

MODEL_ARRAYS = {
    "baseline_rf": ("baseline_rf_pred", "baseline_rf_score"),
    "graph_aware_rf": ("graph_aware_rf_pred", "graph_aware_rf_score"),
}


def _recompute(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "n_samples": int(len(y_true)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify measured results against saved predictions.")
    parser.add_argument("--results", default=str(PROJECT_ROOT / "results" / "metrics.json"))
    parser.add_argument("--predictions", default=str(PROJECT_ROOT / "results" / "predictions" / "test_predictions.npz"))
    parser.add_argument("--alerts", default=str(PROJECT_ROOT / "results" / "alerts.json"))
    parser.add_argument("--strict", action="store_true",
                        help="also fail when the payload is not fully measured")
    args = parser.parse_args(argv)

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[FAIL] results file not found: {results_path}")
        print("       Run the pipeline first: python run_all.py")
        return 1
    payload = json.loads(results_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    checks: list[str] = []

    def check(name: str, passed: bool, detail: str) -> None:
        (checks if passed else failures).append(f"{'PASS' if passed else 'FAIL'} | {name} | {detail}")

    schema = validate_results(payload)
    check("schema_validation", schema["valid"],
          f"{schema['n_checks'] - schema['n_failed']}/{schema['n_checks']} checks passed"
          + ("" if schema["valid"] else f"; failed: {schema['failed_checks']}"))

    status = payload.get("status")
    check("pipeline_status_declared", status in ("measured", "partial", "failed", "not_executed"), f"status={status}")
    check("metrics_published_flag", bool(payload.get("metrics_published")),
          f"measured={payload.get('measured')}, metrics_published={payload.get('metrics_published')}")

    predictions_path = Path(args.predictions)
    arrays: dict[str, np.ndarray] = {}
    if not predictions_path.exists():
        check("predictions_available", False,
              f"{predictions_path} missing - metrics cannot be independently reproduced")
    else:
        check("predictions_available", True, str(predictions_path))
        with np.load(predictions_path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        y_true = arrays["y_true"].astype(int)
        record_ids = arrays.get("test_record_ids")
        for model, (pred_key, score_key) in MODEL_ARRAYS.items():
                block = (payload.get("models") or {}).get(model) or {}
                if block.get("metrics_status") != "measured" or not block.get("metrics"):
                    check(f"{model}_declared_unmeasured",
                          block.get("metrics") is None,
                          f"metrics_status={block.get('metrics_status')} and metrics are null")
                    continue
                if pred_key not in arrays:
                    check(f"{model}_predictions_present", False, f"{pred_key} missing from the npz archive")
                    continue

                y_pred = arrays[pred_key].astype(int)
                recomputed = _recompute(y_true, y_pred)
                published = block["metrics"]

                check(f"{model}_n_samples", recomputed["n_samples"] == int(published.get("n_samples") or -1),
                      f"published={published.get('n_samples')} recomputed={recomputed['n_samples']}")

                published_cm = published.get("confusion_matrix")
                check(f"{model}_confusion_matrix_reproduced", published_cm == recomputed["confusion_matrix"],
                      f"published={published_cm} recomputed={recomputed['confusion_matrix']}")

                for metric in ("accuracy", "precision", "recall", "f1"):
                    value = published.get(metric)
                    if value is None:
                        check(f"{model}_{metric}", False, "published value is null")
                        continue
                    difference = abs(float(value) - recomputed[metric])
                    check(f"{model}_{metric}", difference <= TOLERANCE,
                          f"published={float(value):.8f} recomputed={recomputed[metric]:.8f} diff={difference:.2e}")

                if score_key in arrays:
                    scores = arrays[score_key]
                    threshold = float(block.get("predictor", {}).get("decision_threshold", 0.5))
                    derived = (scores >= threshold).astype(int)
                    agreement = float(np.mean(derived == y_pred))
                    check(f"{model}_label_matches_threshold_rule", agreement == 1.0,
                          f"predictions reproduce from scores with threshold {threshold} for {agreement:.4%} of records")

        alerts_path = Path(args.alerts)
        if alerts_path.exists() and record_ids is not None and "graph_aware_rf_score" in arrays:
            alerts_payload = json.loads(alerts_path.read_text(encoding="utf-8"))
            lookup = {int(rid): float(score) for rid, score in zip(record_ids, arrays["graph_aware_rf_score"])}
            mismatches = []
            for alert in alerts_payload:
                rid = alert.get("record_id")
                score = alert.get("threat_score")
                if rid is None or score is None or int(rid) not in lookup:
                    mismatches.append(f"alert for record {rid} cannot be matched to a prediction")
                    continue
                if abs(lookup[int(rid)] - float(score)) > TOLERANCE:
                    mismatches.append(
                        f"record {rid}: alert score {float(score):.8f} != model score {lookup[int(rid)]:.8f}"
                    )
            check("alert_scores_match_model_output", not mismatches,
                  "; ".join(mismatches[:5]) if mismatches else f"{len(alerts_payload)} alert score(s) verified")
            # Every explanation must carry its evidence source and limitations.
            incomplete = [
                alert.get("alert_id") for alert in alerts_payload
                if not alert.get("evidence_source") or not alert.get("limitations") or not alert.get("analyst_explanation")
            ]
            check("explanations_carry_evidence_and_limitations", not incomplete,
                  f"alerts missing fields: {incomplete[:5]}" if incomplete else
                  f"{len(alerts_payload)} explanations include evidence source, limitations and analyst text")
        else:
            check("alert_verification_skipped", True,
                  "no alerts.json or prediction archive available; alert cross-check not applicable")

    resource_block = payload.get("resources")
    if resource_block:
        stages = resource_block.get("stages") or {}
        missing_runs = [name for name, block in stages.items() if not block.get("per_run")]
        check("resource_per_run_retained", not missing_runs,
              f"stages without per-run data: {missing_runs}" if missing_runs else
              f"{len(stages)} stage(s) keep per-run measurements")

    print("=" * 92)
    print(f"verification of {results_path}")
    print("=" * 92)
    for line in checks + failures:
        print(line)
    print("-" * 92)
    print(f"{len(checks)} passed, {len(failures)} failed")

    if payload.get("status") != "measured":
        print()
        print(f"NOTE: pipeline status is '{payload.get('status')}'. "
              "Stages that did not complete contain no metrics; do not report their values.")
    if payload.get("notes"):
        print()
        for note in payload["notes"]:
            print(f"NOTE: {note}")

    if failures:
        return 1
    if args.strict and not payload.get("metrics_published"):
        print("\n[FAIL] --strict requested but the payload is not fully measured/published.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
