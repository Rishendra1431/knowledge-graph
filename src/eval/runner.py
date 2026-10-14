"""Reproducible end-to-end evaluation.

Every stage is wrapped in :class:`ResourceMonitor`, every measurement is added to
a :class:`ResourceTracker`, and the final payload is validated before it is
written. If a stage raises, the payload is finalised as ``failed``/``partial``,
written to disk and the exception is re-raised: an artifact stating "not
executed" is produced instead of a plausible-looking number.

Usage::

    python -m src.eval.runner --config config.yaml
"""
from __future__ import annotations

import argparse
import copy
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings, get_logger, load_settings, setup_logging
from src.data.acquire import acquire_dataset
from src.data.preprocess import PreprocessResult, run_preprocessing
from src.eval.metrics import (
    auto_claim,
    classification_metrics,
    mcnemar_exact,
    paired_bootstrap_difference,
)
from src.eval.resources import ResourceMonitor, ResourceTracker, mean_std, overhead_table
from src.eval.results import (
    STATUS_MEASURED,
    STATUS_NOT_EXECUTED,
    blank_results,
    finalize,
    record_stage,
    write_artifact_bundle,
    write_results,
)
from src.explain.explainer import AlertExplainer
from src.graph.builder import build_intel_index, build_knowledge_graph
from src.graph.features import GraphFeatureExtractor
from src.models.detectors import ThreatDetector, build_baseline, build_graph_aware

logger = get_logger("eval.runner")

PRIMARY_METRIC = "f1"


def _stage(tracker: ResourceTracker, monitor: ResourceMonitor, run_index: int | None = None) -> None:
    tracker.add(monitor, run_index)


def _metric_summary(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "accuracy", "precision", "recall", "f1", "f1_macro", "specificity",
        "balanced_accuracy", "mcc", "roc_auc", "average_precision",
        "false_positive_rate", "false_negative_rate",
    )
    summary: dict[str, Any] = {}
    for key in keys:
        values = [m.get(key) for m in metrics_list]
        summary[key] = mean_std([v for v in values if v is not None])
    return summary


def _detector_block(
    detector: ThreatDetector,
    metrics_per_run: list[dict[str, Any]],
    prediction_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "metrics_status": STATUS_MEASURED,
        "metrics": metrics_per_run[0],
        "metrics_per_run": metrics_per_run,
        "metrics_summary_across_runs": _metric_summary(metrics_per_run),
        "predictor": detector.metadata(),
        "top_features": detector.top_features(20),
        "inference": prediction_summaries[0],
        "inference_per_run": prediction_summaries,
    }


def _evaluate_detector(
    detector: ThreatDetector,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    settings: Settings,
    tracker: ResourceTracker,
    run_index: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, Any]:
    with ResourceMonitor(f"{detector.name}.training", settings, n_items=len(X_train)) as monitor:
        detector.fit(X_train, y_train)
    _stage(tracker, monitor, run_index)

    with ResourceMonitor(f"{detector.name}.inference", settings, n_items=len(X_test)) as monitor:
        prediction = detector.predict_proba(X_test)
    _stage(tracker, monitor, run_index)

    metrics = classification_metrics(y_test, prediction.y_pred, prediction.y_score, threshold=detector.threshold)
    return metrics, prediction.y_pred, prediction.y_score, prediction


def run_pipeline(settings: Settings, mode: str = "full", write: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute the complete evaluation. Returns the payload and the alert list."""
    payload = blank_results(settings, mode)
    tracker = ResourceTracker(settings)
    alerts: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}

    logger.info("=" * 78)
    logger.info("starting evaluation | mode=%s | seed=%d | runs=%d", mode, settings.seed, settings.evaluation.n_runs)
    logger.info("=" * 78)

    try:
        # ---------------------------------------------------------------
        # 1. Data acquisition
        # ---------------------------------------------------------------
        with ResourceMonitor("01_data_acquisition", settings) as monitor:
            raw_frame, provenance = acquire_dataset(settings)
        _stage(tracker, monitor, 0)
        payload["dataset"] = provenance.to_dict()
        record_stage(payload, "data_acquisition", "success",
                     f"{len(raw_frame)} rows, origin={provenance.origin}")
        if provenance.is_synthetic:
            payload["notes"].append(
                "DATASET IS SYNTHETIC: all metrics below characterise the synthetic generator and must "
                "not be reported as findings about real network traffic."
            )

        # ---------------------------------------------------------------
        # 2. Preprocessing + leakage audit
        # ---------------------------------------------------------------
        with ResourceMonitor("02_preprocessing", settings, n_items=len(raw_frame)) as monitor:
            pre: PreprocessResult = run_preprocessing(raw_frame, settings, persist=True)
        _stage(tracker, monitor, 0)
        payload["preprocessing"] = pre.summary()
        payload["split"] = pre.split_report.to_dict()
        payload["leakage_audit"] = pre.leakage_audit
        record_stage(payload, "preprocessing", "success",
                     f"train={len(pre.train)} test={len(pre.test)} features={len(pre.feature_columns)}")
        if not pre.leakage_audit["passed_all"]:
            record_stage(payload, "leakage_audit", "failed",
                         f"failed checks: {pre.leakage_audit['failed_checks']}")

        train_frame = pre.train.reset_index(drop=True)
        test_frame = pre.test.reset_index(drop=True)
        y_train = train_frame[pre.target_column].to_numpy(dtype=int)
        y_test = test_frame[pre.target_column].to_numpy(dtype=int)
        X_train_tabular = train_frame[pre.feature_columns]
        X_test_tabular = test_frame[pre.feature_columns]

        # ---------------------------------------------------------------
        # 3. Knowledge graph + graph features
        # ---------------------------------------------------------------
        combined = pd.concat([train_frame, test_frame], ignore_index=True)
        with ResourceMonitor("03_threat_intel_index", settings) as monitor:
            intel_index = build_intel_index(train_frame, settings)
        _stage(tracker, monitor, 0)
        payload["threat_intel_index"] = {
            key: value for key, value in intel_index.items() if key != "ips"
        }

        scope = settings.graph.structural_scope
        scope_events = combined if scope == "full" else train_frame
        with ResourceMonitor("04_graph_construction", settings, n_items=len(scope_events)) as monitor:
            knowledge_graph = build_knowledge_graph(
                combined, settings, name=f"{settings.dataset.name}-kg",
                include_labels=False, intel_index=intel_index,
                dataset_origin=provenance.origin,
            )
            extractor = GraphFeatureExtractor(settings, structural_scope=scope)
            extractor.fit(scope_events, intel_index)
        _stage(tracker, monitor, 0)
        payload["knowledge_graph"] = knowledge_graph.summary()
        payload["graph_features"] = extractor.report()
        payload["graph_scope"] = scope
        record_stage(payload, "graph_construction", "success",
                     f"{knowledge_graph.graph.number_of_nodes()} nodes, "
                     f"{knowledge_graph.graph.number_of_edges()} edges, scope={scope}")

        # Persist the graph for inspection and for the dashboard.
        results_dir = settings.path("results_dir")
        subgraph_path = results_dir / "knowledge_graph_subgraph.json"
        try:
            from src.config import dump_json as _dump_json

            _dump_json(
                knowledge_graph.to_node_link(max_nodes=int(settings.graph.persist_subgraph_nodes)),
                subgraph_path,
            )
            artifacts["knowledge_graph_subgraph_json"] = str(subgraph_path)
            if settings.graph.export_graphml:
                graphml_path = results_dir / "knowledge_graph.graphml"
                knowledge_graph.to_graphml(str(graphml_path))
                artifacts["knowledge_graph_graphml"] = str(graphml_path)
        except Exception as exc:
            logger.warning("could not persist the knowledge graph: %s", exc)

        with ResourceMonitor("05_graph_feature_transform", settings, n_items=len(combined)) as monitor:
            graph_train = extractor.transform(train_frame)
            unseen_train = getattr(extractor, "_last_unseen_fraction", None)
            graph_test = extractor.transform(test_frame)
            unseen_test = getattr(extractor, "_last_unseen_fraction", None)
        _stage(tracker, monitor, 0)
        graph_feature_columns = list(graph_train.columns)
        if not graph_feature_columns:
            record_stage(payload, "graph_features", "skipped",
                         "no graph features available for this dataset; only the baseline is evaluated")
        payload["graph_feature_unseen_entity_fraction"] = {
            "train": unseen_train,
            "test": unseen_test,
            "note": (
                "Share of events whose endpoints were absent from the structural graph scope. "
                "Non-zero values only occur in the train_only (inductive) scope."
            ),
        }

        X_train_graph = pd.concat([X_train_tabular, graph_train], axis=1)
        X_test_graph = pd.concat([X_test_tabular, graph_test], axis=1)

        # ---------------------------------------------------------------
        # 4. Repeated runs: baseline and graph-aware on identical splits
        # ---------------------------------------------------------------
        n_runs = max(int(settings.evaluation.n_runs), 1)
        run_seeds = [settings.seed + index for index in range(n_runs)]
        baseline_metrics_runs: list[dict[str, Any]] = []
        graph_metrics_runs: list[dict[str, Any]] = []
        baseline_predictions: list[dict[str, Any]] = []
        graph_predictions: list[dict[str, Any]] = []
        primary_detectors: dict[str, ThreatDetector] = {}
        primary_predictions: dict[str, Any] = {}

        for run_index, seed in enumerate(run_seeds):
            run_settings = copy.deepcopy(settings)
            run_settings.seed = seed
            logger.info("-" * 78)
            logger.info("run %d/%d (seed=%d)", run_index + 1, n_runs, seed)

            baseline = build_baseline(run_settings, pre.feature_columns)
            metrics_b, pred_b, score_b, summary_b = _evaluate_detector(
                baseline, X_train_tabular, y_train, X_test_tabular, y_test, run_settings, tracker, run_index
            )
            baseline_metrics_runs.append(metrics_b)
            baseline_predictions.append({"run_index": run_index, "seed": seed, "y_pred": pred_b, "y_score": score_b})
            baseline_predictions[-1]["summary"] = summary_b.to_dict()

            graph_aware = None
            if graph_feature_columns:
                graph_aware = build_graph_aware(run_settings, pre.feature_columns, graph_feature_columns)
                metrics_g, pred_g, score_g, summary_g = _evaluate_detector(
                    graph_aware, X_train_graph, y_train, X_test_graph, y_test, run_settings, tracker, run_index
                )
                graph_metrics_runs.append(metrics_g)
                graph_predictions.append({"run_index": run_index, "seed": seed, "y_pred": pred_g, "y_score": score_g})
                graph_predictions[-1]["summary"] = summary_g.to_dict()

            if run_index == 0:
                primary_detectors["baseline_rf"] = baseline
                primary_predictions["baseline_rf"] = {"y_pred": pred_b, "y_score": score_b}
                if graph_aware is not None:
                    primary_detectors["graph_aware_rf"] = graph_aware
                    primary_predictions["graph_aware_rf"] = {"y_pred": pred_g, "y_score": score_g}

            logger.info(
                "run %d metrics | baseline f1=%s recall=%s precision=%s | graph f1=%s",
                run_index,
                _fmt_metric(metrics_b.get("f1")),
                _fmt_metric(metrics_b.get("recall")),
                _fmt_metric(metrics_b.get("precision")),
                _fmt_metric(graph_metrics_runs[-1].get("f1")) if graph_metrics_runs else "n/a",
            )

        payload["models"]["baseline_rf"] = _detector_block(
            primary_detectors["baseline_rf"], baseline_metrics_runs,
            [entry["summary"] for entry in baseline_predictions],
        )
        record_stage(payload, "baseline_training_and_evaluation", "success",
                     f"{n_runs} run(s), seeds={run_seeds}")

        if "graph_aware_rf" in primary_detectors:
            payload["models"]["graph_aware_rf"] = _detector_block(
                primary_detectors["graph_aware_rf"], graph_metrics_runs,
                [entry["summary"] for entry in graph_predictions],
            )
            record_stage(payload, "graph_aware_training_and_evaluation", "success",
                         f"{n_runs} run(s), {len(graph_feature_columns)} graph features")
        else:
            payload["models"]["graph_aware_rf"] = {
                "metrics_status": STATUS_NOT_EXECUTED,
                "metrics": None,
                "reason": "no graph features were available for this dataset",
            }
            record_stage(payload, "graph_aware_training_and_evaluation", "skipped",
                         "no graph features available")

        # ---------------------------------------------------------------
        # 5. Comparison (paired bootstrap + McNemar)
        # ---------------------------------------------------------------
        if graph_feature_columns:
            pred_b = primary_predictions["baseline_rf"]["y_pred"]
            pred_g = primary_predictions["graph_aware_rf"]["y_pred"]
            score_g = primary_predictions["graph_aware_rf"]["y_score"]

            bootstrap = paired_bootstrap_difference(
                y_test, pred_b, pred_g,
                score_a=primary_predictions["baseline_rf"]["y_score"], score_b=score_g,
                metric=PRIMARY_METRIC,
                n_boot=int(settings.evaluation.bootstrap_samples),
                confidence=float(settings.evaluation.bootstrap_confidence),
                seed=settings.seed,
            )
            secondary = {
                metric: paired_bootstrap_difference(
                    y_test, pred_b, pred_g, metric=metric,
                    n_boot=max(int(settings.evaluation.bootstrap_samples // 2), 200),
                    confidence=float(settings.evaluation.bootstrap_confidence),
                    seed=settings.seed,
                )
                for metric in ("precision", "recall", "accuracy")
            }
            mcnemar = mcnemar_exact(y_test, pred_b, pred_g)
            deltas = [
                (gm.get(PRIMARY_METRIC) or 0.0) - (bm.get(PRIMARY_METRIC) or 0.0)
                for bm, gm in zip(baseline_metrics_runs, graph_metrics_runs)
            ]
            claim = auto_claim(bootstrap, "baseline_rf", "graph_aware_rf", "F1", provenance.is_synthetic)
            payload["comparison"] = {
                "primary": {
                    "primary_metric": PRIMARY_METRIC,
                    "bootstrap": bootstrap,
                    "secondary_bootstrap": secondary,
                    "mcnemar": mcnemar,
                    "delta_across_runs": [round(float(d), 8) for d in deltas],
                    "consistent_across_runs": bool(all(d > 0 for d in deltas)) or bool(all(d < 0 for d in deltas)) or bool(all(d == 0 for d in deltas)),
                    "claim": claim,
                    "same_split": True,
                    "note": (
                        "Both arms use the same train/test split, the same algorithm, the same "
                        "hyperparameters and the same seed within each run. Only the feature set differs."
                    ),
                },
                "assignment_of_difference": (
                    "A positive difference cannot be attributed to the graph features alone if the two "
                    "feature sets differ in dimensionality; with the same algorithm and split, feature "
                    "count is the only other difference. Report both metrics and the overhead."
                ),
            }
            record_stage(payload, "model_comparison", "success", claim["claim"])

            # -----------------------------------------------------------
            # 6. Leakage sensitivity: strict inductive structural scope
            # -----------------------------------------------------------
            if settings.evaluation.run_leakage_sensitivity and scope == "full":
                try:
                    inductive = GraphFeatureExtractor(settings, structural_scope="train_only")
                    with ResourceMonitor("06a_leakage_sensitivity_graph_fit", settings, n_items=len(train_frame)) as monitor:
                        inductive.fit(train_frame, intel_index)
                        inductive_test = inductive.transform(test_frame)
                        inductive_train = inductive.transform(train_frame)
                    _stage(tracker, monitor, 0)
                    if list(inductive_test.columns) == graph_feature_columns:
                        run_settings = copy.deepcopy(settings)
                        run_settings.seed = settings.seed
                        detector = build_graph_aware(run_settings, pre.feature_columns, graph_feature_columns)
                        X_train_ind = pd.concat([X_train_tabular, inductive_train], axis=1)
                        X_test_ind = pd.concat([X_test_tabular, inductive_test], axis=1)
                        metrics_ind, pred_ind, score_ind, _ = _evaluate_detector(
                            detector, X_train_ind, y_train, X_test_ind, y_test, run_settings, tracker, 0
                        )
                        bootstrap_ind = paired_bootstrap_difference(
                            y_test, pred_b, pred_ind, metric=PRIMARY_METRIC,
                            n_boot=max(int(settings.evaluation.bootstrap_samples // 2), 200),
                            confidence=float(settings.evaluation.bootstrap_confidence),
                            seed=settings.seed,
                        )
                        transductive_value = payload["models"]["graph_aware_rf"]["metrics"].get(PRIMARY_METRIC)
                        inductive_value = metrics_ind.get(PRIMARY_METRIC)
                        payload["leakage_sensitivity"] = {
                            "status": "measured",
                            "configuration": "train_only (inductive): structural graph built from training events only",
                            "metrics": metrics_ind,
                            "transductive_metric": transductive_value,
                            "delta_transductive_minus_inductive_f1": (
                                None if transductive_value is None or inductive_value is None
                                else round(float(transductive_value) - float(inductive_value), 8)
                            ),
                            "bootstrap_vs_baseline": bootstrap_ind,
                            "unseen_entity_fraction_test": getattr(inductive, "_last_unseen_fraction", None),
                            "interpretation": (
                                "If the inductive result drops sharply, part of the transductive result came "
                                "from test-time graph context that would not exist in a live deployment."
                            ),
                        }
                        record_stage(payload, "leakage_sensitivity", "success",
                                     f"inductive {PRIMARY_METRIC}={metrics_ind.get(PRIMARY_METRIC)}")
                    else:
                        record_stage(payload, "leakage_sensitivity", "failed",
                                     "inductive feature columns differ from the transductive ones")
                except Exception as exc:  # sensitivity analysis must not break the run
                    logger.exception("leakage sensitivity experiment failed")
                    payload["leakage_sensitivity"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                    record_stage(payload, "leakage_sensitivity", "failed", str(exc))

        # ---------------------------------------------------------------
        # 7. Alert explanations
        # ---------------------------------------------------------------
        if "graph_aware_rf" in primary_detectors:
            explainer = AlertExplainer(
                detector=primary_detectors["graph_aware_rf"],
                settings=settings,
                graph_extractor=extractor,
                knowledge_graph=knowledge_graph,
                intel_index=intel_index,
                dataset_name=provenance.name,
                dataset_origin=provenance.origin,
                feature_reference=train_frame[X_train_graph.columns],
            )
            n_explain = min(int(settings.explain.n_alerts), int((primary_predictions["graph_aware_rf"]["y_pred"] == 1).sum()))
            with ResourceMonitor("07_explanation_generation", settings, n_items=max(n_explain, 1)) as monitor:
                alerts = explainer.explain_top_alerts(
                    X=X_test_graph,
                    y_score=primary_predictions["graph_aware_rf"]["y_score"],
                    y_pred=primary_predictions["graph_aware_rf"]["y_pred"],
                    rows=test_frame,
                    y_true=y_test,
                    n_alerts=n_explain,
                )
            _stage(tracker, monitor, 0)
            payload["explanations"] = {
                "status": "measured",
                "attribution_method": {"name": explainer.method, "note": explainer.method_note},
                "n_alerts": len(alerts),
                "n_events_scored": int(len(y_test)),
                "alert_selection_rule": (
                    "highest threat scores among records the detector labelled as attack; "
                    "no ground-truth label influences which alerts are explained"
                ),
                "includes_graph_paths": any(
                    any(path.get("available") and path.get("n_hops", 0) > 0 for path in alert.get("supporting_graph_paths", []))
                    for alert in alerts
                ),
                "mean_supporting_paths_per_alert": (
                    round(float(np.mean([len([p for p in alert.get("supporting_graph_paths", []) if p.get("available")]) for alert in alerts])), 4)
                    if alerts else None
                ),
                "mean_relationships_per_alert": (
                    round(float(np.mean([len(alert.get("relationships", [])) for alert in alerts])), 4) if alerts else None
                ),
            }
            record_stage(payload, "explanation_generation", "success", f"{len(alerts)} alerts explained")
        else:
            payload["explanations"] = {"status": STATUS_NOT_EXECUTED, "n_alerts": 0}
            record_stage(payload, "explanation_generation", "skipped", "graph-aware detector unavailable")

        # ---------------------------------------------------------------
        # 8. Resources
        # ---------------------------------------------------------------
        payload["resources"] = tracker.to_dict()
        stages = payload["resources"]["stages"]
        baseline_inference = stages.get("baseline_rf.inference")
        graph_inference = stages.get("graph_aware_rf.inference")
        if baseline_inference and graph_inference:
            payload["resource_overhead"] = {
                "inference": overhead_table(baseline_inference, graph_inference),
                "dataset_rows_evaluated": int(len(y_test)),
                "dataset_rows_trained": int(len(y_train)),
                "batch_size": int(settings.evaluation.inference_batch_size),
                "n_repeated_runs": n_runs,
                "measurement_notes": payload["resources"]["caveats"],
            }
        record_stage(payload, "resource_measurement", "success",
                     f"{len(stages)} instrumented stages, {n_runs} run(s)")

        # ---------------------------------------------------------------
        # 9. Persist predictions for independent verification
        # ---------------------------------------------------------------
        if settings.evaluation.save_predictions:
            predictions_path = settings.path("predictions_dir") / "test_predictions.npz"
            payload_arrays: dict[str, Any] = {
                "y_true": y_test,
                "test_record_ids": test_frame["__record_id"].to_numpy(dtype=np.int64),
                "run_seeds": np.asarray(run_seeds, dtype=np.int64),
                "baseline_rf_pred": primary_predictions["baseline_rf"]["y_pred"],
                "baseline_rf_score": primary_predictions["baseline_rf"]["y_score"],
            }
            if "graph_aware_rf" in primary_predictions:
                payload_arrays["graph_aware_rf_pred"] = primary_predictions["graph_aware_rf"]["y_pred"]
                payload_arrays["graph_aware_rf_score"] = primary_predictions["graph_aware_rf"]["y_score"]
            np.savez_compressed(predictions_path, **payload_arrays)
            artifacts["predictions_npz"] = str(predictions_path)
            logger.info("saved test predictions to %s", predictions_path)

        payload["artifacts"] = artifacts
        record_stage(payload, "artifact_persistence", "success", f"{len(artifacts)} file(s)")

    except Exception as exc:
        logger.exception("evaluation failed: %s", exc)
        record_stage(payload, "pipeline", "failed", f"{type(exc).__name__}: {exc}")
        payload["notes"].append(
            f"The pipeline stopped with {type(exc).__name__}: {exc}. Stages that did not complete are "
            "marked as failed and contain no metrics."
        )
        payload = finalize(payload)
        if write:
            try:
                payload["artifacts"] = _write_all(payload, settings, alerts)
                write_results(payload, settings.path("results_dir") / "metrics.json")
            except Exception as write_exc:
                logger.error("could not persist the failure record: %s", write_exc)
        raise

    payload = finalize(payload)
    if write:
        payload["artifacts"] = _write_all(payload, settings, alerts)
        # Rewrite metrics.json once more so it carries the artifact inventory.
        write_results(payload, settings.path("results_dir") / "metrics.json")

    _log_summary(payload)
    return payload, alerts


def _write_all(
    payload: dict[str, Any],
    settings: Settings,
    alerts: list[dict[str, Any]],
) -> dict[str, str]:
    written = write_artifact_bundle(payload, settings, alerts)
    written["metrics_json"] = str(settings.path("results_dir") / "metrics.json")
    try:
        from src.report.build_report import build_reports

        written.update(build_reports(payload, alerts, settings))
    except Exception as exc:  # reporting must not invalidate measurements
        logger.warning("report generation failed: %s", exc)
    return written


def _fmt_metric(value: Any) -> str:
    if value is None or isinstance(value, str):
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _log_summary(payload: dict[str, Any]) -> None:
    logger.info("=" * 78)
    logger.info("EVALUATION STATUS: %s (measured=%s, metrics_published=%s)",
                payload["status"], payload["measured"], payload["metrics_published"])
    for name, block in payload["models"].items():
        metrics = block.get("metrics")
        if not metrics:
            logger.info("  %-16s NOT MEASURED (%s)", name, block.get("reason", block.get("metrics_status")))
            continue
        logger.info(
            "  %-16s acc=%s prec=%s rec=%s f1=%s roc_auc=%s",
            name,
            _fmt_metric(metrics.get("accuracy")),
            _fmt_metric(metrics.get("precision")),
            _fmt_metric(metrics.get("recall")),
            _fmt_metric(metrics.get("f1")),
            _fmt_metric(metrics.get("roc_auc")),
        )
    claim = ((payload.get("comparison") or {}).get("primary") or {}).get("claim")
    if claim:
        logger.info("  COMPARISON: %s", claim.get("claim"))
    validation = payload.get("validation") or {}
    logger.info("  VALIDATION: %s (%d/%d checks passed)",
                "valid" if validation.get("valid") else "FAILED",
                validation.get("n_checks", 0) - validation.get("n_failed", 0),
                validation.get("n_checks", 0))
    logger.info("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the reproducible evaluation pipeline.")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--mode", default="full", help="label recorded in the results file")
    parser.add_argument("--no-write", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    setup_logging(settings)
    try:
        payload, _ = run_pipeline(settings, mode=args.mode, write=not args.no_write)
    except Exception:
        logger.error("evaluation did not complete; see results/metrics.json for the recorded failure")
        return 1
    return 0 if payload.get("metrics_published") else 2


if __name__ == "__main__":
    sys.exit(main())
