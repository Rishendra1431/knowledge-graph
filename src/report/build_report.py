"""Report generation.

``docs/RESULTS_TEMPLATE.md`` holds the skeleton for a research write-up with two
kinds of placeholder:

* ``[[dotted.path]]``      - a scalar taken from ``results/metrics.json``
* ``[[TABLE:name]]``       - a markdown table generated from the payload

Rendering substitutes measured values and, crucially, writes ``NOT MEASURED``
for anything that is ``null``, missing or from a stage that did not run. Any
placeholder that cannot be resolved at all is listed in a final section so a
typo can never silently become a blank in a paper table.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import Settings, PROJECT_ROOT, get_logger

logger = get_logger("report.build")

NOT_MEASURED = "NOT MEASURED"

EXCLUDED_PREFIXES = (
    "config.",
    "environment.config_snapshot",
    "models.baseline_rf.metrics_per_run",
    "models.graph_aware_rf.metrics_per_run",
    "models.baseline_rf.inference_per_run",
    "models.graph_aware_rf.inference_per_run",
    "models.baseline_rf.predictor.feature_columns",
    "models.graph_aware_rf.predictor.feature_columns",
    "graph_features.feature_documentation",
    "validation.checks",
    "leakage_audit.checks",
    "preprocessing.encoder_metadata.imputation.medians",
    "preprocessing.encoder_metadata.outlier_clipping.bounds",
    "preprocessing.cleaning_report.missing_values_per_column_top",
    "resources.stages",
)

TOKEN_PATTERN = re.compile(r"\[\[([A-Za-z0-9_.:\-]+)\]\]")


# ---------------------------------------------------------------------------
# Flattening and formatting
# ---------------------------------------------------------------------------
def flatten(payload: Any, prefix: str = "", exclude: tuple[str, ...] = EXCLUDED_PREFIXES) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}{key}"
            if any(path == e.rstrip(".") or path.startswith(e) for e in exclude):
                continue
            out.update(flatten(value, f"{path}."))
    elif isinstance(payload, list):
        out[f"{prefix.rstrip('.')}.__count__"] = len(payload)
        for index, value in enumerate(payload):
            out.update(flatten(value, f"{prefix}{index}."))
    else:
        out[prefix.rstrip(".")] = payload
    return out


def format_value(value: Any) -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:
            return NOT_MEASURED
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        if abs(value) < 1e-4 and value != 0.0:
            return f"{value:.3e}"
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    text = str(value)
    return text if text else NOT_MEASURED


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, float) and value != value:
        return NOT_MEASURED
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _mean_std(block: dict[str, Any] | None) -> str:
    if not block:
        return NOT_MEASURED
    mean, std = block.get("mean"), block.get("std")
    if mean is None:
        return NOT_MEASURED
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return f"| {' | '.join(headers)} |\n|{'|'.join([' --- '] * len(headers))}|\n| {' | '.join([NOT_MEASURED] * len(headers))} |"
    lines = [f"| {' | '.join(headers)} |", f"|{'|'.join([' --- '] * len(headers))}|"]
    for row in rows:
        lines.append(f"| {' | '.join('' if cell is None else str(cell) for cell in row)} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def build_tables(payload: dict[str, Any], alerts: list[dict[str, Any]] | None = None) -> dict[str, str]:
    alerts = alerts or []
    models = payload.get("models") or {}
    tables: dict[str, str] = {}

    # Comparison table (deliverable 9)
    headers = ["Metric", "Baseline (flow features)", "Graph-aware (+ graph features)", "Difference (B-A)", "Relative change"]
    rows: list[list[Any]] = []
    metric_order = [
        ("accuracy", "Accuracy"), ("precision", "Precision (attack class)"), ("recall", "Recall (attack class)"),
        ("f1", "F1 (attack class)"), ("f1_macro", "F1 (macro)"), ("specificity", "Specificity"),
        ("balanced_accuracy", "Balanced accuracy"), ("mcc", "Matthews correlation coefficient"),
        ("roc_auc", "ROC-AUC"), ("average_precision", "PR-AUC (average precision)"),
        ("false_positive_rate", "False positive rate"), ("false_negative_rate", "False negative rate"),
        ("true_positives", "True positives"), ("false_positives", "False positives"),
        ("false_negatives", "False negatives"), ("true_negatives", "True negatives"),
        ("n_samples", "Test records"),
    ]
    baseline_metrics = ((models.get("baseline_rf") or {}).get("metrics")) or {}
    graph_metrics = ((models.get("graph_aware_rf") or {}).get("metrics")) or {}
    for key, label in metric_order:
        left = baseline_metrics.get(key)
        right = graph_metrics.get(key)
        if left is None and right is None:
            rows.append([label, NOT_MEASURED, NOT_MEASURED, NOT_MEASURED, NOT_MEASURED])
            continue
        difference = None if left is None or right is None else float(right) - float(left)
        relative = None if not left or right is None else 100.0 * (float(right) - float(left)) / float(left)
        digits = 0 if key in ("true_positives", "false_positives", "false_negatives", "true_negatives", "n_samples") else 4
        rows.append([
            label,
            _number(left, digits),
            _number(right, digits),
            NOT_MEASURED if difference is None else f"{difference:+.4f}",
            NOT_MEASURED if relative is None else f"{relative:+.2f}%",
        ])
    tables["comparison"] = _markdown_table(headers, rows)

    # Confusion matrices
    for name, key in (("baseline_rf", "confusion_baseline"), ("graph_aware_rf", "confusion_graph")):
        metrics = ((models.get(name) or {}).get("metrics")) or {}
        cm = metrics.get("confusion_matrix")
        if not cm:
            tables[key] = _markdown_table(["", "Predicted benign", "Predicted attack"], [["---", NOT_MEASURED, NOT_MEASURED]] * 2)
            continue
        tables[key] = _markdown_table(
            ["", "Predicted benign", "Predicted attack"],
            [["Actual benign", cm[0][0], cm[0][1]], ["Actual attack", cm[1][0], cm[1][1]]],
        )

    # Per-class metrics
    for name, key in (("baseline_rf", "per_class_baseline"), ("graph_aware_rf", "per_class_graph")):
        per_class = ((models.get(name) or {}).get("metrics") or {}).get("per_class") or {}
        rows = [
            [label, _number(values.get("precision")), _number(values.get("recall")),
             _number(values.get("f1_score")), values.get("support")]
            for label, values in per_class.items()
        ]
        tables[key] = _markdown_table(["Class", "Precision", "Recall", "F1", "Support"], rows)

    # Resources per stage
    stages = ((payload.get("resources") or {}).get("stages")) or {}
    rows = []
    for stage, block in stages.items():
        rows.append([
            stage,
            block.get("n_runs"),
            _mean_std(block.get("wall_seconds")),
            _mean_std(block.get("cpu_percent_avg")),
            _mean_std(block.get("cpu_percent_peak")),
            _mean_std(block.get("rss_peak_mb")),
            _mean_std(block.get("tracemalloc_peak_mb")),
            _mean_std(block.get("items_per_second")),
        ])
    tables["resources"] = _markdown_table(
        ["Stage", "Runs", "Wall time (s)", "CPU avg (%)", "CPU peak (%)", "Peak RAM (MB)",
         "tracemalloc peak (MB)", "Throughput (events/s)"],
        rows,
    )

    # Resource overhead
    overhead = (payload.get("resource_overhead") or {}).get("inference") or {}
    rows = [
        [row.get("metric"), _number(row.get("baseline_rf"), 4) if not isinstance(row.get("baseline_rf"), int) else row.get("baseline_rf"),
         _number(row.get("graph_aware_rf"), 4) if not isinstance(row.get("graph_aware_rf"), int) else row.get("graph_aware_rf"),
         _number(row.get("absolute_difference"), 4),
         NOT_MEASURED if row.get("relative_change_percent") is None else f"{row['relative_change_percent']:+.2f}%"]
        for row in overhead.get("rows", [])
    ]
    tables["overhead"] = _markdown_table(
        ["Inference metric", "Baseline", "Graph-aware", "Absolute difference", "Relative change"], rows
    )

    # Leakage audit
    audit = payload.get("leakage_audit") or {}
    rows = [
        [item.get("name"), "PASS" if item.get("passed") else "FAIL", item.get("detail")]
        for item in audit.get("checks", [])
    ]
    tables["leakage_checks"] = _markdown_table(["Check", "Result", "Detail"], rows)

    # Pipeline stages
    rows = [
        [stage.get("stage"), stage.get("status"), stage.get("detail")]
        for stage in payload.get("stages", [])
    ]
    tables["stages"] = _markdown_table(["Stage", "Status", "Detail"], rows)

    # Feature importance
    for name, key in (("baseline_rf", "importance_baseline"), ("graph_aware_rf", "importance_graph")):
        entries = ((models.get(name) or {}).get("top_features")) or []
        rows = [[index + 1, item.get("feature"), _number(item.get("importance"), 6)]
                for index, item in enumerate(entries[:15])]
        tables[key] = _markdown_table(["Rank", "Feature", "Gini importance"], rows)

    # Alerts
    rows = []
    for alert in alerts[:20]:
        entities = {entity.get("role"): entity.get("value") for entity in alert.get("connected_entities", [])}
        features = alert.get("important_features") or []
        paths = [p for p in alert.get("supporting_graph_paths", []) if p.get("available") and p.get("n_hops", 0) > 0]
        rows.append([
            alert.get("alert_id"),
            _number(alert.get("threat_score")),
            alert.get("ground_truth_label") or "unknown",
            entities.get("source"),
            entities.get("destination"),
            features[0].get("feature") if features else None,
            min((p.get("n_hops") for p in paths), default=None),
        ])
    tables["alerts"] = _markdown_table(
        ["Alert", "Threat score", "Dataset label", "Source IP", "Destination IP", "Top feature", "Min hops to intel"],
        rows,
    )

    # Limitations collected from the generated explanations
    unique: list[str] = []
    for alert in alerts:
        for limitation in alert.get("limitations", []):
            if limitation not in unique:
                unique.append(limitation)
    tables["limitations"] = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(unique)) or NOT_MEASURED

    return tables


def build_scalars(payload: dict[str, Any]) -> dict[str, str]:
    flat = flatten(payload)
    scalars = {key: format_value(value) for key, value in flat.items()}
    # Convenience aliases used by the template.
    aliases = {
        "status.pipeline_status": payload.get("status"),
        "status.measured": payload.get("measured"),
        "status.metrics_published": payload.get("metrics_published"),
        "status.failed_stages": payload.get("failed_stages"),
        "status.generated_at": payload.get("generated_at_utc"),
        "dataset.is_synthetic": (payload.get("dataset") or {}).get("is_synthetic"),
        "dataset.origin": (payload.get("dataset") or {}).get("origin"),
        "dataset.name": (payload.get("dataset") or {}).get("name"),
        "dataset.notes": (payload.get("dataset") or {}).get("notes"),
        "graph.scope": payload.get("graph_scope"),
        "graph.n_nodes": (payload.get("knowledge_graph") or {}).get("n_nodes"),
        "graph.n_edges": (payload.get("knowledge_graph") or {}).get("n_edges"),
        "graph.n_components": (payload.get("knowledge_graph") or {}).get("n_connected_components"),
        "graph.community_method": ((payload.get("graph_features") or {}).get("community_detection")),
        "intel.n_flagged_ips": (payload.get("threat_intel_index") or {}).get("n_flagged_ips"),
        "intel.source": (payload.get("threat_intel_index") or {}).get("source"),
        "comparison.claim": ((payload.get("comparison") or {}).get("primary") or {}).get("claim", {}).get("claim"),
        "comparison.delta_f1": ((payload.get("comparison") or {}).get("primary") or {}).get("bootstrap", {}).get("observed_difference_b_minus_a"),
        "comparison.ci_lower": ((payload.get("comparison") or {}).get("primary") or {}).get("bootstrap", {}).get("ci_lower"),
        "comparison.ci_upper": ((payload.get("comparison") or {}).get("primary") or {}).get("bootstrap", {}).get("ci_upper"),
        "comparison.p_value": ((payload.get("comparison") or {}).get("primary") or {}).get("bootstrap", {}).get("p_value_two_sided"),
        "comparison.mcnemar_p": ((payload.get("comparison") or {}).get("primary") or {}).get("mcnemar", {}).get("p_value"),
        "comparison.consistent_across_runs": ((payload.get("comparison") or {}).get("primary") or {}).get("consistent_across_runs"),
        "split.strategy": (payload.get("split") or {}).get("strategy_used"),
        "split.train_rows": (payload.get("split") or {}).get("train_rows"),
        "split.test_rows": (payload.get("split") or {}).get("test_rows"),
        "features.n_tabular": (payload.get("preprocessing") or {}).get("n_features_tabular"),
        "features.n_graph": (payload.get("preprocessing") or {}).get("n_features_graph"),
        "features.n_total": (payload.get("preprocessing") or {}).get("n_features_total"),
        "leakage.passed_all": (payload.get("leakage_audit") or {}).get("passed_all"),
        "leakage.n_failed": (payload.get("leakage_audit") or {}).get("n_failed"),
        "validation.valid": (payload.get("validation") or {}).get("valid"),
        "validation.n_failed": (payload.get("validation") or {}).get("n_failed"),
        "explanations.n_alerts": (payload.get("explanations") or {}).get("n_alerts"),
        "explanations.method": ((payload.get("explanations") or {}).get("attribution_method") or {}).get("name"),
        "explanations.mean_paths": (payload.get("explanations") or {}).get("mean_supporting_paths_per_alert"),
        "explanations.mean_relationships": (payload.get("explanations") or {}).get("mean_relationships_per_alert"),
        "environment.cpu_brand": ((payload.get("environment") or {}).get("cpu") or {}).get("cpu_brand"),
        "environment.logical_cores": ((payload.get("environment") or {}).get("cpu") or {}).get("logical_cores"),
        "environment.physical_cores": ((payload.get("environment") or {}).get("cpu") or {}).get("physical_cores"),
        "environment.total_ram_mb": ((payload.get("environment") or {}).get("cpu") or {}).get("total_ram_mb"),
        "environment.os": (payload.get("environment") or {}).get("os"),
        "environment.python_version": (payload.get("environment") or {}).get("python_version"),
    }
    for key, value in aliases.items():
        scalars[key] = format_value(value)
    return scalars


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_report(
    payload: dict[str, Any],
    alerts: list[dict[str, Any]] | None,
    template_text: str,
) -> tuple[str, list[str]]:
    scalars = build_scalars(payload)
    tables = build_tables(payload, alerts)
    unresolved: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token.startswith("TABLE:"):
            name = token.split(":", 1)[1]
            if name in tables:
                return tables[name]
            unresolved.append(token)
            return f"{NOT_MEASURED} (table '{name}' unavailable)"
        if token in scalars:
            return scalars[token]
        unresolved.append(token)
        return NOT_MEASURED

    rendered = TOKEN_PATTERN.sub(_replace, template_text)
    if unresolved:
        rendered += (
            "\n\n---\n\n## Unresolved placeholders\n\n"
            "The following template tokens had no corresponding value in `results/metrics.json`. "
            "They are listed here instead of being silently dropped:\n\n"
            + "\n".join(f"- `[[{token}]]`" for token in sorted(set(unresolved)))
            + "\n"
        )
    return rendered, sorted(set(unresolved))


def build_reports(
    payload: dict[str, Any],
    alerts: list[dict[str, Any]] | None,
    settings: Settings,
    template_path: Path | None = None,
) -> dict[str, str]:
    """Write the filled paper report and the standalone comparison table."""
    reports_dir = settings.path("reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    template_file = template_path or (PROJECT_ROOT / "docs" / "RESULTS_TEMPLATE.md")
    if template_file.exists():
        rendered, unresolved = render_report(payload, alerts, template_file.read_text(encoding="utf-8"))
        header = (
            f"<!-- generated from {template_file.name} at "
            f"{datetime.now(timezone.utc).isoformat()} by src/report/build_report.py -->\n"
            f"<!-- pipeline status: {payload.get('status')} | metrics published: {payload.get('metrics_published')} -->\n\n"
        )
        target = reports_dir / "results_for_paper.md"
        target.write_text(header + rendered, encoding="utf-8")
        written["report_markdown"] = str(target)
        if unresolved:
            (reports_dir / "unresolved_placeholders.txt").write_text("\n".join(unresolved), encoding="utf-8")
            written["unresolved_placeholders"] = str(reports_dir / "unresolved_placeholders.txt")
    else:
        logger.warning("results template not found at %s; writing tables only", template_file)

    comparison = reports_dir / "comparison_table.md"
    comparison.write_text(
        "# Baseline vs graph-aware detection\n\n"
        f"Pipeline status: **{payload.get('status')}** "
        f"(metrics published: {payload.get('metrics_published')})\n\n"
        + (build_tables(payload, alerts).get("comparison") or NOT_MEASURED)
        + "\n\n"
        + str(((payload.get("comparison") or {}).get("primary") or {}).get("claim", {}).get("claim", NOT_MEASURED))
        + "\n",
        encoding="utf-8",
    )
    written["comparison_markdown"] = str(comparison)

    resources = reports_dir / "resource_evaluation.md"
    tables = build_tables(payload, alerts)
    resources.write_text(
        "# Resource evaluation\n\n"
        "Measured on the machine recorded in `results/metrics.json` (`environment` block). "
        "Caveats that apply to these numbers:\n\n"
        + "\n".join(f"- {caveat}" for caveat in ((payload.get("resources") or {}).get("caveats") or [NOT_MEASURED]))
        + "\n\n## Per-stage measurements\n\n"
        + (tables.get("resources") or NOT_MEASURED)
        + "\n\n## Baseline vs graph-aware inference overhead\n\n"
        + (tables.get("overhead") or NOT_MEASURED)
        + "\n",
        encoding="utf-8",
    )
    written["resource_report"] = str(resources)
    return written
