"""Streamlit dashboard for the explainable KG-based threat detection prototype.

Run with::

    streamlit run app/dashboard.py

The dashboard is a *viewer*. It never computes or simulates metrics: it reads
``results/metrics.json`` (plus the CSV/JSON artifacts next to it) and renders
them. If results are missing or were not measured, it says so instead of showing
placeholder numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

NODE_COLORS = {
    "ip": "#4C78A8",
    "host": "#72B7B2",
    "service": "#F58518",
    "protocol": "#BAB0AC",
    "event": "#54A24B",
    "technique": "#B279A2",
    "intel": "#E45756",
    "unknown": "#999999",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision (attack)",
    "recall": "Recall (attack)",
    "f1": "F1 (attack)",
    "f1_macro": "F1 (macro)",
    "specificity": "Specificity",
    "balanced_accuracy": "Balanced accuracy",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC",
    "false_positive_rate": "False positive rate",
    "false_negative_rate": "False negative rate",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_json(path_str: str, mtime: float) -> dict[str, Any] | list[Any] | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(show_spinner=False)
def load_csv(path_str: str, mtime: float) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_npz(path_str: str, mtime: float) -> dict[str, np.ndarray] | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with np.load(path) as archive:
            return {key: archive[key] for key in archive.files}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def status_banner(payload: dict[str, Any]) -> None:
    status = payload.get("status", "unknown")
    headline = {
        "measured": ("success", "All stages measured and validated."),
        "partial": ("warning", "Some stages did not complete. Unmeasured values are shown as NOT MEASURED."),
        "failed": ("error", "The pipeline did not produce measured metrics. Do not use any number from this file."),
        "not_executed": ("error", "The pipeline has not been executed successfully yet."),
    }.get(status, ("warning", f"Unrecognised status: {status}"))

    level, message = headline
    text = f"**Status: {status}** — {message}"
    if not payload.get("metrics_published"):
        text += " Metrics are **not published** (validation failed or no model was measured)."
    getattr(st, level)(text)

    dataset = payload.get("dataset") or {}
    if dataset.get("is_synthetic"):
        st.warning(
            "**Synthetic dataset.** These numbers describe the synthetic generator, not real network "
            "traffic. They must not be reported as detection performance."
        )
    validation = payload.get("validation") or {}
    if validation and not validation.get("valid"):
        st.error(
            "Internal validation failed: "
            + ", ".join(validation.get("failed_checks") or ["unknown"])
        )
    failed = payload.get("failed_stages") or []
    if failed:
        st.info("Failed or skipped stages: " + ", ".join(failed))


def metric_row(payload: dict[str, Any]) -> None:
    models = payload.get("models") or {}
    columns = st.columns(4)
    test_rows = ((payload.get("split") or {}).get("test_rows"))
    columns[0].metric("Test records", f"{test_rows:,}" if isinstance(test_rows, int) else "n/a")

    baseline = ((models.get("baseline_rf") or {}).get("metrics")) or {}
    graph = ((models.get("graph_aware_rf") or {}).get("metrics")) or {}
    for index, (label, key) in enumerate((("Baseline F1", "f1"), ("Graph-aware F1", "f1")), start=1):
        source = baseline if index == 1 else graph
        value = source.get(key) if source else None
        columns[index].metric(
            label,
            "NOT MEASURED" if value is None else f"{float(value):.4f}",
            delta=None if (value is None or baseline.get("f1") is None or index == 1) else f"{float(value) - float(baseline['f1']):+.4f}",
        )

    cm = graph.get("confusion_matrix") or baseline.get("confusion_matrix")
    if cm:
        columns[3].metric("True positives (test)", f"{int(cm[1][1]):,}")


def counts_frame(payload: dict[str, Any]) -> pd.DataFrame:
    models = payload.get("models") or {}
    graph = ((models.get("graph_aware_rf") or {}).get("metrics")) or (
        (models.get("baseline_rf") or {}).get("metrics")) or {}
    cm = graph.get("confusion_matrix")
    if not cm:
        return pd.DataFrame()
    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    return pd.DataFrame(
        {
            "group": ["Analysed events", "Predicted benign", "Predicted suspicious", "Correctly flagged attacks", "Missed attacks", "False alarms"],
            "count": [tn + fp + fn + tp, tn + fn, fp + tp, tp, fn, fp],
            "kind": ["total", "prediction", "prediction", "outcome", "outcome", "outcome"],
        }
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_overview(payload: dict[str, Any], artifacts: dict[str, Path]) -> None:
    st.subheader("Overview of analysed events")
    metric_row(payload)

    left, right = st.columns([1, 1])
    with left:
        counts = counts_frame(payload)
        if counts.empty:
            st.info("No confusion matrix available - results were not measured.")
        else:
            figure = px.bar(
                counts[counts["kind"] != "total"], x="group", y="count", color="kind",
                title="Predicted and outcome counts on the test split", text="count",
            )
            figure.update_layout(showlegend=False, xaxis_title="", yaxis_title="events")
            st.plotly_chart(figure, use_container_width=True)
    with right:
        st.markdown("#### Dataset and split")
        dataset = payload.get("dataset") or {}
        split = payload.get("split") or {}
        st.dataframe(
            pd.DataFrame(
                [
                    ["Dataset", dataset.get("name")],
                    ["Origin", dataset.get("origin")],
                    ["Records loaded", dataset.get("rows_loaded")],
                    ["Split strategy", split.get("strategy_used")],
                    ["Training records", split.get("train_rows")],
                    ["Test records", split.get("test_rows")],
                    ["Training attack rate", split.get("train_positive_rate")],
                    ["Test attack rate", split.get("test_positive_rate")],
                    ["Leakage audit passed", (payload.get("leakage_audit") or {}).get("passed_all")],
                ],
                columns=["Property", "Value"],
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### Baseline vs graph-aware")
    comparison = artifacts.get("comparison")
    table = load_csv(str(comparison), _mtime(comparison)) if comparison else None
    if table is None:
        st.info("comparison_table.csv not found.")
    else:
        st.dataframe(table, hide_index=True, use_container_width=True)

    claim = ((payload.get("comparison") or {}).get("primary") or {}).get("claim") or {}
    if claim.get("claim"):
        st.info(claim["claim"])
        if claim.get("rule"):
            st.caption(claim["rule"])

    sensitivity = payload.get("leakage_sensitivity") or {}
    if sensitivity.get("status") == "measured":
        metrics = sensitivity.get("metrics") or {}
        st.markdown("#### Leakage sensitivity (inductive graph scope)")
        st.dataframe(
            pd.DataFrame(
                [
                    ["Inductive graph-aware F1", metrics.get("f1")],
                    ["Transductive graph-aware F1", sensitivity.get("transductive_metric")],
                    ["Transductive minus inductive", sensitivity.get("delta_transductive_minus_inductive_f1")],
                    ["Unseen test entities", sensitivity.get("unseen_entity_fraction_test")],
                ],
                columns=["Property", "Value"],
            ),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Feature importance (global, Gini)"):
        models = payload.get("models") or {}
        frames = []
        for name in ("baseline_rf", "graph_aware_rf"):
            entries = ((models.get(name) or {}).get("top_features")) or []
            for entry in entries[:20]:
                frames.append({"model": name, "feature": entry["feature"], "importance": entry["importance"]})
        if frames:
            frame = pd.DataFrame(frames)
            figure = px.bar(
                frame, x="importance", y="feature", color="model", orientation="h",
                title="Top 20 features per detector", barmode="group",
            )
            figure.update_layout(yaxis={"categoryorder": "total ascending"}, height=650)
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No feature importances recorded.")


def tab_alerts(payload: dict[str, Any], artifacts: dict[str, Path]) -> None:
    st.subheader("Threat alerts and threat scores")
    alerts_path = artifacts.get("alerts_json")
    alerts = load_json(str(alerts_path), _mtime(alerts_path)) if alerts_path else None
    alerts_frame = load_csv(str(artifacts["alerts_csv"]), _mtime(artifacts["alerts_csv"])) if artifacts.get("alerts_csv") else None

    scores = st.session_state.get("scores")
    if scores is not None:
        left, right = st.columns(2)
        with left:
            frame = pd.DataFrame(
                {
                    "threat score": np.concatenate([scores["baseline_rf_score"], scores["graph_aware_rf_score"]]),
                    "detector": ["baseline"] * len(scores["baseline_rf_score"]) + ["graph-aware"] * len(scores["graph_aware_rf_score"]),
                }
            )
            figure = px.histogram(frame, x="threat score", color="detector", nbins=50, barmode="overlay",
                                  title="Distribution of predicted attack probability (test split)")
            st.plotly_chart(figure, use_container_width=True)
        with right:
            truth = scores["y_true"]
            frame = pd.DataFrame(
                {
                    "detector": ["baseline"] * len(truth) + ["graph-aware"] * len(truth),
                    "actual class": np.where(np.concatenate([truth, truth]) == 1, "attack", "benign"),
                    "threat score": np.concatenate([scores["baseline_rf_score"], scores["graph_aware_rf_score"]]),
                }
            )
            figure = px.box(frame, x="detector", y="threat score", color="actual class",
                            title="Score separation by actual class")
            st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("Prediction archive not found; score distributions are unavailable.")

    if alerts_frame is None:
        st.info("alerts.csv not found (no alerts were generated or the pipeline was not run).")
        return

    minimum = st.slider("Minimum threat score", 0.0, 1.0, 0.5, 0.01)
    show_false_only = st.checkbox("Only alerts that are false positives against the dataset label", value=False)
    filtered = alerts_frame[alerts_frame["threat_score"] >= minimum]
    if show_false_only:
        filtered = filtered[filtered["ground_truth_label"] == "benign"]
    st.caption(f"{len(filtered)} of {len(alerts_frame)} explained alerts shown.")
    st.dataframe(
        filtered[["alert_id", "threat_score", "ground_truth_label", "ground_truth_category",
                  "source_ip", "destination_ip", "top_feature_1", "min_hops_to_intel"]],
        hide_index=True, use_container_width=True,
    )

    if alerts and st.checkbox("Show the full explanation of one alert", value=True):
        options = [f"{alert['alert_id']} (score {alert['threat_score']:.3f})" for alert in alerts]
        choice = st.selectbox("Alert", options, index=0)
        render_alert(alerts[options.index(choice)])


def render_alert(alert: dict[str, Any]) -> None:
    st.markdown("### Alert explanation")
    columns = st.columns(4)
    columns[0].metric("Threat score", f"{alert['threat_score']:.4f}")
    columns[1].metric("Predicted", alert.get("predicted_class", "n/a"))
    columns[2].metric("Dataset label", alert.get("ground_truth_label") or "unknown")
    columns[3].metric("Detector", alert.get("model", "n/a"))

    st.markdown(f"**Analyst explanation.** {alert.get('analyst_explanation', '')}")
    st.caption(alert.get("disclaimer", ""))

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Important contributing features")
        features = pd.DataFrame(alert.get("important_features") or [])
        if features.empty:
            st.info("No attributions recorded.")
        else:
            figure = px.bar(
                features.sort_values("attribution"), x="attribution", y="feature", orientation="h",
                color="direction", hover_data=["value", "feature_provenance", "uses_training_labels"],
                title=f"Attribution method: {(alert.get('attribution_method') or {}).get('name')}",
            )
            st.plotly_chart(figure, use_container_width=True)
            st.dataframe(
                features[["feature", "value", "attribution", "direction", "uses_training_labels",
                          "training_percentile", "description"]],
                hide_index=True, use_container_width=True,
            )
            st.caption((alert.get("attribution_method") or {}).get("note", ""))
    with right:
        st.markdown("#### Connected entities")
        entities = pd.DataFrame(alert.get("connected_entities") or [])
        if not entities.empty:
            st.dataframe(entities, hide_index=True, use_container_width=True)
        st.markdown("#### Relationships")
        relationships = alert.get("relationships") or []
        if relationships:
            st.dataframe(
                pd.DataFrame(relationships)[["source", "target", "relation", "confidence",
                                             "confidence_rule", "evidence_count"]],
                hide_index=True, use_container_width=True,
            )
            st.caption("Each relationship carries the rule that produced its confidence value.")
        else:
            st.info("No graph relationships recorded for these entities.")

    st.markdown("#### Supporting graph paths")
    paths = [path for path in alert.get("supporting_graph_paths", []) if path.get("available")]
    if not paths:
        st.info("No path to threat-intel activity was found within the configured hop limit.")
    for path in paths:
        hops = path.get("n_hops")
        if hops in (0, None):
            continue
        st.markdown(f"- **{path.get('role')} {path.get('address')}** — {hops} hop(s): "
                    + " → ".join(path.get("entities", [])))
        st.caption(path.get("interpretation", ""))
        st.dataframe(pd.DataFrame(path.get("hops", [])), hide_index=True, use_container_width=True)

    st.markdown("#### Evidence source and limitations")
    left, right = st.columns([1, 1])
    with left:
        st.json(alert.get("evidence_source") or {})
    with right:
        for limitation in alert.get("limitations", []):
            st.markdown(f"- {limitation}")


def tab_knowledge_graph(payload: dict[str, Any], artifacts: dict[str, Path]) -> None:
    st.subheader("Knowledge graph")
    summary = payload.get("knowledge_graph") or {}
    if summary:
        columns = st.columns(4)
        columns[0].metric("Nodes", f"{summary.get('n_nodes', 0):,}")
        columns[1].metric("Edges", f"{summary.get('n_edges', 0):,}")
        columns[2].metric("Components", summary.get("n_connected_components", "n/a"))
        columns[3].metric("Mean edge confidence",
                          f"{(summary.get('confidence') or {}).get('mean', 'n/a')}")

    path = artifacts.get("kg")
    node_link = load_json(str(path), _mtime(path)) if path else None
    if not node_link:
        st.info("knowledge_graph_subgraph.json not found. Run the pipeline to generate it.")
        return

    nodes = node_link.get("nodes") or []
    edges = node_link.get("edges") or []
    if node_link.get("truncated"):
        st.caption(
            f"Showing a degree-ranked subgraph: {len(nodes):,} of {node_link.get('n_nodes_total'):,} nodes."
        )

    type_counts = pd.Series([node.get("node_type") for node in nodes]).value_counts()
    left, right = st.columns([1, 3])
    with left:
        st.markdown("#### Entity types")
        st.dataframe(type_counts.rename("count"), use_container_width=True)
        st.markdown("#### Relationship types")
        st.dataframe(pd.Series([edge.get("relation") for edge in edges]).value_counts().rename("count"),
                     use_container_width=True)
        focus = st.text_input("Focus on an IP (optional)", value="", help="Show only entities within a few hops of this IP.")
        radius = st.slider("Hops from the focus IP", 1, 3, 1)
        maximum = st.slider("Maximum nodes to draw", 20, 800, 150, 10)
    with right:
        graph = nx.Graph()
        for node in nodes:
            graph.add_node(node["id"], **node)
        for edge in edges:
            if edge["source"] in graph and edge["target"] in graph:
                graph.add_edge(edge["source"], edge["target"], **edge)

        if focus:
            if focus in graph:
                graph = nx.ego_graph(graph, focus, radius=radius)
            else:
                st.warning(f"'{focus}' is not present in the persisted subgraph.")
        if graph.number_of_nodes() > maximum:
            degrees = dict(graph.degree())
            keep = sorted(degrees, key=lambda n: degrees[n], reverse=True)[:maximum]
            graph = graph.subgraph(keep).copy()
        if graph.number_of_nodes() == 0:
            st.info("Nothing to draw.")
            return

        positions = nx.spring_layout(graph, seed=42, k=0.6, iterations=60)
        edge_x, edge_y = [], []
        for source, target in graph.edges():
            edge_x += [positions[source][0], positions[target][0], None]
            edge_y += [positions[source][1], positions[target][1], None]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"width": 0.8, "color": "#CCCCCC"},
                                    hoverinfo="none", showlegend=False))
        for node_type in sorted({data.get("node_type", "unknown") for _, data in graph.nodes(data=True)}):
            selected = [(node, data) for node, data in graph.nodes(data=True) if data.get("node_type") == node_type]
            figure.add_trace(
                go.Scatter(
                    x=[positions[node][0] for node, _ in selected],
                    y=[positions[node][1] for node, _ in selected],
                    mode="markers",
                    name=f"{node_type} ({len(selected)})",
                    marker={"size": [10 + min(int(data.get("evidence_count") or 1), 25) for _, data in selected],
                            "color": NODE_COLORS.get(node_type, NODE_COLORS["unknown"])},
                    text=[f"{data.get('label')}<br>type={node_type}<br>confidence={data.get('confidence')}"
                          f"<br>evidence={data.get('evidence_count')}" for _, data in selected],
                    hoverinfo="text",
                )
            )
        figure.update_layout(
            title=f"Knowledge graph ({graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships)",
            showlegend=True, height=620,
            xaxis={"visible": False}, yaxis={"visible": False}, margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "Edges are observed relationships or training-derived threat-intel flags. Confidence values "
            "follow the documented rules (observed = 1.0, aggregated = 1 - exp(-n/K), intel = smoothed ratio)."
        )
    with st.expander("Graph build log and provenance notes"):
        for line in summary.get("build_log", []) or ["(none)"]:
            st.markdown(f"- {line}")
        st.json((payload.get("graph_features") or {}).get("ip_graph") or {})


def tab_resources(payload: dict[str, Any], artifacts: dict[str, Path]) -> None:
    st.subheader("Resource consumption (measured)")
    resources = payload.get("resources")
    if not resources:
        st.info("No resource measurements recorded: the pipeline did not reach that stage.")
        return

    stages = resources.get("stages") or {}
    rows = []
    for stage, block in stages.items():
        wall = block.get("wall_seconds") or {}
        rows.append(
            {
                "stage": stage,
                "runs": block.get("n_runs"),
                "wall_seconds_mean": wall.get("mean"),
                "wall_seconds_std": wall.get("std"),
                "cpu_avg_pct": (block.get("cpu_percent_avg") or {}).get("mean"),
                "cpu_peak_pct": (block.get("cpu_percent_peak") or {}).get("mean"),
                "rss_peak_mb": (block.get("rss_peak_mb") or {}).get("mean"),
                "tracemalloc_peak_mb": (block.get("tracemalloc_peak_mb") or {}).get("mean"),
                "items_per_second": (block.get("items_per_second") or {}).get("mean"),
                "rss_peak_source": block.get("rss_peak_source"),
            }
        )
    frame = pd.DataFrame(rows).dropna(axis=1, how="all")
    st.dataframe(frame, hide_index=True, use_container_width=True)

    chart_rows = [row for row in rows if row["wall_seconds_mean"] is not None]
    if chart_rows:
        left, right = st.columns(2)
        with left:
            figure = px.bar(
                pd.DataFrame(chart_rows), x="wall_seconds_mean", y="stage", orientation="h",
                error_x="wall_seconds_std", title="Runtime per stage (mean ± std over runs)",
            )
            figure.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(figure, use_container_width=True)
        with right:
            cpu_rows = [row for row in chart_rows if row["cpu_avg_pct"] is not None]
            if cpu_rows:
                melted = pd.DataFrame(cpu_rows).melt(
                    id_vars="stage", value_vars=["cpu_avg_pct", "cpu_peak_pct"],
                    var_name="measure", value_name="percent",
                )
                figure = px.bar(melted, x="percent", y="stage", color="measure", orientation="h",
                                barmode="group", title="CPU utilisation (% of one core; >100% = multiple cores)")
                figure.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(figure, use_container_width=True)

        memory_rows = [row for row in chart_rows if row["rss_peak_mb"] is not None]
        if memory_rows:
            figure = px.bar(pd.DataFrame(memory_rows), x="stage", y="rss_peak_mb",
                            title="Peak process RAM per stage (MB)")
            st.plotly_chart(figure, use_container_width=True)

        throughput_rows = [row for row in chart_rows if row["items_per_second"] is not None]
        if throughput_rows:
            figure = px.bar(pd.DataFrame(throughput_rows), x="stage", y="items_per_second",
                            title="Throughput per stage (events/s)")
            st.plotly_chart(figure, use_container_width=True)

    overhead = (payload.get("resource_overhead") or {}).get("inference")
    if overhead:
        st.markdown("#### Inference overhead: graph-aware vs baseline")
        table = pd.DataFrame(overhead.get("rows") or [])
        st.dataframe(table, hide_index=True, use_container_width=True)
        relative = table.dropna(subset=["relative_change_percent"]) if "relative_change_percent" in table else pd.DataFrame()
        if not relative.empty:
            figure = px.bar(relative, x="metric", y="relative_change_percent",
                            title="Relative change from adding graph features (%)")
            st.plotly_chart(figure, use_container_width=True)
        st.caption(overhead.get("note", ""))

    if artifacts.get("resources_csv"):
        st.caption("Per-run measurements are available in results/resources.csv.")

    with st.expander("Measurement caveats (read before quoting these numbers)"):
        for caveat in resources.get("caveats", []):
            st.markdown(f"- {caveat}")
        st.markdown(f"- {resources.get('one_time_vs_per_event', '')}")
        st.json(
            {
                "n_runs": resources.get("n_runs"),
                "sample_interval_seconds": resources.get("sample_interval_seconds"),
                "tracemalloc_enabled": resources.get("tracemalloc_enabled"),
                "environment": payload.get("environment", {}).get("cpu"),
                "python": payload.get("environment", {}).get("python_version"),
                "os": payload.get("environment", {}).get("os"),
            }
        )


def tab_downloads(payload: dict[str, Any], artifacts: dict[str, Path]) -> None:
    st.subheader("Evaluation report and downloadable results")
    status = payload.get("status")
    if status != "measured":
        st.warning(
            "The full report may contain NOT MEASURED entries because the pipeline status is "
            f"'{status}'. That is intentional."
        )

    for key, label in (
        ("metrics_json", "metrics.json (all measured values)"),
        ("comparison_csv", "comparison_table.csv (baseline vs graph-aware)"),
        ("alerts_csv", "alerts.csv (explained alerts)"),
        ("resources_csv", "resources.csv (per-run resource measurements)"),
        ("report_markdown", "results_for_paper.md (filled template)"),
        ("leakage_audit_json", "leakage_audit.json"),
    ):
        path = artifacts.get(key)
        if path and Path(path).exists():
            data = Path(path).read_bytes()
            st.download_button(label, data=data, file_name=Path(path).name, key=f"download-{key}")

    report_path = artifacts.get("report_markdown")
    if report_path and Path(report_path).exists():
        with st.expander("Filled paper report", expanded=False):
            st.markdown(Path(report_path).read_text(encoding="utf-8"))
    else:
        st.info("The filled paper report has not been generated yet.")

    with st.expander("Raw results payload"):
        st.json(payload)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Explainable KG Threat Detection", layout="wide", page_icon="🛡️")
    st.title("🛡️ Explainable Knowledge Graph-Based Cyber Threat Detection")
    st.caption(
        "Research prototype dashboard. It renders measured values only; where a stage did not run, "
        "it shows NOT MEASURED rather than a placeholder."
    )

    results_dir = PROJECT_ROOT / "results"
    default = results_dir / "metrics.json"

    with st.sidebar:
        st.header("Results source")
        chosen = st.text_input("Path to metrics.json", value=str(default))
        payload = load_json(chosen, _mtime(Path(chosen)))
        uploaded = st.file_uploader("...or upload metrics.json", type="json")
        if uploaded is not None:
            try:
                payload = json.loads(uploaded.getvalue().decode("utf-8"))
            except json.JSONDecodeError:
                st.error("That file is not valid JSON.")
        st.divider()
        st.markdown(
            "**Generate results**\n\n"
            "```bash\npython run_all.py\n```\n\n"
            "**Open the dashboard**\n\n"
            "```bash\nstreamlit run app/dashboard.py\n```"
        )

    if not payload:
        st.error(
            f"No results found at `{chosen}`. Run the pipeline first:\n\n"
            "```bash\npython run_all.py\n```"
        )
        st.stop()

    results_root = Path(chosen).parent
    artifacts: dict[str, Path] = {
        "metrics_json": Path(chosen),
        "comparison_csv": results_root / "comparison_table.csv",
        "alerts_csv": results_root / "alerts.csv",
        "alerts_json": results_root / "alerts.json",
        "resources_csv": results_root / "resources.csv",
        "feature_importance_csv": results_root / "feature_importance.csv",
        "leakage_audit_json": results_root / "leakage_audit.json",
        "report_markdown": results_root / "reports" / "results_for_paper.md",
        "kg": results_root / "knowledge_graph_subgraph.json",
    }
    predictions = results_root / "predictions" / "test_predictions.npz"
    st.session_state["scores"] = load_npz(str(predictions), _mtime(predictions))

    status_banner(payload)
    st.divider()

    overview, alerts_tab, graph_tab, resources_tab, downloads_tab = st.tabs(
        ["Overview", "Events & alerts", "Knowledge graph", "Resources", "Report"]
    )

    with overview:
        tab_overview(payload, artifacts)
    with alerts_tab:
        tab_alerts(payload, artifacts)
    with graph_tab:
        tab_knowledge_graph(payload, artifacts)
    with resources_tab:
        tab_resources(payload, artifacts)
    with downloads_tab:
        tab_downloads(payload, artifacts)


if __name__ == "__main__":
    main()
