"""Explainable alerts.

For every detected suspicious event the explainer assembles, from data that
actually exists:

1. predicted class, threat score, threshold and decision;
2. the strongest contributing features, with their values, direction and an
   explicit statement of which attribution method produced them
   (SHAP when installed, otherwise a clearly labelled approximation);
3. the connected entities and relationships taken from the knowledge graph,
   each with its confidence value, confidence rule and provenance;
4. supporting graph paths to training-derived threat intel, when such a path
   exists within the configured hop limit;
5. the evidence source (dataset, file, record id, timestamp) and the limitations
   that apply to *this* alert;
6. an analyst-facing paragraph built only from the facts above.

Two rules are enforced in code and in the generated text:

* A graph path is reported as *proximity context*, never as proof of intent.
* The threat score is reported as a statistical model output, never as proof of
  malicious activity.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings, get_logger
from src.data.preprocess import CATEGORY_COLUMN, TIMESTAMP_INTERNAL
from src.graph.builder import KnowledgeGraph
from src.graph.features import FEATURE_DESCRIPTIONS, GraphFeatureExtractor
from src.graph.paths import resolve_entity_columns
from src.models.detectors import ThreatDetector

logger = get_logger("explain.explainer")

METHOD_SHAP = "shap_tree_explainer"
METHOD_APPROX = "tree_importance_x_standardised_deviation"

BASE_LIMITATIONS = [
    "A detection score is a statistical output of a model trained on historical traffic; "
    "it is not proof of malicious intent and does not by itself justify an enforcement action.",
    "Feature attributions explain the model, not the adversary: they describe which inputs moved "
    "the score, not why the traffic occurred.",
    "Absence of a graph path to known threat activity does not indicate benignity.",
]

GRAPH_SCOPE_LIMITATION = (
    "Structural graph features were computed over the configured graph scope. In the transductive "
    "'full' scope the graph covers all evaluated events (labels still never enter the graph); in the "
    "'train_only' scope the graph covers training events and unseen entities fall back to sentinels."
)


class AlertExplainer:
    """Builds explanations for individual alerts."""

    def __init__(
        self,
        detector: ThreatDetector,
        settings: Settings,
        graph_extractor: GraphFeatureExtractor | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        intel_index: dict[str, Any] | None = None,
        dataset_name: str = "unknown",
        dataset_origin: str = "unknown",
        feature_reference: pd.DataFrame | None = None,
    ) -> None:
        self.detector = detector
        self.settings = settings
        self.graph_extractor = graph_extractor
        self.knowledge_graph = knowledge_graph
        self.intel_index = intel_index or {"available": False, "ips": {}}
        self.dataset_name = dataset_name
        self.dataset_origin = dataset_origin
        self.method = "unavailable"
        self.method_note = "no attribution method initialised"
        self._explainer = None
        self._reference_median: pd.Series | None = None
        self._reference_std: pd.Series | None = None

        # Reference statistics come from the TRAINING split only. A deterministic
        # subsample keeps the per-alert percentile lookup cheap on large splits.
        self._reference_values: pd.DataFrame | None = None
        if feature_reference is not None and len(feature_reference):
            numeric = feature_reference[self.detector.feature_columns].astype("float64")
            self._reference_median = numeric.median()
            std = numeric.std(ddof=0).replace(0.0, np.nan)
            self._reference_std = std
            cap = 20000
            if len(numeric) > cap:
                step = max(len(numeric) // cap, 1)
                numeric = numeric.iloc[::step].head(cap)
            self._reference_values = numeric

        self._initialise_attribution()

    # ------------------------------------------------------------------
    def _initialise_attribution(self) -> None:
        requested = self.settings.explain.method
        if requested in ("auto", "shap"):
            try:
                import shap  # noqa: F401

                self._explainer = shap.TreeExplainer(self.detector.model)
                self.method = METHOD_SHAP
                self.method_note = (
                    "SHAP TreeExplainer on the fitted RandomForest. Values are Shapley values for the "
                    "attack class in log-odds space; they are model attributions, not causal effects."
                )
                logger.info("explanations: using SHAP TreeExplainer")
                return
            except Exception as exc:
                if requested == "shap":
                    logger.warning("shap requested but unavailable (%s); falling back", exc)
                else:
                    logger.info("shap not available (%s); using the documented approximation", exc)
        self._explainer = None
        self.method = METHOD_APPROX
        self.method_note = (
            "APPROXIMATION, not SHAP: global tree importance multiplied by the feature's standardised "
            "deviation from the training median. It shows which features are influential for this row "
            "but does not satisfy Shapley additivity, so magnitudes must not be compared across rows."
        )

    # ------------------------------------------------------------------
    def attribute_features(self, X: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
        """Per-row feature attributions for the rows in ``X``."""
        k = int(top_k or self.settings.explain.top_k_features)
        frame = X[self.detector.feature_columns]
        if self.method == METHOD_SHAP and self._explainer is not None:
            try:
                values = self._explainer.shap_values(frame.to_numpy(dtype="float64"))
                matrix = self._normalise_shap(values, len(frame))
                return pd.DataFrame(matrix, index=frame.index, columns=self.detector.feature_columns)
            except Exception as exc:
                logger.warning("SHAP computation failed (%s); using the approximation instead", exc)
                self.method = METHOD_APPROX
                self.method_note = (
                    "SHAP was requested but failed at runtime; falling back to the documented "
                    "approximation (global importance x standardised deviation)."
                )

        importances = self.detector.feature_importances()
        if self._reference_median is not None:
            median = self._reference_median.reindex(frame.columns)
            std = self._reference_std.reindex(frame.columns)
            deviation = (frame - median).div(std.replace(0.0, np.nan)).fillna(0.0)
        else:
            deviation = frame.sub(frame.mean()).div(frame.std(ddof=0).replace(0.0, np.nan)).fillna(0.0)
        weighted = deviation.mul(importances.reindex(frame.columns), axis=1)
        return weighted.fillna(0.0)

    @staticmethod
    def _normalise_shap(values: Any, n_rows: int) -> np.ndarray:
        """SHAP return shapes differ across versions; normalise to (rows, features)."""
        array = np.asarray(values)
        if isinstance(values, list):  # [class0, class1] each (rows, features)
            array = np.asarray(values[-1])
        if array.ndim == 3:  # (rows, features, classes)
            array = array[:, :, -1]
        if array.shape[0] != n_rows and array.shape[-1] == n_rows:
            array = array.T
        return array

    # ------------------------------------------------------------------
    def _entities(self, row: pd.Series) -> list[dict[str, Any]]:
        columns = resolve_entity_columns(pd.DataFrame([row]))
        entities: list[dict[str, Any]] = []
        for role, key in (("source", "src_ip"), ("destination", "dst_ip")):
            column = columns.get(key)
            if not column or pd.isna(row.get(column)):
                continue
            address = str(row[column])
            node_id = f"ip:{address}"
            attributes = {}
            if self.knowledge_graph is not None and node_id in self.knowledge_graph.graph:
                attributes = dict(self.knowledge_graph.graph.nodes[node_id])
                attributes.pop("provenance_id", None)
            intel = (self.intel_index.get("ips") or {}).get(address) if self.intel_index.get("available") else None
            entities.append(
                {
                    "role": role,
                    "type": "ip",
                    "value": address,
                    "knowledge_graph_node": node_id if attributes else None,
                    "graph_attributes": attributes or None,
                    "threat_intel_evidence": intel,
                    "is_internal": bool(attributes.get("is_internal", False)),
                }
            )
        for role, key, node_prefix, node_type, extra_key in (
            ("destination_service", "dst_port", "svc", "service", None),
            ("protocol", "protocol", "proto", "protocol", None),
        ):
            column = columns.get(key)
            if not column or pd.isna(row.get(column)):
                continue
            value = row[column]
            entities.append(
                {
                    "role": role,
                    "type": node_type,
                    "value": value,
                    "source_column": column,
                }
            )
        return entities

    def _relationships(self, row: pd.Series) -> list[dict[str, Any]]:
        """Observed relationships touching this event's entities, from the KG."""
        if self.knowledge_graph is None:
            return []
        columns = resolve_entity_columns(pd.DataFrame([row]))
        endpoints = []
        for key in ("src_ip", "dst_ip"):
            column = columns.get(key)
            if column and not pd.isna(row.get(column)):
                node = f"ip:{row[column]}"
                if node in self.knowledge_graph.graph:
                    endpoints.append(node)
        relationships: list[dict[str, Any]] = []
        for node in endpoints:
            for source, target, data in self.knowledge_graph.graph.in_edges(node, data=True):
                if data.get("relation") in ("COMMUNICATES_WITH", "FLAGGED_AS_INTEL", "USES_SERVICE"):
                    relationships.append(
                        {
                            "source": source,
                            "target": target,
                            "relation": data.get("relation"),
                            "confidence": data.get("confidence"),
                            "confidence_rule": data.get("confidence_rule"),
                            "evidence_count": data.get("evidence_count"),
                            "provenance": self.knowledge_graph.provenance_of(data.get("provenance_id", "")),
                        }
                    )
                if len(relationships) >= 25:
                    break
            for source, target, data in self.knowledge_graph.graph.out_edges(node, data=True):
                if data.get("relation") in ("COMMUNICATES_WITH", "USES_SERVICE", "ALIAS_OF"):
                    relationships.append(
                        {
                            "source": source,
                            "target": target,
                            "relation": data.get("relation"),
                            "confidence": data.get("confidence"),
                            "confidence_rule": data.get("confidence_rule"),
                            "evidence_count": data.get("evidence_count"),
                            "provenance": self.knowledge_graph.provenance_of(data.get("provenance_id", "")),
                        }
                    )
                if len(relationships) >= 50:
                    break
        deduplicated: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in relationships:
            deduplicated[(item["source"], item["target"], str(item["relation"]))] = item
        return list(deduplicated.values())[:50]

    def _paths(self, row: pd.Series) -> list[dict[str, Any]]:
        if self.graph_extractor is None:
            return []
        columns = resolve_entity_columns(pd.DataFrame([row]))
        paths: list[dict[str, Any]] = []
        for role, key in (("source", "src_ip"), ("destination", "dst_ip")):
            column = columns.get(key)
            if not column or pd.isna(row.get(column)):
                continue
            address = str(row[column])
            if not self.graph_extractor.intel_paths:
                continue
            evidence = self.graph_extractor.path_evidence(address)
            evidence["role"] = role
            evidence["address"] = address
            paths.append(evidence)
        return paths

    # ------------------------------------------------------------------
    def explain_row(
        self,
        row: pd.Series,
        y_score: float,
        y_pred: int,
        attributions: pd.Series,
        true_label: int | None = None,
    ) -> dict[str, Any]:
        top_k = int(self.settings.explain.top_k_features)
        ranked = attributions.reindex(attributions.abs().sort_values(ascending=False).index).head(top_k)

        contributing_features: list[dict[str, Any]] = []
        for name, value in ranked.items():
            documentation = FEATURE_DESCRIPTIONS.get(name, {})
            observed = row.get(name)
            percentile = self._percentile_of(name, observed)
            contributing_features.append(
                {
                    "feature": name,
                    "value": None if observed is None or (isinstance(observed, float) and np.isnan(observed)) else float(observed),
                    "attribution": round(float(value), 8),
                    "direction": "pushes toward attack" if value > 0 else "pushes toward benign",
                    "method": self.method,
                    "description": documentation.get("description"),
                    "feature_provenance": documentation.get("source"),
                    "uses_training_labels": documentation.get("uses_training_labels"),
                    "training_percentile": None if percentile is None else round(float(percentile), 4),
                }
            )

        entities = self._entities(row)
        relationships = self._relationships(row)
        paths = self._paths(row)

        record_id = row.get("__record_id")
        timestamp = row.get(TIMESTAMP_INTERNAL)
        source_file = row.get("__source_file")
        attack_category = row.get(CATEGORY_COLUMN)

        limitations = list(BASE_LIMITATIONS)
        if self.graph_extractor is not None:
            limitations.append(GRAPH_SCOPE_LIMITATION)
        if self.method == METHOD_APPROX:
            limitations.append(
                "Feature attributions use an approximation, not SHAP: only the ranking and sign of the "
                "contributions are meaningful."
            )
        if self.dataset_origin == "synthetic":
            limitations.append(
                "The model was trained and evaluated on SYNTHETIC data. This alert demonstrates that the "
                "mechanism works; it says nothing about real-world detection performance."
            )
        supervised = [f["feature"] for f in contributing_features if f.get("uses_training_labels")]
        if supervised:
            limitations.append(
                "The following contributing features are derived from training-split labels (hindsight "
                f"information): {', '.join(supervised)}. They would have to come from an external threat-"
                "intelligence feed in a live deployment."
            )
        if not paths:
            limitations.append(
                "No supporting graph path was found within the configured hop limit; this reflects graph "
                "coverage, not innocence."
            )

        explanation = self._analyst_text(
            row=row,
            y_score=float(y_score),
            y_pred=int(y_pred),
            features=contributing_features,
            entities=entities,
            paths=paths,
            relationships=relationships,
            record_id=record_id,
            attack_category=attack_category,
        )

        return {
            "alert_id": None if record_id is None else f"alert-{int(record_id)}",
            "record_id": None if record_id is None else int(record_id),
            "predicted_class": "attack" if int(y_pred) == 1 else "benign",
            "threat_score": round(float(y_score), 6),
            "decision_threshold": self.detector.threshold,
            "decision_rule": f"threat_score >= {self.detector.threshold}",
            "model": self.detector.name,
            "detector_uses_graph_features": self.detector.uses_graph_features,
            "ground_truth_label": None if true_label is None else ("attack" if int(true_label) == 1 else "benign"),
            "ground_truth_category": None if attack_category is None else str(attack_category),
            "evidence_source": {
                "dataset": self.dataset_name,
                "dataset_origin": self.dataset_origin,
                "file": None if source_file is None or (isinstance(source_file, float) and np.isnan(source_file)) else str(source_file),
                "record_id": None if record_id is None else int(record_id),
                "timestamp": None if timestamp is None or pd.isna(timestamp) else str(timestamp),
                "label_source": "dataset label (used for evaluation only, never as a model input)",
            },
            "important_features": contributing_features,
            "attribution_method": {"name": self.method, "note": self.method_note},
            "connected_entities": entities,
            "relationships": relationships,
            "supporting_graph_paths": paths,
            "limitations": limitations,
            "analyst_explanation": explanation,
            "disclaimer": (
                "This explanation does not prove malicious intent. It documents which observed features "
                "and which recorded relationships the model used, and which data the model was trained on."
            ),
        }

    # ------------------------------------------------------------------
    def _analyst_text(
        self,
        row: pd.Series,
        y_score: float,
        y_pred: int,
        features: list[dict[str, Any]],
        entities: list[dict[str, Any]],
        paths: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        record_id: Any,
        attack_category: Any,
    ) -> str:
        sentences: list[str] = []
        label = "malicious" if y_pred == 1 else "benign"
        sentences.append(
            f"Record {record_id if record_id is not None else 'n/a'} was classified as {label} with a "
            f"threat score of {y_score:.3f} (threshold {self.detector.threshold:.2f}) by {self.detector.name}."
        )

        endpoints = ", ".join(
            f"{entity['role']} {entity['value']}"
            + (" (internal)" if entity.get("is_internal") else " (external)")
            for entity in entities
            if entity.get("type") == "ip"
        )
        if endpoints:
            sentences.append(f"Involved endpoints: {endpoints}.")

        top = features[:3]
        if top:
            described = "; ".join(
                f"{item['feature']} = {item['value']} ({item['direction']}, attribution {item['attribution']:+.4f})"
                for item in top
            )
            sentences.append(f"The strongest model-level contributors were: {described}.")

        intel_notes = []
        for entity in entities:
            evidence = entity.get("threat_intel_evidence")
            if evidence and evidence.get("n_malicious_records"):
                intel_notes.append(
                    f"{entity['value']} appeared in {evidence['n_malicious_records']} training-labelled "
                    f"malicious record(s) out of {evidence['n_total_records']} "
                    f"(smoothed confidence {evidence['confidence']})"
                )
        if intel_notes:
            sentences.append("Threat-intelligence context: " + "; ".join(intel_notes) + ".")

        path_notes = []
        for path in paths:
            if path.get("available") and path.get("n_hops", 0) > 0:
                chain = " -> ".join(path["entities"])
                path_notes.append(f"{path['role']} {path['address']} reaches intel-flagged activity in {path['n_hops']} hop(s): {chain}")
        if path_notes:
            sentences.append("Graph path evidence (proximity context, not proof): " + "; ".join(path_notes) + ".")
        elif paths:
            sentences.append(
                "No observed graph path within the hop limit connects these endpoints to an IP with "
                "training-labelled malicious activity."
            )

        if relationships:
            relations = ", ".join(
                f"{item['relation']} ({item['confidence']:.2f} confidence)" for item in relationships[:5]
            )
            sentences.append(f"Recorded relationships touching these entities: {relations}.")

        if attack_category is not None and not pd.isna(attack_category):
            sentences.append(
                f"For reference, the dataset label for this record is '{attack_category}'. Labels are never "
                "model inputs; this is shown only to help you judge the alert."
            )

        sentences.append(
            "Interpretation: this is a statistical detection with the evidence above. It does not prove "
            "malicious intent, and no relationship in this report should be read as evidence of intent."
        )
        return " ".join(sentences)

    # ------------------------------------------------------------------
    def explain_top_alerts(
        self,
        X: pd.DataFrame,
        y_score: np.ndarray,
        y_pred: np.ndarray,
        rows: pd.DataFrame,
        y_true: np.ndarray | None = None,
        n_alerts: int | None = None,
        training_reference: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        """Explain the highest-scoring positive decisions."""
        n = int(n_alerts or self.settings.explain.n_alerts)
        positive_positions = np.where(np.asarray(y_pred) == 1)[0]
        if len(positive_positions) == 0:
            logger.info("no positive decisions to explain")
            return []
        order = positive_positions[np.argsort(-np.asarray(y_score)[positive_positions])][:n]

        attributions = self.attribute_features(X.loc[order])
        alerts: list[dict[str, Any]] = []
        for position in order:
            row = rows.loc[position]
            alerts.append(
                self.explain_row(
                    row=row,
                    y_score=float(y_score[position]),
                    y_pred=int(y_pred[position]),
                    attributions=attributions.loc[position],
                    true_label=None if y_true is None else int(y_true[position]),
                )
            )
        logger.info("generated %d alert explanations", len(alerts))
        return alerts

    def _percentile_of(self, feature: str, value: Any) -> float | None:
        """Share of training rows at or below this value, computed on the training split."""
        if self._reference_values is None or feature not in self._reference_values.columns:
            return None
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        column = self._reference_values[feature].dropna()
        if column.empty:
            return None
        try:
            return float((column <= float(value)).mean())
        except (TypeError, ValueError):
            return None
