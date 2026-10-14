"""Explainer tests: completeness, traceability and honest wording."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.data.acquire import generate_synthetic_dataset
from src.data.preprocess import run_preprocessing
from src.explain.explainer import METHOD_APPROX, METHOD_SHAP, AlertExplainer
from src.graph.builder import build_intel_index, build_knowledge_graph
from src.graph.features import GraphFeatureExtractor
from src.models.detectors import build_graph_aware


@pytest.fixture(scope="module")
def explained():
    settings = Settings()
    settings.seed = 17
    settings.dataset.synthetic_rows = 1500
    settings.model.n_estimators = 15
    settings.model.n_jobs = 1
    settings.explain.n_alerts = 4
    settings.graph.persist_subgraph_nodes = 200
    settings.ensure_directories()

    frame, provenance = generate_synthetic_dataset(settings)
    pre = run_preprocessing(frame, settings, persist=False)
    train, test = pre.train.reset_index(drop=True), pre.test.reset_index(drop=True)
    y_train = train[pre.target_column].to_numpy(dtype=int)
    y_test = test[pre.target_column].to_numpy(dtype=int)

    combined = pd.concat([train, test], ignore_index=True)
    intel = build_intel_index(train, settings)
    knowledge_graph = build_knowledge_graph(combined, settings, include_labels=False, intel_index=intel)
    extractor = GraphFeatureExtractor(settings, structural_scope="full").fit(combined, intel)

    graph_train = extractor.transform(train)
    graph_test = extractor.transform(test)
    X_train = pd.concat([train[pre.feature_columns], graph_train], axis=1)
    X_test = pd.concat([test[pre.feature_columns], graph_test], axis=1)

    detector = build_graph_aware(settings, pre.feature_columns, list(graph_train.columns))
    detector.fit(X_train, y_train)
    prediction = detector.predict_proba(X_test)

    explainer = AlertExplainer(
        detector=detector, settings=settings, graph_extractor=extractor,
        knowledge_graph=knowledge_graph, intel_index=intel,
        dataset_name=provenance.name, dataset_origin=provenance.origin,
        feature_reference=X_train,
    )
    alerts = explainer.explain_top_alerts(
        X=X_test, y_score=prediction.y_score, y_pred=prediction.y_pred,
        rows=test, y_true=y_test, n_alerts=settings.explain.n_alerts,
    )
    return alerts, test, prediction, explainer, provenance


def test_alerts_were_produced(explained):
    alerts, _, prediction, _, _ = explained
    assert (prediction.y_pred == 1).sum() > 0, "the fixture must produce positive decisions"
    assert alerts, "at least one alert explanation is required"


def test_alert_contains_every_required_element(explained):
    alerts, _, _, _, _ = explained
    required = {
        "alert_id", "record_id", "predicted_class", "threat_score", "decision_threshold",
        "important_features", "attribution_method", "connected_entities", "relationships",
        "supporting_graph_paths", "evidence_source", "limitations", "analyst_explanation",
        "disclaimer", "ground_truth_label",
    }
    for alert in alerts:
        assert required <= set(alert)
        assert alert["important_features"], "each alert needs contributing features"
        assert alert["analyst_explanation"]
        assert alert["evidence_source"]["dataset"]
        assert alert["evidence_source"]["record_id"] is not None
        assert alert["limitations"]


def test_threat_score_matches_the_model_output(explained):
    alerts, test, prediction, _, _ = explained
    position_by_record = {int(rid): index for index, rid in enumerate(test["__record_id"])}
    for alert in alerts:
        index = position_by_record[int(alert["record_id"])]
        assert alert["threat_score"] == pytest.approx(float(prediction.y_score[index]), abs=1e-6)
        assert alert["threat_score"] >= alert["decision_threshold"]


def test_attributions_are_labelled_with_their_method(explained):
    alerts, _, _, explainer, _ = explained
    assert explainer.method in (METHOD_SHAP, METHOD_APPROX)
    for alert in alerts:
        assert alert["attribution_method"]["name"] == explainer.method
        for feature in alert["important_features"]:
            assert feature["method"] == explainer.method
            assert feature["direction"] in ("pushes toward attack", "pushes toward benign")
            assert feature["description"], "every feature must carry a documented meaning"
    if explainer.method == METHOD_APPROX:
        assert any("APPROXIMATION" in limitation for limitation in alerts[0]["limitations"])


def test_supervised_features_are_disclosed(explained):
    alerts, _, _, _, _ = explained
    for alert in alerts:
        supervised = [f["feature"] for f in alert["important_features"] if f["uses_training_labels"]]
        if supervised:
            assert any("training-split labels" in limitation for limitation in alert["limitations"]), \
                "hindsight-derived features must be disclosed in the limitations"


def test_explanation_never_claims_proof_of_intent(explained):
    alerts, _, _, _, _ = explained
    for alert in alerts:
        text = (alert["analyst_explanation"] + " " + alert["disclaimer"]).lower()
        assert "does not prove malicious intent" in text
        assert "proof" in text or "not evidence" in text
        joined = " ".join(alert["limitations"]).lower()
        assert "not proof of malicious intent" in joined
        for path in alert["supporting_graph_paths"]:
            if path.get("available"):
                assert "context" in path["interpretation"].lower() or "not proof" in path["interpretation"].lower()


def test_graph_evidence_is_traceable(explained):
    alerts, _, _, _, _ = explained
    for alert in alerts:
        for relationship in alert["relationships"]:
            assert relationship["relation"]
            assert relationship["confidence_rule"], "a confidence value without its rule is not acceptable"
            assert relationship["provenance"] is not None

        for path in alert["supporting_graph_paths"]:
            if not path.get("available"):
                assert path.get("reason")
                continue
            for hop in path.get("hops", []):
                if hop.get("edge_confidence") is not None:
                    assert 0.0 <= hop["edge_confidence"] <= 1.0
                    assert hop["confidence_rule"]


def test_synthetic_origin_is_disclosed_in_every_explanation(explained):
    alerts, _, _, _, provenance = explained
    assert provenance.is_synthetic
    for alert in alerts:
        assert any("SYNTHETIC" in limitation.upper() for limitation in alert["limitations"])
        assert alert["evidence_source"]["dataset_origin"] == "synthetic"


def test_shap_request_falls_back_cleanly(explained):
    alerts, _, _, explainer, _ = explained
    settings = Settings()
    settings.explain.method = "shap"
    fallback = AlertExplainer(
        detector=explainer.detector, settings=settings,
        feature_reference=pd.DataFrame(np.zeros((4, len(explainer.detector.feature_columns))),
                                       columns=explainer.detector.feature_columns),
    )
    assert fallback.method in (METHOD_SHAP, METHOD_APPROX)
    assert fallback.method_note
