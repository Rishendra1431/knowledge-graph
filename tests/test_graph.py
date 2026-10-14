"""Knowledge graph tests: entities, provenance, confidence rules, features."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.data.acquire import generate_synthetic_dataset
from src.data.preprocess import run_preprocessing
from src.graph.builder import (
    RELATION_TYPES,
    aggregated_confidence,
    build_intel_index,
    build_knowledge_graph,
    intel_confidence,
)
from src.graph.features import FEATURE_DESCRIPTIONS, GraphFeatureExtractor
from src.graph.paths import build_ip_graph, describe_path, paths_to_intel


@pytest.fixture(scope="module")
def graph_inputs():
    settings = Settings()
    settings.seed = 5
    settings.dataset.synthetic_rows = 1200
    settings.graph.persist_subgraph_nodes = 200
    settings.ensure_directories()
    frame, _ = generate_synthetic_dataset(settings)
    result = run_preprocessing(frame, settings, persist=False)
    train, test = result.train.reset_index(drop=True), result.test.reset_index(drop=True)
    return train, test, settings, result


def test_confidence_rules_are_documented_and_bounded():
    # aggregated: 1 - exp(-n/K)
    assert aggregated_confidence(0, 10.0) == 0.0
    values = [aggregated_confidence(n, 10.0) for n in (1, 5, 10, 50, 200, 5000)]
    assert all(0.0 <= value <= 1.0 for value in values)
    assert all(b >= a for a, b in zip(values, values[1:])), "confidence must be monotone in evidence"
    assert values[0] > 0

    # intel: Laplace-smoothed malicious ratio, never exactly 0 or 1
    assert intel_confidence(0, 0) == pytest.approx(0.5)
    assert intel_confidence(5, 5) < 1.0
    assert intel_confidence(0, 1000) > 0.0
    assert 0.0 < intel_confidence(1, 100) < 1.0


def test_knowledge_graph_entities_relationships_and_provenance(graph_inputs):
    train, test, settings, _ = graph_inputs
    combined = pd.concat([train, test], ignore_index=True)
    intel = build_intel_index(train, settings)
    kg = build_knowledge_graph(combined, settings, include_labels=False, intel_index=intel)

    counts = kg.node_type_counts()
    assert counts.get("ip", 0) > 0
    assert counts.get("service", 0) > 0
    assert counts.get("protocol", 0) > 0
    assert counts.get("event", 0) > 0

    relations = kg.relation_counts()
    assert relations.get("COMMUNICATES_WITH", 0) > 0
    assert set(relations) <= set(RELATION_TYPES)

    # Provenance must resolve for every node and edge.
    for _, data in kg.graph.nodes(data=True):
        assert kg.provenance_of(data["provenance_id"]) is not None
    for _, _, data in kg.graph.edges(data=True):
        entry = kg.provenance_of(data["provenance_id"])
        assert entry is not None
        assert entry["dataset"]
        assert data["confidence_rule"]

    for _, _, data in kg.graph.edges(data=True):
        assert 0.0 <= float(data["confidence"]) <= 1.0
        assert int(data["evidence_count"]) >= 1

    summary = kg.summary()
    assert summary["n_nodes"] == kg.graph.number_of_nodes()
    assert summary["confidence"]["mean"] is not None
    node_link = kg.to_node_link(max_nodes=50)
    assert len(node_link["nodes"]) <= 50
    assert node_link["n_nodes_total"] == kg.graph.number_of_nodes()


def test_label_based_technique_edges_are_opt_in(graph_inputs):
    train, _, settings, _ = graph_inputs
    without = build_knowledge_graph(train, settings, include_labels=False)
    with_labels = build_knowledge_graph(train, settings, include_labels=True)
    assert "MATCHES_TECHNIQUE" not in without.relation_counts()
    assert with_labels.relation_counts().get("MATCHES_TECHNIQUE", 0) > 0


def test_ip_graph_and_path_search(graph_inputs):
    train, test, settings, _ = graph_inputs
    combined = pd.concat([train, test], ignore_index=True)
    graph, meta = build_ip_graph(combined, settings)
    assert meta["available"] is True
    assert graph.number_of_nodes() > 0
    assert all(0.0 < data["confidence"] <= 1.0 for _, _, data in graph.edges(data=True))

    intel = build_intel_index(train, settings)
    flagged = set(intel["flagged_ips"])
    assert flagged, "the synthetic training split must contain malicious IPs"
    paths = paths_to_intel(graph, flagged, max_hops=3)
    assert paths, "expected at least the flagged IPs themselves to be reachable"
    for node, entry in paths.items():
        if entry["hops"] > 0:
            assert entry["path"][0] in flagged
            assert entry["path"][-1] == node
            described = describe_path(graph, entry["path"], intel)
            assert described["available"] is True
            assert described["n_hops"] == entry["hops"]
            assert "not evidence of" in described["interpretation"] or "not proof" in described["interpretation"] \
                or "context" in described["interpretation"]


def test_graph_feature_extractor_columns_and_sentinels(graph_inputs):
    train, test, settings, _ = graph_inputs
    intel = build_intel_index(train, settings)
    extractor = GraphFeatureExtractor(settings, structural_scope="full").fit(
        pd.concat([train, test], ignore_index=True), intel
    )
    features = extractor.transform(test)

    assert set(features.columns) == set(FEATURE_DESCRIPTIONS)
    assert len(extractor.feature_names_) == len(FEATURE_DESCRIPTIONS)
    hop_columns = [name for name in features.columns if "hops_to_intel" in name]
    non_hop = features.drop(columns=hop_columns)
    assert not non_hop.isna().any().any(), "structural features must fall back to documented sentinels"
    assert np.isfinite(non_hop.to_numpy(dtype="float64")).all()

    report = extractor.report()
    assert report["available"] is True
    assert report["supervised_features"], "the supervised graph features must be declared"
    assert all(FEATURE_DESCRIPTIONS[name]["uses_training_labels"] for name in report["supervised_features"])


def test_inductive_scope_degrades_gracefully(graph_inputs):
    train, test, settings, _ = graph_inputs
    intel = build_intel_index(train, settings)
    extractor = GraphFeatureExtractor(settings, structural_scope="train_only").fit(train, intel)
    features = extractor.transform(test)
    assert list(features.columns) == extractor.feature_names_
    assert np.isfinite(features.to_numpy(dtype="float64")).sum() >= 0  # NaN hops allowed
    assert extractor.report()["structural_scope"] == "train_only"


def test_supervised_features_follow_the_training_labels_only(graph_inputs):
    train, test, settings, _ = graph_inputs
    intel = build_intel_index(train, settings)
    extractor = GraphFeatureExtractor(settings, structural_scope="full").fit(
        pd.concat([train, test], ignore_index=True), intel
    )
    before = extractor.transform(test)

    # 1. Changing test labels must not change any feature.
    flipped = test.copy()
    flipped[settings.preprocess.target_column] = 1 - flipped[settings.preprocess.target_column]
    after = extractor.transform(flipped)
    pd.testing.assert_frame_equal(before, after)

    # 2. Changing the training-derived intel index must change the intel features.
    target_ip = test["Source IP"].value_counts().index[0]
    augmented = copy.deepcopy(intel)
    augmented["ips"][target_ip] = {
        "n_total_records": 10, "n_malicious_records": 9, "confidence": 0.9, "is_internal": False,
    }
    augmented["flagged_ips"] = sorted(set(intel["flagged_ips"]) | {target_ip})
    changed = GraphFeatureExtractor(settings, structural_scope="full").fit(
        pd.concat([train, test], ignore_index=True), augmented
    ).transform(test)

    mask = test["Source IP"].astype(str) == str(target_ip)
    assert mask.any()
    assert np.allclose(changed.loc[mask, "g_src_intel_ratio"], 0.9)
    assert not np.allclose(
        before.loc[mask, "g_src_intel_ratio"].to_numpy(),
        changed.loc[mask, "g_src_intel_ratio"].to_numpy(),
    )
    assert np.allclose(changed.loc[mask, "g_src_is_flagged_intel"], 1.0)
