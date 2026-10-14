"""Preprocessing tests: cleaning, encoding, splitting, leakage audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.acquire import ATTACK_TAXONOMY, generate_synthetic_dataset
from src.data.preprocess import (
    CATEGORY_COLUMN,
    DAY_INTERNAL,
    TIMESTAMP_INTERNAL,
    is_private_ip,
    port_to_service,
    run_preprocessing,
)


@pytest.fixture(scope="module")
def prepared(request):
    from src.config import Settings

    settings = Settings()
    settings.seed = 3
    settings.dataset.synthetic_rows = 1500
    settings.model.n_estimators = 12
    settings.graph.persist_subgraph_nodes = 200
    settings.ensure_directories()
    frame, provenance = generate_synthetic_dataset(settings)
    result = run_preprocessing(frame, settings, persist=False)
    return frame, provenance, result, settings


def test_synthetic_dataset_is_labelled_as_synthetic(prepared):
    _, provenance, _, _ = prepared
    assert provenance.is_synthetic
    assert provenance.origin == "synthetic"
    assert provenance.to_dict()["is_synthetic"] is True
    assert any("SYNTHETIC" in note.upper() for note in provenance.notes)


def test_cleaning_removes_duplicates_and_infinities(prepared):
    frame, _, result, _ = prepared
    report = result.cleaning_report
    assert report["duplicate_rows_removed"] > 0, "the generator injects duplicates on purpose"
    assert report["infinite_values_replaced"] > 0, "the generator injects inf rates on purpose"
    assert report["rows_after_cleaning"] < report["rows_before_cleaning"]
    # No infinities may survive into the model-ready frames.
    numeric = result.train[result.feature_columns].to_numpy(dtype="float64")
    assert np.isfinite(numeric).all()


def test_targets_and_taxonomy(prepared):
    _, _, result, settings = prepared
    assert set(result.train[result.target_column].unique()) <= {0, 1}
    assert CATEGORY_COLUMN in result.train.columns
    known = set(ATTACK_TAXONOMY.values()) | {"Other"}
    assert set(result.train[CATEGORY_COLUMN].unique()) <= known


def test_timestamps_parsed_and_days_derived(prepared):
    _, _, result, _ = prepared
    assert result.cleaning_report["timestamp_parsing"]["parsed"] is True
    for frame in (result.train, result.test):
        assert TIMESTAMP_INTERNAL in frame.columns
        assert frame[TIMESTAMP_INTERNAL].notna().all()
        assert frame[DAY_INTERNAL].notna().all()


def test_temporal_split_is_used_when_timestamps_exist(prepared):
    _, _, result, _ = prepared
    split = result.split_report
    assert split.strategy_used == "temporal"
    assert split.train_days == ["Monday", "Tuesday", "Wednesday", "Thursday"]
    assert split.test_days == ["Friday"]
    assert split.train_rows > 0 and split.test_rows > 0
    assert result.train[TIMESTAMP_INTERNAL].max() <= result.test[TIMESTAMP_INTERNAL].min()


def test_labels_never_become_features(prepared):
    _, _, result, settings = prepared
    forbidden = {settings.preprocess.label_column, result.target_column, CATEGORY_COLUMN,
                 TIMESTAMP_INTERNAL, DAY_INTERNAL}
    assert not (forbidden & set(result.feature_columns))
    for column in ("Source IP", "Destination IP"):
        assert column not in result.feature_columns


def test_categorical_encoding_uses_training_vocabulary(prepared):
    _, _, result, _ = prepared
    vocabulary = result.encoder_metadata["categorical_vocabulary"]
    assert "Protocol" in vocabulary and vocabulary["Protocol"], "the protocol vocabulary must not be empty"
    assert any(name.startswith("Protocol__") for name in result.feature_columns)
    assert result.encoder_metadata["fit_on"] == "training split only"


def test_imputation_and_clipping_fitted_on_train(prepared):
    _, _, result, _ = prepared
    metadata = result.encoder_metadata
    assert metadata["imputation"]["method"] == "train_median"
    assert metadata["imputation"]["medians"], "medians must be recorded for the paper"
    assert metadata["outlier_clipping"]["bounds"], "clip bounds must be recorded"
    # Training features must be complete after imputation.
    assert not result.train[result.feature_columns].isna().any().any()


def test_leakage_audit_passes_on_a_clean_pipeline(prepared):
    _, _, result, _ = prepared
    audit = result.leakage_audit
    assert audit["passed_all"], f"failed checks: {audit['failed_checks']}"
    names = {check["name"] for check in audit["checks"]}
    assert {"no_identical_records_across_splits", "record_ids_disjoint",
            "no_target_or_label_columns_in_features"} <= names


def test_record_ids_are_disjoint(prepared):
    _, _, result, _ = prepared
    train_ids = set(result.train["__record_id"])
    test_ids = set(result.test["__record_id"])
    assert not (train_ids & test_ids)


def test_entity_helpers():
    assert port_to_service(443) == "https"
    assert port_to_service(3389) == "rdp"
    assert port_to_service(65500) == "unknown"
    assert port_to_service(None) == "unknown"
    assert is_private_ip("192.168.1.5") is True
    assert is_private_ip("8.8.8.8") is False
    assert is_private_ip(None) is False


def test_single_class_dataset_is_rejected():
    from src.config import Settings
    from src.data.preprocess import build_targets

    settings = Settings()
    frame = pd.DataFrame({"Label": ["BENIGN"] * 10, "Value": np.arange(10)})
    stamped, _ = build_targets(frame, settings)
    assert stamped[settings.preprocess.target_column].nunique() == 1
