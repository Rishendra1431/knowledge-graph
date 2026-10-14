"""Leakage tests: the project's core integrity guarantee.

Each test either (a) confirms that a documented split strategy keeps train and
test apart, or (b) injects contamination and asserts that the audit catches it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.data.acquire import generate_synthetic_dataset
from src.data.preprocess import (
    CATEGORY_COLUMN,
    FeatureEncoder,
    TIMESTAMP_INTERNAL,
    leakage_audit,
    run_preprocessing,
    split_dataset,
)


@pytest.fixture(scope="module")
def frame_and_settings():
    settings = Settings()
    settings.seed = 13
    settings.dataset.synthetic_rows = 1200
    settings.ensure_directories()
    frame, _ = generate_synthetic_dataset(settings)
    return frame, settings


def test_group_split_has_no_shared_conversations(frame_and_settings):
    frame, base = frame_and_settings
    settings = Settings()
    settings.seed = base.seed
    settings.preprocess.split_strategy = "group"
    settings.ensure_directories()

    result = run_preprocessing(frame, settings, persist=False)
    assert result.split_report.strategy_used in ("group", "random")
    if result.split_report.strategy_used == "group":
        train_pairs = set(zip(result.train["Source IP"], result.train["Destination IP"]))
        test_pairs = set(zip(result.test["Source IP"], result.test["Destination IP"]))
        assert not (train_pairs & test_pairs), "a conversation must not span both splits"


def test_temporal_split_is_chronological(frame_and_settings):
    frame, base = frame_and_settings
    settings = Settings()
    settings.seed = base.seed
    settings.ensure_directories()

    train, test, report = split_dataset(
        _prepared_frame(frame, settings), settings
    )
    if report.strategy_used == "temporal":
        assert train[TIMESTAMP_INTERNAL].max() <= test[TIMESTAMP_INTERNAL].min()


def test_temporal_split_falls_back_when_timestamps_are_missing(frame_and_settings):
    frame, base = frame_and_settings
    settings = Settings()
    settings.seed = base.seed
    settings.preprocess.split_strategy = "temporal"
    settings.ensure_directories()

    without_timestamps = frame.drop(columns=["Timestamp"])
    result = run_preprocessing(without_timestamps, settings, persist=False)
    assert result.split_report.strategy_used == "random"
    assert result.split_report.fallback_reason, "the reason for the fallback must be recorded"
    assert result.split_report.train_rows > 0 and result.split_report.test_rows > 0


def _prepared_frame(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    from src.data.preprocess import build_targets, clean_dataframe, parse_timestamps

    working = frame.copy()
    working, _ = parse_timestamps(working, settings)
    working, _ = build_targets(working, settings)
    working, _ = clean_dataframe(working, settings)
    return working


def test_leakage_audit_flags_an_injected_duplicate(frame_and_settings):
    frame, settings = frame_and_settings
    result = run_preprocessing(frame, settings, persist=False)
    assert result.leakage_audit["passed_all"], "baseline pipeline must be clean"

    train = result.train.copy()
    test = result.test.copy()
    duplicated = test.iloc[[0]].copy()
    contaminated_train = pd.concat([train, duplicated], ignore_index=True)

    encoder = FeatureEncoder(settings)
    encoder.fitted = True
    audit = leakage_audit(contaminated_train, test, result.feature_columns, settings, encoder)

    assert audit["passed_all"] is False
    assert "no_identical_records_across_splits" in audit["failed_checks"]
    assert audit["interpretation"].startswith("At least one leakage check failed")


def test_leakage_audit_flags_shared_record_ids(frame_and_settings):
    frame, settings = frame_and_settings
    result = run_preprocessing(frame, settings, persist=False)

    train = result.train.copy()
    test = result.test.copy()
    train["__record_id"] = 1  # collide with every test record id 1..n
    test["__record_id"] = 1

    encoder = FeatureEncoder(settings)
    encoder.fitted = True
    audit = leakage_audit(train, test, result.feature_columns, settings, encoder)
    assert "record_ids_disjoint" in audit["failed_checks"]


def test_leakage_audit_flags_label_columns_used_as_features(frame_and_settings):
    frame, settings = frame_and_settings
    result = run_preprocessing(frame, settings, persist=False)
    encoder = FeatureEncoder(settings)
    encoder.fitted = True

    contaminated_features = list(result.feature_columns) + [CATEGORY_COLUMN]
    audit = leakage_audit(result.train, result.test, contaminated_features, settings, encoder)
    assert "no_target_or_label_columns_in_features" in audit["failed_checks"]


def test_encoder_requires_fit_before_transform(frame_and_settings):
    frame, settings = frame_and_settings
    encoder = FeatureEncoder(settings)
    with pytest.raises(RuntimeError):
        encoder.transform(frame)


def test_encoder_handles_unseen_categories(frame_and_settings):
    frame, settings = frame_and_settings
    prepared = _prepared_frame(frame, settings)
    encoder = FeatureEncoder(settings).fit(prepared)
    unseen = prepared.copy()
    unseen["Protocol"] = 999  # never present during fitting
    transformed = encoder.transform(unseen)
    protocol_columns = [name for name in transformed.columns if name.startswith("Protocol__")]
    assert protocol_columns, "protocol one-hot columns must exist"
    assert (transformed[protocol_columns].sum(axis=1) == 0).all(), "unseen categories must be all-zero"
    assert np.isfinite(transformed.to_numpy(dtype="float64")).all()
