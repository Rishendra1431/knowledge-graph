"""Threat detectors.

Two configurations of the *same* algorithm share every hyperparameter and the
same random seed, so the only difference between them is the feature set:

* ``baseline_rf``    - RandomForest on the preprocessed flow features.
* ``graph_aware_rf`` - the same RandomForest on flow features **plus** the
  graph-derived features (structural + training-label-only relational features).

Using one algorithm for both arms keeps the comparison interpretable: a
difference cannot be attributed to a different model family.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import Settings, get_logger

logger = get_logger("models.detectors")


@dataclass
class PredictionResult:
    y_pred: np.ndarray
    y_score: np.ndarray
    threshold: float
    n_items: int
    inference_seconds: float
    batch_size: int | None = None
    batch_times: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def events_per_second(self) -> float | None:
        if self.inference_seconds <= 0:
            return None
        return self.n_items / self.inference_seconds

    @property
    def milliseconds_per_event(self) -> float | None:
        if self.n_items <= 0:
            return None
        return 1000.0 * self.inference_seconds / self.n_items

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "threshold": self.threshold,
            "inference_seconds": round(self.inference_seconds, 6),
            "events_per_second": None if self.events_per_second is None else round(self.events_per_second, 4),
            "milliseconds_per_event": None if self.milliseconds_per_event is None else round(self.milliseconds_per_event, 6),
            "batch_size": self.batch_size,
            "n_batches": len(self.batch_times),
            "batch_time_mean_seconds": round(float(np.mean(self.batch_times)), 6) if self.batch_times else None,
            "batch_time_std_seconds": round(float(np.std(self.batch_times, ddof=1)), 6) if len(self.batch_times) > 1 else 0.0,
            "notes": self.notes,
        }


class ThreatDetector:
    """Thin, auditable wrapper around ``RandomForestClassifier``."""

    def __init__(
        self,
        name: str,
        settings: Settings,
        feature_columns: list[str],
        description: str = "",
        uses_graph_features: bool = False,
    ) -> None:
        self.name = name
        self.settings = settings
        self.feature_columns = list(feature_columns)
        self.description = description
        self.uses_graph_features = uses_graph_features
        params = settings.model
        self.model = RandomForestClassifier(
            n_estimators=int(params.n_estimators),
            max_depth=params.max_depth,
            min_samples_leaf=int(params.min_samples_leaf),
            class_weight=params.class_weight,
            n_jobs=params.n_jobs,
            random_state=settings.seed,
        )
        self.threshold = float(params.decision_threshold)
        self.training_seconds: float | None = None
        self.n_training_rows: int | None = None
        self.notes: list[str] = []

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ThreatDetector":
        X = self._select(X)
        self.n_training_rows = int(len(X))
        start = time.perf_counter()
        self.model.fit(X.to_numpy(dtype="float64"), np.asarray(y).astype(int))
        self.training_seconds = time.perf_counter() - start
        classes = list(getattr(self.model, "classes_", []))
        if classes != [0, 1]:
            self.notes.append(
                f"model was fitted on classes {classes}; positive-class indexing assumes [0, 1]"
            )
        logger.info(
            "fitted %s on %d rows x %d features in %.3fs",
            self.name, self.n_training_rows, len(self.feature_columns), self.training_seconds,
        )
        return self

    # ------------------------------------------------------------------
    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise KeyError(f"{self.name}: missing feature columns {missing[:8]}")
        return X[self.feature_columns]

    def predict_proba(self, X: pd.DataFrame, batch_size: int | None = None) -> PredictionResult:
        """Batched inference. Batching bounds the memory of a single call and
        lets per-batch timing be reported instead of one opaque total."""
        frame = self._select(X)
        batch = int(batch_size or self.settings.evaluation.inference_batch_size)
        batch = max(batch, 1)
        n = len(frame)
        scores = np.empty(n, dtype="float64")
        batch_times: list[float] = []
        notes: list[str] = []

        start = time.perf_counter()
        if n == 0:
            notes.append("empty input: no inference performed")
        else:
            matrix = frame.to_numpy(dtype="float64")
            positive_index = self._positive_class_index()
            for offset in range(0, n, batch):
                chunk = matrix[offset: offset + batch]
                tick = time.perf_counter()
                probabilities = self.model.predict_proba(chunk)
                batch_times.append(time.perf_counter() - tick)
                scores[offset: offset + len(chunk)] = probabilities[:, positive_index]
        total = time.perf_counter() - start
        y_pred = (scores >= self.threshold).astype(int) if n else np.empty(0, dtype=int)

        return PredictionResult(
            y_pred=y_pred,
            y_score=scores,
            threshold=self.threshold,
            n_items=n,
            inference_seconds=total,
            batch_size=batch,
            batch_times=batch_times,
            notes=notes,
        )

    def _positive_class_index(self) -> int:
        classes = list(getattr(self.model, "classes_", [0, 1]))
        return classes.index(1) if 1 in classes else len(classes) - 1

    # ------------------------------------------------------------------
    def feature_importances(self) -> pd.Series:
        if not hasattr(self.model, "feature_importances_"):
            return pd.Series(dtype="float64")
        return pd.Series(
            self.model.feature_importances_, index=self.feature_columns, name="importance"
        ).sort_values(ascending=False)

    def top_features(self, k: int = 15) -> list[dict[str, Any]]:
        return [
            {"feature": name, "importance": round(float(value), 8)}
            for name, value in self.feature_importances().head(k).items()
        ]

    def metadata(self) -> dict[str, Any]:
        params = self.settings.model
        return {
            "name": self.name,
            "description": self.description,
            "algorithm": "sklearn.ensemble.RandomForestClassifier",
            "hyperparameters": {
                "n_estimators": int(params.n_estimators),
                "max_depth": params.max_depth,
                "min_samples_leaf": int(params.min_samples_leaf),
                "class_weight": params.class_weight,
                "random_state": int(self.settings.seed),
                "n_jobs": int(params.n_jobs),
            },
            "decision_threshold": self.threshold,
            "uses_graph_features": self.uses_graph_features,
            "n_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "n_training_rows": self.n_training_rows,
            "training_seconds": None if self.training_seconds is None else round(self.training_seconds, 6),
            "notes": self.notes,
        }


def build_baseline(settings: Settings, tabular_features: list[str]) -> ThreatDetector:
    return ThreatDetector(
        name="baseline_rf",
        settings=settings,
        feature_columns=list(tabular_features),
        description="RandomForest on preprocessed flow features only (no graph information).",
        uses_graph_features=False,
    )


def build_graph_aware(
    settings: Settings, tabular_features: list[str], graph_features: list[str]
) -> ThreatDetector:
    if not graph_features:
        raise ValueError(
            "graph-aware detector requested but no graph features are available; "
            "run with graph features enabled or evaluate the baseline only"
        )
    return ThreatDetector(
        name="graph_aware_rf",
        settings=settings,
        feature_columns=list(tabular_features) + list(graph_features),
        description=(
            "Same RandomForest, same hyperparameters and seed, on flow features plus "
            "graph-derived structural and training-label-only relational features."
        ),
        uses_graph_features=True,
    )
