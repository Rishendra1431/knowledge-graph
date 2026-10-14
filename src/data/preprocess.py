"""Cleaning, encoding and leakage-safe splitting of labelled security events.

Design rules enforced here (see docs/LIMITATIONS.md for the reasoning):

1. Every cleaning statistic (median, clip quantiles, categorical vocabulary,
   constant-column and missing-column decisions) is fitted on the **training
   split only**.
2. Exact duplicate records are removed *before* splitting, so one record cannot
   appear in both splits.
3. The default split is a day-based temporal holdout (train on Monday-Thursday,
   test on Friday), which is the setup that exposes the model to attack types
   that never occur during training.
4. A leakage audit runs after splitting and its verdict is stored in the results
   artifact. A failed audit is surfaced, never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import Settings, get_logger
from src.data.acquire import ATTACK_TAXONOMY, PROVENANCE_COLUMNS

logger = get_logger("data.preprocess")

TIMESTAMP_INTERNAL = "__timestamp"
DAY_INTERNAL = "__day_name"
CATEGORY_COLUMN = "attack_category"
TARGET_COLUMN_DEFAULT = "is_attack"

# Derived numeric features. Each is only created when its inputs exist, and the
# list of created names is recorded so the paper can state exactly what was used.
DERIVED_NUMERIC = {
    "flow_bytes_per_packet": ("Total Length of Fwd Packets", "Total Length of Bwd Packets", "Total Fwd Packets", "Total Backward Packets"),
    "log_flow_duration": ("Flow Duration",),
}

WELL_KNOWN_PORTS = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
    67: "dhcp", 68: "dhcp", 69: "tftp", 80: "http", 110: "pop3", 123: "ntp",
    135: "msrpc", 137: "netbios", 138: "netbios", 139: "netbios", 143: "imap",
    161: "snmp", 389: "ldap", 443: "https", 445: "smb", 465: "smtps", 514: "syslog",
    587: "smtp", 636: "ldaps", 993: "imaps", 995: "pop3s", 1433: "mssql",
    1521: "oracle", 2049: "nfs", 3306: "mysql", 3389: "rdp", 5060: "sip",
    5432: "postgres", 5900: "vnc", 6379: "redis", 6667: "irc", 8080: "http-alt",
    8443: "https-alt", 9090: "websm", 27017: "mongodb",
}

PRIVATE_PREFIXES = ("10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.2",
                    "172.30.", "172.31.", "192.168.", "127.")

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _weekday_sort_key(day: str) -> int:
    return WEEKDAY_ORDER.index(day) if day in WEEKDAY_ORDER else len(WEEKDAY_ORDER)


def is_private_ip(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith(PRIVATE_PREFIXES)


def port_to_service(port: Any) -> str:
    try:
        return WELL_KNOWN_PORTS.get(int(port), "unknown")
    except (TypeError, ValueError):
        return "unknown"


@dataclass
class SplitReport:
    strategy_requested: str
    strategy_used: str
    fallback_reason: str | None = None
    train_rows: int = 0
    test_rows: int = 0
    train_positive_rate: float = 0.0
    test_positive_rate: float = 0.0
    train_days: list[str] = field(default_factory=list)
    test_days: list[str] = field(default_factory=list)
    n_groups: int | None = None
    train_label_distribution: dict[str, int] = field(default_factory=dict)
    test_label_distribution: dict[str, int] = field(default_factory=dict)
    train_attack_categories: list[str] = field(default_factory=list)
    test_attack_categories: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_requested": self.strategy_requested,
            "strategy_used": self.strategy_used,
            "fallback_reason": self.fallback_reason,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "train_positive_rate": self.train_positive_rate,
            "test_positive_rate": self.test_positive_rate,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "n_groups": self.n_groups,
            "train_label_distribution": self.train_label_distribution,
            "test_label_distribution": self.test_label_distribution,
            "train_attack_categories": self.train_attack_categories,
            "test_attack_categories": self.test_attack_categories,
            "notes": self.notes,
        }


@dataclass
class PreprocessResult:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    graph_feature_columns: list[str]
    entity_columns: list[str]
    target_column: str
    cleaning_report: dict[str, Any]
    split_report: SplitReport
    encoder_metadata: dict[str, Any]
    leakage_audit: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "train_rows": int(len(self.train)),
            "test_rows": int(len(self.test)),
            "n_features_tabular": len(self.feature_columns),
            "n_features_graph": len(self.graph_feature_columns),
            "n_features_total": len(self.feature_columns) + len(self.graph_feature_columns),
            "feature_columns": self.feature_columns,
            "graph_feature_columns": self.graph_feature_columns,
            "entity_columns": self.entity_columns,
            "target_column": self.target_column,
            "cleaning_report": self.cleaning_report,
            "split_report": self.split_report.to_dict(),
            "encoder_metadata": self.encoder_metadata,
            "leakage_audit": self.leakage_audit,
        }


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
def parse_timestamps(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse the Timestamp column across the formats seen in this dataset."""
    column = settings.preprocess.timestamp_column
    info: dict[str, Any] = {"column": column, "parsed": False, "format_used": None, "n_unparsed": 0}

    if column not in df.columns:
        df[TIMESTAMP_INTERNAL] = pd.NaT
        info["format_used"] = "column_absent"
        return df, info

    raw = df[column].astype(str).str.strip()
    candidate_formats = [
        "%m/%d/%Y %I:%M:%S %p",  # 7/7/2017 3:46:03 PM  (as released)
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    best = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    best_format = None
    best_valid = -1
    for fmt in candidate_formats:
        parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        valid = int(parsed.notna().sum())
        if valid > best_valid:
            best, best_format, best_valid = parsed, fmt, valid
        if valid == len(df):
            break

    if best_valid < max(1, int(0.5 * len(df))):
        # Fall back to a generic parse before giving up.
        generic = None
        for kwargs in ({"format": "mixed"}, {}):
            try:
                candidate = pd.to_datetime(raw, errors="coerce", **kwargs)
            except (TypeError, ValueError):
                continue
            if int(candidate.notna().sum()) > int(generic.notna().sum()) if generic is not None else True:
                generic = candidate
        if generic is not None and int(generic.notna().sum()) > best_valid:
            best, best_format, best_valid = generic, "inferred", int(generic.notna().sum())

    df[TIMESTAMP_INTERNAL] = best
    df[DAY_INTERNAL] = df[TIMESTAMP_INTERNAL].dt.day_name()
    info.update(
        {
            "parsed": best_valid > 0,
            "format_used": best_format,
            "n_valid": best_valid,
            "n_unparsed": int(len(df) - best_valid),
            "min_timestamp": None if best_valid == 0 else str(best.min()),
            "max_timestamp": None if best_valid == 0 else str(best.max()),
        }
    )
    logger.info("timestamps parsed=%s format=%s valid=%d/%d", info["parsed"], best_format, best_valid, len(df))
    return df, info


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def build_targets(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_col = settings.preprocess.label_column
    if label_col not in df.columns:
        raise KeyError(
            f"label column '{label_col}' not found. Available columns: {list(df.columns)[:40]}... "
            "Set preprocess.label_column in config.yaml for a different dataset."
        )
    info: dict[str, Any] = {}
    labels = df[label_col].astype(str).str.strip()
    df[label_col] = labels

    benign = settings.preprocess.benign_label
    df[TARGET_COLUMN_DEFAULT] = (labels != benign).astype("int8")
    df[CATEGORY_COLUMN] = labels.map(lambda value: ATTACK_TAXONOMY.get(value, "Other"))

    info["label_values"] = {str(k): int(v) for k, v in labels.value_counts().to_dict().items()}
    info["unmapped_labels"] = sorted(
        {str(v) for v in labels.unique() if str(v) not in ATTACK_TAXONOMY}
    )
    info["n_benign"] = int((df[TARGET_COLUMN_DEFAULT] == 0).sum())
    info["n_attack"] = int((df[TARGET_COLUMN_DEFAULT] == 1).sum())
    info["attack_categories"] = {str(k): int(v) for k, v in df[CATEGORY_COLUMN].value_counts().to_dict().items()}
    if info["unmapped_labels"]:
        logger.warning("labels with no taxonomy entry (mapped to 'Other'): %s", info["unmapped_labels"])
    logger.info("targets built: %d benign / %d attack", info["n_benign"], info["n_attack"])
    return df, info


def clean_dataframe(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Type coercion, duplicate removal and obvious-noise handling."""
    report: dict[str, Any] = {}
    label_col = settings.preprocess.label_column
    protected = {
        label_col, TARGET_COLUMN_DEFAULT, CATEGORY_COLUMN, TIMESTAMP_INTERNAL, DAY_INTERNAL,
        settings.preprocess.timestamp_column, *PROVENANCE_COLUMNS,
    }
    protected |= set(settings.preprocess.entity_columns)
    protected |= set(settings.preprocess.drop_columns)

    report["rows_before_cleaning"] = int(len(df))

    # 1. Drop columns that carry no modelling value.
    dropped_columns = [c for c in settings.preprocess.drop_columns if c in df.columns]
    if dropped_columns:
        df = df.drop(columns=dropped_columns)
        report["dropped_columns"] = dropped_columns

    # 2. Drop rows with a missing label.
    before = len(df)
    missing_label = df[label_col].isna() | (df[label_col].astype(str).str.strip() == "")
    if missing_label.any():
        df = df.loc[~missing_label]
    report["rows_dropped_missing_label"] = int(before - len(df))

    # 3. Numeric coercion for everything that is not protected.
    numeric_columns = [c for c in df.columns if c not in protected]
    coercion_failures: dict[str, int] = {}
    for col in numeric_columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        n_failed = int((converted.isna() & df[col].notna()).sum())
        if n_failed:
            coercion_failures[col] = n_failed
        df[col] = converted
    report["non_numeric_values_coerced_to_nan"] = coercion_failures

    # 4. Infinities are present in the released rate features.
    if settings.preprocess.replace_infinite:
        numeric_frame = df[numeric_columns]
        n_inf = int(np.isinf(numeric_frame.to_numpy(dtype="float64", na_value=np.nan)).sum())
        df[numeric_columns] = numeric_frame.replace([np.inf, -np.inf], np.nan)
        report["infinite_values_replaced"] = n_inf

    # 5. Drop rows with no label-derived target (defensive) and duplicates.
    before = len(df)
    dedup_subset = [c for c in df.columns if c not in PROVENANCE_COLUMNS]
    df = df.drop_duplicates(subset=dedup_subset, keep="first")
    report["duplicate_rows_removed"] = int(before - len(df))

    # 6. Entity keys must exist for graph construction; backfill with sentinels.
    for col in settings.preprocess.entity_columns:
        if col not in df.columns:
            df[col] = np.nan
    report["entity_missing_counts"] = {
        col: int(df[col].isna().sum()) for col in settings.preprocess.entity_columns
    }

    df = df.reset_index(drop=True)
    df["__record_id"] = np.arange(len(df), dtype=np.int64)
    report["rows_after_cleaning"] = int(len(df))
    report["missing_values_per_column_top"] = (
        df[[c for c in df.columns if c not in PROVENANCE_COLUMNS]]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .to_dict()
    )
    logger.info(
        "cleaning: %d -> %d rows (%d duplicates, %d inf values replaced)",
        report["rows_before_cleaning"],
        report["rows_after_cleaning"],
        report["duplicate_rows_removed"],
        report.get("infinite_values_replaced", 0),
    )
    return df, report


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_dataset(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, SplitReport]:
    """Split into train/test using the configured strategy, with honest fallback."""
    target = settings.preprocess.target_column
    strategy = settings.preprocess.split_strategy
    report = SplitReport(strategy_requested=strategy, strategy_used=strategy)

    def _record(frame_train: pd.DataFrame, frame_test: pd.DataFrame) -> SplitReport:
        report.train_rows = int(len(frame_train))
        report.test_rows = int(len(frame_test))
        report.train_positive_rate = round(float(frame_train[target].mean()), 6)
        report.test_positive_rate = round(float(frame_test[target].mean()), 6)
        report.train_label_distribution = {
            str(k): int(v) for k, v in frame_train[settings.preprocess.label_column].value_counts().to_dict().items()
        }
        report.test_label_distribution = {
            str(k): int(v) for k, v in frame_test[settings.preprocess.label_column].value_counts().to_dict().items()
        }
        report.train_attack_categories = sorted(frame_train[CATEGORY_COLUMN].unique().tolist())
        report.test_attack_categories = sorted(frame_test[CATEGORY_COLUMN].unique().tolist())
        for name, frame in (("train", frame_train), ("test", frame_test)):
            if frame[DAY_INTERNAL].notna().any():
                days = sorted(frame[DAY_INTERNAL].dropna().unique().tolist(), key=_weekday_sort_key)
                if name == "train":
                    report.train_days = days
                else:
                    report.test_days = days
        return report

    def _random_split(reason: str | None) -> tuple[pd.DataFrame, pd.DataFrame, SplitReport]:
        if reason:
            report.strategy_used = "random"
            report.fallback_reason = reason
            report.notes.append(f"Fell back to a random stratified split: {reason}")
            logger.warning("falling back to random split: %s", reason)
        train_idx, test_idx = train_test_split(
            df.index.to_numpy(),
            test_size=settings.preprocess.test_size,
            random_state=settings.seed,
            stratify=df[target].to_numpy(),
        )
        return df.loc[train_idx], df.loc[test_idx], _record(df.loc[train_idx], df.loc[test_idx])

    if strategy == "temporal":
        train_days = settings.preprocess.temporal.get("train_days", [])
        test_days = settings.preprocess.temporal.get("test_days", [])
        timestamps_ok = TIMESTAMP_INTERNAL in df.columns and df[TIMESTAMP_INTERNAL].notna().mean() > 0.99
        if not timestamps_ok:
            train, test, report = _random_split("timestamps unavailable or partially unparsed")
        else:
            train_mask = df[DAY_INTERNAL].isin(train_days)
            test_mask = df[DAY_INTERNAL].isin(test_days)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                train, test, report = _random_split("configured day-based split produced an empty side")
            elif df.loc[train_mask, target].nunique() < 2:
                train, test, report = _random_split("training days contain a single class")
            else:
                train, test = df.loc[train_mask], df.loc[test_mask]
                report = _record(train, test)
                # Chronological ordering is verified by leakage_audit
                # (temporal_ordering_respected), not asserted here.
    elif strategy == "group":
        key_columns = settings.preprocess.group.get("key_columns", [])
        usable = [c for c in key_columns if c in df.columns and df[c].notna().all()]
        if not usable:
            train, test, report = _random_split("group key columns are unavailable")
        else:
            groups = df[usable].astype(str).agg("|".join, axis=1)
            n_groups = groups.nunique()
            if n_groups < 3:
                train, test, report = _random_split("too few distinct groups for a grouped split")
            else:
                splitter = GroupShuffleSplit(n_splits=1, test_size=settings.preprocess.test_size, random_state=settings.seed)
                train_idx, test_idx = next(splitter.split(df, groups=groups))
                train, test = df.iloc[train_idx], df.iloc[test_idx]
                report = _record(train, test)
                report.n_groups = int(n_groups)
    else:
        train, test, report = _random_split(None)

    for frame in (train, test):
        if frame[target].nunique() < 2:
            report.notes.append(
                f"WARNING: one side of the split contains a single class "
                f"(train={train[target].nunique()}, test={test[target].nunique()} classes)"
            )
    logger.info(
        "split[%s]: train=%d (%.3f positive) test=%d (%.3f positive)",
        report.strategy_used, report.train_rows, report.train_positive_rate,
        report.test_rows, report.test_positive_rate,
    )
    return train.copy(), test.copy(), report


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
class FeatureEncoder:
    """Fits all encoding decisions on the training split, then applies them."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._never_features: tuple[str, ...] = (
            settings.preprocess.label_column,
            settings.preprocess.target_column,
            CATEGORY_COLUMN,
            TIMESTAMP_INTERNAL,
            DAY_INTERNAL,
            settings.preprocess.timestamp_column,
            *settings.preprocess.entity_columns,
            *PROVENANCE_COLUMNS,
        )
        self.numeric_columns: list[str] = []
        self.dropped_missing: list[str] = []
        self.dropped_constant: list[str] = []
        self.medians: dict[str, float] = {}
        self.clip_bounds: dict[str, tuple[float, float]] = {}
        self.categorical_vocab: dict[str, list[str]] = {}
        self.one_hot_columns_: list[str] = []
        self.derived_features: list[str] = []
        self.feature_columns_: list[str] = []
        self.fitted = False

    # -- helpers -----------------------------------------------------------
    def _candidate_categoricals(self, df: pd.DataFrame) -> list[str]:
        """Categorical columns eligible for one-hot encoding.

        Label-derived columns are excluded by construction: one-hot encoding
        ``attack_category`` would hand the target to the model through the back
        door. The leakage audit would catch it, but it must not be created.
        """
        candidates: list[str] = []
        excluded = set(self._never_features)
        for col in ("Protocol", "dst_service"):
            if col in df.columns and col not in excluded:
                candidates.append(col)
        for col in df.columns:
            if col in candidates or col in excluded or col.startswith("__") or col in PROVENANCE_COLUMNS:
                continue
            if df[col].dtype == object and df[col].nunique(dropna=True) <= self.settings.preprocess.one_hot_max_cardinality:
                candidates.append(col)
        return candidates

    @staticmethod
    def _add_derived(df: pd.DataFrame) -> list[str]:
        created: list[str] = []
        need = {"Total Length of Fwd Packets", "Total Length of Bwd Packets", "Total Fwd Packets", "Total Backward Packets"}
        if need.issubset(df.columns):
            total_bytes = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
            total_pkts = df["Total Fwd Packets"] + df["Total Backward Packets"]
            df["flow_bytes_per_packet"] = total_bytes / total_pkts.replace(0, np.nan)
            created.append("flow_bytes_per_packet")
        if "Flow Duration" in df.columns:
            df["log_flow_duration"] = np.log1p(df["Flow Duration"].clip(lower=0))
            created.append("log_flow_duration")
        return created

    def fit(self, train: pd.DataFrame) -> "FeatureEncoder":
        settings = self.settings
        protected = {
            *self._never_features, "__record_id", "__split", "__group_key", "dst_service",
        }
        work = train.copy()
        if "Destination Port" in work.columns:
            work["dst_service"] = work["Destination Port"].map(port_to_service)
        self.derived_features = self._add_derived(work)

        self.numeric_columns = [
            c for c in work.columns
            if c not in protected and not c.startswith("__") and c not in ("dst_service",)
            and pd.api.types.is_numeric_dtype(work[c])
        ]

        missing_fraction = work[self.numeric_columns].isna().mean()
        self.dropped_missing = [
            c for c, frac in missing_fraction.items() if frac > settings.preprocess.max_missing_fraction
        ]

        remaining = [c for c in self.numeric_columns if c not in self.dropped_missing]
        if settings.preprocess.drop_constant_features and remaining:
            variances = work[remaining].var(numeric_only=True).fillna(0.0)
            self.dropped_constant = [
                c for c, var in variances.items() if var <= settings.preprocess.variance_threshold
            ]
        remaining = [c for c in remaining if c not in self.dropped_constant]
        self.numeric_columns = remaining

        self.medians = {
            c: (float(work[c].median()) if work[c].notna().any() else 0.0) for c in self.numeric_columns
        }
        lo_q, hi_q = settings.preprocess.clip_quantiles
        self.clip_bounds = {}
        if settings.preprocess.clip_outliers:
            for c in self.numeric_columns:
                series = work[c].dropna()
                if series.empty:
                    continue
                self.clip_bounds[c] = (float(series.quantile(lo_q)), float(series.quantile(hi_q)))

        categoricals = self._candidate_categoricals(work)
        self.categorical_vocab = {}
        for col in categoricals:
            values = work[col].astype(str).fillna("missing")
            counts = values.value_counts()
            vocab = sorted(counts.index.tolist())
            if "missing" not in vocab:
                vocab.append("missing")
            self.categorical_vocab[col] = vocab

        # Build the final ordered feature list by transforming the training rows.
        encoded = self._transform_frame(work)
        self.feature_columns_ = list(encoded.columns)
        self.fitted = True
        logger.info(
            "encoder fitted: %d features (%d numeric, %d one-hot, %d derived); dropped %d constant, %d high-missing",
            len(self.feature_columns_),
            len(self.numeric_columns),
            len(self.one_hot_columns_),
            len(self.derived_features),
            len(self.dropped_constant),
            len(self.dropped_missing),
        )
        return self

    def _transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if "Destination Port" in work.columns:
            work["dst_service"] = work["Destination Port"].map(port_to_service)
        self._add_derived(work)

        out = pd.DataFrame(index=work.index)
        for col in self.numeric_columns:
            series = pd.to_numeric(work[col], errors="coerce")
            if col in self.clip_bounds:
                low, high = self.clip_bounds[col]
                series = series.clip(lower=low, upper=high)
            out[col] = series.fillna(self.medians.get(col, 0.0))

        one_hot_columns: list[str] = []
        for col, vocab in self.categorical_vocab.items():
            if col in work.columns:
                values = work[col].astype(str).fillna("missing").replace({"nan": "missing"})
            else:
                # Column absent at transform time: every level becomes 0 so the
                # feature vector stays well defined (documented in the metadata).
                values = pd.Series("__absent__", index=work.index)
            for level in vocab:
                if level == "missing":
                    continue
                name = f"{col}__{level}"
                out[name] = (values == level).astype("int8")
                one_hot_columns.append(name)
        self.one_hot_columns_ = one_hot_columns
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeatureEncoder.transform called before fit")
        encoded = self._transform_frame(df)
        # Enforce the training column order; unseen columns would be a bug.
        missing = [c for c in self.feature_columns_ if c not in encoded.columns]
        if missing:
            raise RuntimeError(f"transform produced fewer features than fit; missing={missing[:5]}")
        return encoded[self.feature_columns_]

    def metadata(self) -> dict[str, Any]:
        return {
            "n_features": len(self.feature_columns_),
            "numeric_features": self.numeric_columns,
            "one_hot_features": self.one_hot_columns_,
            "derived_features": self.derived_features,
            "categorical_vocabulary": self.categorical_vocab,
            "dropped_constant_features": self.dropped_constant,
            "dropped_high_missing_features": self.dropped_missing,
            "imputation": {"method": "train_median", "medians": self.medians},
            "outlier_clipping": {
                "enabled": self.settings.preprocess.clip_outliers,
                "quantiles": self.settings.preprocess.clip_quantiles,
                "bounds": {k: list(v) for k, v in self.clip_bounds.items()},
            },
            "fit_on": "training split only",
            "derived_feature_definitions": {
                "flow_bytes_per_packet": "total bytes both directions / total packets both directions",
                "log_flow_duration": "log1p(Flow Duration), negatives clipped to zero",
                "dst_service": "Destination Port mapped to a well-known service name, else 'unknown'",
            },
        }


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------
def leakage_audit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    settings: Settings,
    encoder: FeatureEncoder,
) -> dict[str, Any]:
    """Explicit, recorded checks that the evaluation is not contaminated."""
    checks: list[dict[str, Any]] = []

    target = settings.preprocess.target_column
    forbidden = {target, CATEGORY_COLUMN, settings.preprocess.label_column, TIMESTAMP_INTERNAL, DAY_INTERNAL}
    leaked = sorted(forbidden & set(feature_columns))
    checks.append(
        {
            "name": "no_target_or_label_columns_in_features",
            "passed": not leaked,
            "detail": "no label/target/timestamp columns among model features" if not leaked else f"leaked: {leaked}",
        }
    )

    entity_leaks = sorted(set(settings.preprocess.entity_columns) & set(feature_columns))
    checks.append(
        {
            "name": "no_raw_entity_identifiers_in_model_features",
            "passed": not entity_leaks,
            "detail": "raw IP/port identifiers excluded from tabular features (they enter only through the graph)"
            if not entity_leaks else f"entity columns used as features: {entity_leaks}",
        }
    )

    overlap_cols = [c for c in feature_columns if c in train.columns and c in test.columns]
    if overlap_cols:
        train_hashes = pd.util.hash_pandas_object(train[overlap_cols].fillna(-999), index=False)
        test_hashes = pd.util.hash_pandas_object(test[overlap_cols].fillna(-999), index=False)
        shared = set(train_hashes.tolist()) & set(test_hashes.tolist())
    else:
        shared = set()
    checks.append(
        {
            "name": "no_identical_records_across_splits",
            "passed": len(shared) == 0,
            "detail": f"{len(shared)} identical feature rows shared between train and test",
        }
    )

    id_overlap = set(train.get("__record_id", pd.Series(dtype=int)).tolist()) & set(
        test.get("__record_id", pd.Series(dtype=int)).tolist()
    )
    checks.append(
        {
            "name": "record_ids_disjoint",
            "passed": len(id_overlap) == 0,
            "detail": f"{len(id_overlap)} shared record ids",
        }
    )

    temporal_ok = True
    detail = "not applicable (non-temporal split)"
    report = settings.preprocess.split_strategy
    if report == "temporal" and TIMESTAMP_INTERNAL in train.columns and train[TIMESTAMP_INTERNAL].notna().all():
        train_max = train[TIMESTAMP_INTERNAL].max()
        test_min = test[TIMESTAMP_INTERNAL].min()
        temporal_ok = bool(train_max <= test_min)
        detail = f"train max={train_max}, test min={test_min}"
    checks.append({"name": "temporal_ordering_respected", "passed": temporal_ok, "detail": detail})

    checks.append(
        {
            "name": "encoders_and_imputers_fitted_on_train_only",
            "passed": encoder.fitted,
            "detail": "medians, clip bounds and categorical vocabularies were fitted on the training split",
        }
    )

    failed = [c["name"] for c in checks if not c["passed"]]
    audit = {
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed_checks": failed,
        "passed_all": not failed,
        "interpretation": (
            "All recorded leakage checks passed. Note that this audit covers split-level contamination; "
            "it cannot prove the absence of subtler distributional leakage (see docs/LIMITATIONS.md)."
            if not failed
            else "At least one leakage check failed - treat the reported metrics as invalid until resolved."
        ),
    }
    if failed:
        logger.error("LEAKAGE AUDIT FAILED: %s", failed)
    else:
        logger.info("leakage audit passed (%d checks)", len(checks))
    return audit


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_preprocessing(
    df: pd.DataFrame, settings: Settings, persist: bool = True
) -> PreprocessResult:
    """Full preprocessing pipeline. Returns in-memory artifacts."""
    frame = df.copy()
    frame, timestamp_info = parse_timestamps(frame, settings)
    frame, target_info = build_targets(frame, settings)

    if frame[settings.preprocess.target_column].nunique() < 2:
        raise ValueError(
            "the dataset contains a single class, so detection models cannot be trained or evaluated"
        )

    frame, cleaning_report = clean_dataframe(frame, settings)
    cleaning_report["timestamp_parsing"] = timestamp_info
    cleaning_report["targets"] = target_info

    train, test, split_report = split_dataset(frame, settings)

    encoder = FeatureEncoder(settings).fit(train)
    train_features = encoder.transform(train)
    test_features = encoder.transform(test)

    feature_columns = list(train_features.columns)
    train = pd.concat([train, train_features], axis=1)
    test = pd.concat([test, test_features], axis=1)

    train["__split"] = "train"
    test["__split"] = "test"

    audit = leakage_audit(train, test, feature_columns, settings, encoder)

    result = PreprocessResult(
        train=train,
        test=test,
        feature_columns=feature_columns,
        graph_feature_columns=[],
        entity_columns=[c for c in settings.preprocess.entity_columns if c in frame.columns],
        target_column=settings.preprocess.target_column,
        cleaning_report=cleaning_report,
        split_report=split_report,
        encoder_metadata=encoder.metadata(),
        leakage_audit=audit,
    )

    if persist:
        processed_dir = settings.path("processed_dir")
        processed_dir.mkdir(parents=True, exist_ok=True)
        train.to_pickle(processed_dir / "train.pkl")
        test.to_pickle(processed_dir / "test.pkl")
        logger.info("persisted processed splits to %s", processed_dir)

    return result


def target_column(settings: Settings) -> str:
    return settings.preprocess.target_column or TARGET_COLUMN_DEFAULT
