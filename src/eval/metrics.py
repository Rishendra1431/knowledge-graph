"""Classification metrics and a statistically explicit model comparison.

Nothing here is estimated or hard-coded: every number is computed from the
supplied label/prediction arrays. Degenerate cases (a resample or a test split
that contains only one class) return ``None`` with a recorded reason instead of
a fabricated score.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn import metrics as sk_metrics

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc")
POSITIVE_LABEL = 1


def _safe(function: Callable[..., float], *args: Any, **kwargs: Any) -> float | None:
    try:
        value = float(function(*args, **kwargs))
    except Exception:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_score: np.ndarray | list[float] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Binary classification metrics for the 'is_attack' task.

    ``precision``/``recall``/``f1`` refer to the positive (attack) class, which is
    the class an alerting system is judged on. Macro/weighted variants are
    included so the report is not cherry-picked.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}")

    n_samples = int(len(y_true))
    n_positive = int((y_true == POSITIVE_LABEL).sum())
    n_negative = int((y_true != POSITIVE_LABEL).sum())
    notes: list[str] = []

    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    result: dict[str, Any] = {
        "n_samples": n_samples,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_rate": round(n_positive / n_samples, 6) if n_samples else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_labels": ["benign(0)", "attack(1)"],
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "accuracy": _safe(sk_metrics.accuracy_score, y_true, y_pred),
        "precision": _safe(sk_metrics.precision_score, y_true, y_pred, pos_label=1, zero_division=0),
        "recall": _safe(sk_metrics.recall_score, y_true, y_pred, pos_label=1, zero_division=0),
        "f1": _safe(sk_metrics.f1_score, y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": round(tn / (tn + fp), 6) if (tn + fp) else None,
        "balanced_accuracy": _safe(sk_metrics.balanced_accuracy_score, y_true, y_pred),
        "mcc": _safe(sk_metrics.matthews_corrcoef, y_true, y_pred),
        "precision_macro": _safe(sk_metrics.precision_score, y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": _safe(sk_metrics.recall_score, y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": _safe(sk_metrics.f1_score, y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": _safe(sk_metrics.f1_score, y_true, y_pred, average="weighted", zero_division=0),
        "false_positive_rate": round(fp / (fp + tn), 6) if (fp + tn) else None,
        "false_negative_rate": round(fn / (fn + tp), 6) if (fn + tp) else None,
        "threshold": threshold,
    }

    if y_score is not None:
        score = np.asarray(y_score, dtype="float64")
        if score.size != y_true.size:
            raise ValueError("y_score must align with y_true")
        if len(np.unique(y_true)) < 2:
            result["roc_auc"] = None
            result["average_precision"] = None
            notes.append("ROC-AUC/PR-AUC undefined: only one class present in this evaluation set.")
        else:
            result["roc_auc"] = _safe(sk_metrics.roc_auc_score, y_true, score)
            result["average_precision"] = _safe(sk_metrics.average_precision_score, y_true, score)
    else:
        result["roc_auc"] = None
        result["average_precision"] = None

    report = sk_metrics.classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["benign", "attack"],
        output_dict=True, zero_division=0,
    )
    result["per_class"] = {
        str(label): {
            "precision": round(float(values["precision"]), 6),
            "recall": round(float(values["recall"]), 6),
            "f1_score": round(float(values["f1-score"]), 6),
            "support": int(values["support"]),
        }
        for label, values in report.items()
        if isinstance(values, dict)
    }
    result["notes"] = notes
    return result


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """Exact McNemar test on the discordant pairs of two detectors."""
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    # n01: A wrong, B right. n10: A right, B wrong.
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    total = n01 + n10

    payload: dict[str, Any] = {
        "n_a_correct_b_wrong": n10,
        "n_a_wrong_b_correct": n01,
        "n_discordant": total,
        "n_concordant": int(len(y_true) - total),
    }
    if total == 0:
        payload.update({"statistic": None, "p_value": None,
                        "interpretation": "The two detectors agree on every test record; no test is possible."})
        return payload
    try:
        from scipy.stats import binomtest

        test = binomtest(min(n01, n10), total, 0.5, alternative="two-sided")
        payload.update(
            {
                "statistic": float(min(n01, n10)),
                "p_value": float(test.pvalue),
                "method": "exact binomial McNemar test (two-sided)",
                "interpretation": (
                    "p < alpha: the two detectors make a different number of errors"
                    if test.pvalue < 0.05 else
                    "p >= alpha: no detectable difference in the number of errors"
                ),
            }
        )
    except Exception as exc:  # scipy unavailable
        payload.update({"statistic": None, "p_value": None, "method": f"unavailable ({exc})"})
    return payload


def paired_bootstrap_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    score_a: np.ndarray | None = None,
    score_b: np.ndarray | None = None,
    metric: str = "f1",
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap CI for ``metric(B) - metric(A)`` on the *same* resampled rows.

    Pairing matters: it removes test-set difficulty from the comparison and lets
    the difference be read as "B minus A on identical data".
    """
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)
    n = len(y_true)
    if n == 0:
        return {"metric": metric, "available": False, "reason": "empty test set"}

    def _metric_at(y: np.ndarray, pred: np.ndarray, scores: np.ndarray | None) -> float:
        if metric == "f1":
            return float(sk_metrics.f1_score(y, pred, pos_label=1, zero_division=0))
        if metric == "recall":
            return float(sk_metrics.recall_score(y, pred, pos_label=1, zero_division=0))
        if metric == "precision":
            return float(sk_metrics.precision_score(y, pred, pos_label=1, zero_division=0))
        if metric == "accuracy":
            return float(sk_metrics.accuracy_score(y, pred))
        if metric == "roc_auc" and scores is not None:
            if len(np.unique(y)) < 2:
                return float("nan")
            return float(sk_metrics.roc_auc_score(y, scores))
        raise ValueError(f"unsupported metric for bootstrap: {metric}")

    observed_a = _metric_at(y_true, pred_a, score_a)
    observed_b = _metric_at(y_true, pred_b, score_b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(int(n_boot), dtype="float64")
    failed = 0
    for index in range(int(n_boot)):
        sample = rng.integers(0, n, size=n)
        value_a = _metric_at(y_true[sample], pred_a[sample], None if score_a is None else np.asarray(score_a)[sample])
        value_b = _metric_at(y_true[sample], pred_b[sample], None if score_b is None else np.asarray(score_b)[sample])
        delta = value_b - value_a
        if delta != delta:
            failed += 1
            delta = 0.0
        deltas[index] = delta

    alpha = 1.0 - float(confidence)
    lower = float(np.quantile(deltas, alpha / 2))
    upper = float(np.quantile(deltas, 1 - alpha / 2))
    share_below = float(np.mean(deltas <= 0))
    share_above = float(np.mean(deltas >= 0))
    p_value = float(min(1.0, 2 * min(share_below, share_above)))

    return {
        "metric": metric,
        "available": True,
        "n_bootstrap": int(n_boot),
        "confidence": float(confidence),
        "seed": int(seed),
        "metric_a": round(observed_a, 6),
        "metric_b": round(observed_b, 6),
        "observed_difference_b_minus_a": round(observed_b - observed_a, 6),
        "bootstrap_mean_difference": round(float(deltas.mean()), 6),
        "bootstrap_std": round(float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0, 6),
        "ci_lower": round(lower, 6),
        "ci_upper": round(upper, 6),
        "p_value_two_sided": round(p_value, 6),
        "share_resamples_difference_leq_zero": round(share_below, 6),
        "degenerate_resamples": int(failed),
        "method": "paired bootstrap over test records (same resample for both detectors)",
    }


def auto_claim(
    comparison: dict[str, Any],
    model_a: str,
    model_b: str,
    metric: str,
    is_synthetic: bool,
    ci_lower_key: str = "ci_lower",
) -> dict[str, Any]:
    """Derive the comparison statement from the measured CI - never hand-written."""
    if not comparison.get("available"):
        return {
            "supported": False,
            "claim": "not evaluated",
            "reason": comparison.get("reason", "comparison unavailable"),
        }
    lower = comparison.get(ci_lower_key)
    upper = comparison.get("ci_upper")
    delta = comparison.get("observed_difference_b_minus_a")
    confidence = comparison.get("confidence", 0.95)
    if lower is None or upper is None or delta is None:
        return {"supported": False, "claim": "not evaluated", "reason": "incomplete bootstrap output"}

    if lower > 0:
        verdict = f"{model_b} outperforms {model_a} on test-set {metric}"
        supported = True
    elif upper < 0:
        verdict = f"{model_a} outperforms {model_b} on test-set {metric}"
        supported = True
    else:
        verdict = f"no statistically detectable difference between {model_a} and {model_b} on {metric}"
        supported = False

    qualifier = (
        " Evidence from a synthetic dataset: this measures the generator, not real-world detection."
        if is_synthetic else ""
    )
    return {
        "supported": supported,
        "claim": (
            f"{verdict}; observed difference (B-A) = {delta:+.4f} with a "
            f"{int(confidence * 100)}% paired-bootstrap CI [{lower:+.4f}, {upper:+.4f}].{qualifier}"
        ),
        "rule": (
            f"Claim is emitted only when the {int(confidence * 100)}% CI for the paired difference "
            "excludes zero. Otherwise the result is reported as 'no detectable difference'."
        ),
        "delta": delta,
        "ci_lower": lower,
        "ci_upper": upper,
    }
