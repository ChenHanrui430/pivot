from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def topk_metrics(y_true: np.ndarray, logits: np.ndarray, ks: Iterable[int] = (1, 2, 3, 5, 10)) -> dict[str, Any]:
    order = np.argsort(logits, axis=1)
    result: dict[str, Any] = {}
    for k in ks:
        k_eff = min(int(k), logits.shape[1])
        if k_eff == 1:
            result[f"Top-{k}"] = round(float((logits.argmax(axis=1) == y_true).mean() * 100.0), 2)
            continue
        topk = order[:, -k_eff:]
        result[f"Top-{k}"] = round(float(np.any(topk == y_true[:, None], axis=1).mean() * 100.0), 2)
    return result


def summarize_logits(y_true: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
    pred = logits.argmax(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(shifted)
    prob = prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
    order = np.argsort(logits, axis=1)
    true_rank = logits.shape[1] - np.argsort(order, axis=1)[np.arange(logits.shape[0]), y_true]
    top1 = order[:, -1]
    top2 = order[:, -2] if logits.shape[1] > 1 else top1
    margins = logits[np.arange(logits.shape[0]), top1] - logits[np.arange(logits.shape[0]), top2]
    result: dict[str, Any] = {
        "Accuracy": round(float(accuracy_score(y_true, pred) * 100.0), 2),
        "Precision": round(float(precision_score(y_true, pred, average="macro", zero_division=0) * 100.0), 2),
        "Recall": round(float(recall_score(y_true, pred, average="macro", zero_division=0) * 100.0), 2),
        "F1-score": round(float(f1_score(y_true, pred, average="macro", zero_division=0) * 100.0), 2),
        "True-confidence": round(float(prob[np.arange(logits.shape[0]), y_true].mean()), 4),
        "Mean-confidence": round(float(prob[np.arange(logits.shape[0]), top1].mean()), 4),
        "Mean-margin": round(float(margins.mean()), 4),
        "Mean-true-rank": round(float(true_rank.mean()), 2),
        "Median-true-rank": round(float(np.median(true_rank)), 2),
    }
    result.update(topk_metrics(y_true, logits))
    return result



def _percent_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100.0, 2)


def _binary_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 0.0 if denom <= 0.0 else 2.0 * precision * recall / denom


def _conditional_known_summary(
    y_true: np.ndarray,
    logits: np.ndarray,
    known_mask: np.ndarray,
    accepted_known_mask: np.ndarray,
    unknown_label: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "Conditional-num_samples": int(accepted_known_mask.sum()),
        "Conditional-Coverage": _percent_or_none(float(accepted_known_mask.sum()) / max(float(known_mask.sum()), 1.0)),
    }
    if not accepted_known_mask.any():
        for key in ("Accuracy", "Precision", "Recall", "F1-score", "Top-1", "Top-2", "Top-3", "Top-5", "Top-10"):
            result[f"Conditional-{key}"] = None
        return result

    class_indices = [index for index in range(logits.shape[1]) if index != int(unknown_label)]
    remap = {old: new for new, old in enumerate(class_indices)}
    conditional_labels = np.asarray([remap[int(label)] for label in y_true[accepted_known_mask]], dtype=np.int64)
    conditional_logits = logits[accepted_known_mask][:, class_indices]
    summary = summarize_logits(conditional_labels, conditional_logits)
    for key in ("Accuracy", "Precision", "Recall", "F1-score", "Top-1", "Top-2", "Top-3", "Top-5", "Top-10"):
        if key in summary:
            result[f"Conditional-{key}"] = summary[key]
    return result


def summarize_open_world_logits(
    y_true: np.ndarray,
    logits: np.ndarray,
    *,
    unknown_label: int,
    r_values: Iterable[int] = (20,),
) -> dict[str, Any]:
    """Add open-world detection, conditional classification and r-precision metrics.

    r-precision follows the website-fingerprinting convention used by Holmes:
    rP_r = R_TP / (R_TP + R_WP + r * R_FP), where R_FP is the unmonitored-to-monitored false positive rate.
    """

    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    logits = np.nan_to_num(np.asarray(logits), nan=0.0, posinf=1e6, neginf=-1e6)
    pred = logits.argmax(axis=1)
    unknown = int(unknown_label)
    true_unknown = y_true == unknown
    true_known = ~true_unknown
    pred_unknown = pred == unknown
    pred_known = ~pred_unknown

    known_count = int(true_known.sum())
    unknown_count = int(true_unknown.sum())
    correct_known = true_known & pred_known & (pred == y_true)
    wrong_monitored = true_known & pred_known & (pred != y_true)
    rejected_known = true_known & pred_unknown
    false_monitored = true_unknown & pred_known

    known_accuracy = float(correct_known.sum()) / known_count if known_count else 0.0
    known_wrong_monitored = float(wrong_monitored.sum()) / known_count if known_count else 0.0
    known_rejection = float(rejected_known.sum()) / known_count if known_count else 0.0
    unknown_recall = float((true_unknown & pred_unknown).sum()) / unknown_count if unknown_count else 0.0
    unmonitored_fpr = float(false_monitored.sum()) / unknown_count if unknown_count else 0.0
    detection_tpr = float((true_known & pred_known).sum()) / known_count if known_count else 0.0
    predicted_known_count = int(pred_known.sum())
    predicted_unknown_count = int(pred_unknown.sum())
    detection_precision = float((true_known & pred_known).sum()) / predicted_known_count if predicted_known_count else 0.0
    detection_recall = detection_tpr
    unknown_precision = float((true_unknown & pred_unknown).sum()) / predicted_unknown_count if predicted_unknown_count else 0.0

    result: dict[str, Any] = {
        "unknown_label": unknown,
        "known_num_samples": known_count,
        "unknown_num_samples": unknown_count,
        "Known-Accuracy": _percent_or_none(known_accuracy),
        "Known-Wrong-Monitored-Rate": _percent_or_none(known_wrong_monitored),
        "Known-Rejection-Rate": _percent_or_none(known_rejection),
        "Monitored-Recall": _percent_or_none(known_accuracy),
        "Unknown-Recall": _percent_or_none(unknown_recall),
        "Unknown-Precision": _percent_or_none(unknown_precision),
        "Unmonitored-FPR": _percent_or_none(unmonitored_fpr),
        "Detection-TPR": _percent_or_none(detection_tpr),
        "Detection-FPR": _percent_or_none(unmonitored_fpr),
        "Detection-Precision": _percent_or_none(detection_precision),
        "Detection-Recall": _percent_or_none(detection_recall),
        "Detection-F1": _percent_or_none(_binary_f1(detection_precision, detection_recall)),
        # Backward-compatible alias; previous code used this name for known rejection, which was misleading.
        "Open-World-FPR": _percent_or_none(unmonitored_fpr),
    }
    for r in r_values:
        r_int = int(r)
        denominator = known_accuracy + known_wrong_monitored + float(r_int) * unmonitored_fpr
        result[f"rP{r_int}"] = _percent_or_none(known_accuracy / denominator) if denominator > 0.0 else 0.0
        result[f"rP{r_int}-R_TP"] = _percent_or_none(known_accuracy)
        result[f"rP{r_int}-R_WP"] = _percent_or_none(known_wrong_monitored)
        result[f"rP{r_int}-R_FP"] = _percent_or_none(unmonitored_fpr)
    result.update(_conditional_known_summary(y_true, logits, true_known, true_known & pred_known, unknown))
    return result
