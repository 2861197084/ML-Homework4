from __future__ import annotations

from typing import Iterable

import numpy as np


def confusion_matrix_binary(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, object]:
    y_true_array = np.asarray(list(y_true), dtype=int)
    y_pred_array = np.asarray(list(y_pred), dtype=int)
    if len(y_true_array) != len(y_pred_array):
        raise ValueError("y_true and y_pred must have the same length")

    tp = int(np.sum((y_true_array == 1) & (y_pred_array == 1)))
    tn = int(np.sum((y_true_array == 0) & (y_pred_array == 0)))
    fp = int(np.sum((y_true_array == 0) & (y_pred_array == 1)))
    fn = int(np.sum((y_true_array == 1) & (y_pred_array == 0)))
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "matrix": [[tn, fp], [fn, tp]],
    }


def accuracy_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    cm = confusion_matrix_binary(y_true, y_pred)
    total = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
    return (cm["tn"] + cm["tp"]) / total if total else 0.0


def precision_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    cm = confusion_matrix_binary(y_true, y_pred)
    denominator = cm["tp"] + cm["fp"]
    return cm["tp"] / denominator if denominator else 0.0


def recall_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    cm = confusion_matrix_binary(y_true, y_pred)
    denominator = cm["tp"] + cm["fn"]
    return cm["tp"] / denominator if denominator else 0.0


def f1_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    denominator = precision + recall
    return 2 * precision * recall / denominator if denominator else 0.0


def roc_curve(y_true: Iterable[int], y_score: Iterable[float]) -> dict[str, list[float]]:
    y_true_array = np.asarray(list(y_true), dtype=int)
    y_score_array = np.asarray(list(y_score), dtype=float)
    if len(y_true_array) != len(y_score_array):
        raise ValueError("y_true and y_score must have the same length")

    positive_count = int(np.sum(y_true_array == 1))
    negative_count = int(np.sum(y_true_array == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Both positive and negative labels are required for ROC/AUC")

    order = np.argsort(-y_score_array, kind="mergesort")
    y_true_sorted = y_true_array[order]
    y_score_sorted = y_score_array[order]

    distinct_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_indices = np.r_[distinct_indices, len(y_score_sorted) - 1]
    true_positives = np.cumsum(y_true_sorted)[threshold_indices]
    false_positives = (threshold_indices + 1) - true_positives

    true_positives = np.r_[0, true_positives]
    false_positives = np.r_[0, false_positives]
    thresholds = np.r_[np.inf, y_score_sorted[threshold_indices]]
    tpr = true_positives / positive_count
    fpr = false_positives / negative_count

    return {
        "fpr": [float(value) for value in fpr],
        "tpr": [float(value) for value in tpr],
        "thresholds": [float(value) for value in thresholds],
    }


def auc_from_roc(roc: dict[str, list[float]]) -> float:
    fpr = np.asarray(roc["fpr"], dtype=float)
    tpr = np.asarray(roc["tpr"], dtype=float)
    return float(np.trapezoid(tpr, fpr))


def evaluate_binary_classification(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, object]:
    y_true_list = [int(value) for value in y_true]
    y_score_list = [float(value) for value in y_score]
    y_pred = [1 if score >= threshold else 0 for score in y_score_list]

    roc = roc_curve(y_true_list, y_score_list)
    cm = confusion_matrix_binary(y_true_list, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true_list, y_pred)),
        "precision": float(precision_score(y_true_list, y_pred)),
        "recall": float(recall_score(y_true_list, y_pred)),
        "f1": float(f1_score(y_true_list, y_pred)),
        "auc": float(auc_from_roc(roc)),
        "threshold": float(threshold),
        "support": int(len(y_true_list)),
        "positive_rate": float(np.mean(np.asarray(y_true_list, dtype=float))),
        "predicted_positive_rate": float(np.mean(np.asarray(y_pred, dtype=float))),
        "confusion_matrix": cm,
        "roc": roc,
    }
