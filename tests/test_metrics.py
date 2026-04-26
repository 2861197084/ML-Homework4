from __future__ import annotations

import pytest

from adult_lab.metrics import auc_from_roc, confusion_matrix_binary, evaluate_binary_classification, roc_curve


def test_roc_auc_is_one_for_perfect_ranking() -> None:
    roc = roc_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert auc_from_roc(roc) == pytest.approx(1.0)


def test_confusion_matrix_and_binary_metrics() -> None:
    metrics = evaluate_binary_classification([0, 0, 1, 1], [0.1, 0.7, 0.9, 0.2])
    confusion = confusion_matrix_binary([0, 0, 1, 1], [0, 1, 1, 0])

    assert confusion["matrix"] == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
