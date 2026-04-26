from __future__ import annotations

import pandas as pd
import pytest

from adult_lab.models import ID3DecisionTree, RandomForestID3, entropy, information_gain
from adult_lab.models.tree import LABEL_COLUMN


def test_entropy_and_information_gain_on_toy_dataset() -> None:
    frame = pd.DataFrame(
        {
            "outlook": ["sunny", "sunny", "rain", "rain"],
            LABEL_COLUMN: [0, 0, 1, 1],
        }
    )
    assert entropy([0, 0, 1, 1]) == pytest.approx(1.0)
    assert information_gain(frame, "outlook") == pytest.approx(1.0)


def test_decision_tree_fits_toy_dataset_exactly() -> None:
    features = pd.DataFrame(
        {
            "outlook": ["sunny", "sunny", "rain", "rain"],
            "windy": ["false", "true", "false", "true"],
        }
    )
    labels = [0, 0, 1, 1]
    model = ID3DecisionTree(max_depth=3, min_samples_split=2, random_state=7)
    model.fit(features, labels)

    assert model.predict(features) == labels


def test_random_forest_is_reproducible_for_same_seed() -> None:
    features = pd.DataFrame(
        {
            "outlook": ["sunny", "sunny", "rain", "rain", "sunny", "rain"],
            "windy": ["false", "true", "false", "true", "false", "true"],
            "humidity": ["high", "high", "normal", "normal", "normal", "high"],
        }
    )
    labels = [0, 0, 1, 1, 0, 1]
    forest_a = RandomForestID3(n_estimators=5, max_features=2, max_depth=4, random_state=11)
    forest_b = RandomForestID3(n_estimators=5, max_features=2, max_depth=4, random_state=11)

    forest_a.fit(features, labels)
    forest_b.fit(features, labels)

    assert forest_a.predict_proba(features) == forest_b.predict_proba(features)
