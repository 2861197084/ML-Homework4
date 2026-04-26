from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from adult_lab.models.tree import ID3DecisionTree


class RandomForestID3:
    def __init__(
        self,
        *,
        n_estimators: int,
        max_features: int,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_gain: float = 1e-9,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.random_state = random_state
        self.feature_names: list[str] = []
        self.trees: list[ID3DecisionTree] = []
        self._rng = np.random.default_rng(random_state)

    def fit(self, features: pd.DataFrame, labels: Iterable[int]) -> "RandomForestID3":
        features_frame = features.reset_index(drop=True)
        labels_series = pd.Series(list(labels), name="income").reset_index(drop=True)
        self.feature_names = [column for column in features_frame.columns]
        self.trees = []
        sample_count = len(features_frame)

        for _ in range(self.n_estimators):
            bootstrap_indices = self._rng.integers(0, sample_count, size=sample_count)
            tree_seed = int(self._rng.integers(0, 1_000_000_000))
            tree = ID3DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_gain=self.min_gain,
                feature_subsample_size=min(self.max_features, len(self.feature_names)),
                random_state=tree_seed,
            )
            tree.fit(
                features_frame.iloc[bootstrap_indices].reset_index(drop=True),
                labels_series.iloc[bootstrap_indices].reset_index(drop=True),
            )
            self.trees.append(tree)
        return self

    def _require_trees(self) -> list[ID3DecisionTree]:
        if not self.trees:
            raise ValueError("The random forest has not been fitted yet.")
        return self.trees

    def predict_proba(self, features: pd.DataFrame) -> list[float]:
        trees = self._require_trees()
        tree_probabilities = np.asarray([tree.predict_proba(features) for tree in trees], dtype=float)
        return [float(value) for value in tree_probabilities.mean(axis=0)]

    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> list[int]:
        probabilities = self.predict_proba(features)
        return [1 if probability >= threshold else 0 for probability in probabilities]

    def feature_usage_counts(self) -> dict[str, int]:
        trees = self._require_trees()
        counter: Counter[str] = Counter()
        for tree in trees:
            counter.update(tree.feature_usage_counts())
        return dict(counter)

    def to_dict(self) -> dict[str, object]:
        self._require_trees()
        return {
            "model_type": "random_forest_id3",
            "params": {
                "n_estimators": self.n_estimators,
                "max_features": self.max_features,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_gain": self.min_gain,
                "random_state": self.random_state,
            },
            "feature_names": self.feature_names,
            "trees": [tree.to_dict() for tree in self.trees],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RandomForestID3":
        params = dict(payload["params"])
        model = cls(
            n_estimators=int(params["n_estimators"]),
            max_features=int(params["max_features"]),
            max_depth=int(params["max_depth"]) if params.get("max_depth") is not None else None,
            min_samples_split=int(params["min_samples_split"]),
            min_gain=float(params["min_gain"]),
            random_state=int(params["random_state"]) if params.get("random_state") is not None else None,
        )
        model.feature_names = [str(value) for value in payload.get("feature_names", [])]
        model.trees = [ID3DecisionTree.from_dict(tree_payload) for tree_payload in payload["trees"]]
        return model
