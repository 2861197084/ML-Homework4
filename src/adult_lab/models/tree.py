from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_COLUMN = "__label__"


def entropy(labels: Iterable[int]) -> float:
    label_array = np.asarray(list(labels), dtype=int)
    if label_array.size == 0:
        return 0.0
    counts = np.bincount(label_array, minlength=2)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def information_gain(frame: pd.DataFrame, feature_name: str) -> float:
    base_entropy = entropy(frame[LABEL_COLUMN])
    conditional_entropy = 0.0
    total = len(frame)
    for _, subset in frame.groupby(feature_name, sort=False):
        probability = len(subset) / total
        conditional_entropy += probability * entropy(subset[LABEL_COLUMN])
    return float(base_entropy - conditional_entropy)


@dataclass(slots=True)
class TreeNode:
    depth: int
    sample_count: int
    class_counts: dict[int, int]
    majority_class: int
    positive_prob: float
    feature_name: str | None = None
    split_gain: float | None = None
    children: dict[str, "TreeNode"] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.feature_name is None

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "sample_count": self.sample_count,
            "class_counts": {str(key): value for key, value in self.class_counts.items()},
            "majority_class": self.majority_class,
            "positive_prob": self.positive_prob,
            "feature_name": self.feature_name,
            "split_gain": self.split_gain,
            "children": {key: value.to_dict() for key, value in self.children.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TreeNode":
        children_payload = payload.get("children", {})
        return cls(
            depth=int(payload["depth"]),
            sample_count=int(payload["sample_count"]),
            class_counts={int(key): int(value) for key, value in dict(payload["class_counts"]).items()},
            majority_class=int(payload["majority_class"]),
            positive_prob=float(payload["positive_prob"]),
            feature_name=payload.get("feature_name"),
            split_gain=float(payload["split_gain"]) if payload.get("split_gain") is not None else None,
            children={
                str(key): cls.from_dict(value)
                for key, value in dict(children_payload).items()
            },
        )


class ID3DecisionTree:
    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_gain: float = 1e-9,
        feature_subsample_size: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.feature_subsample_size = feature_subsample_size
        self.random_state = random_state
        self.root: TreeNode | None = None
        self.feature_names: list[str] = []
        self._rng = np.random.default_rng(random_state)

    def fit(self, features: pd.DataFrame, labels: Iterable[int]) -> "ID3DecisionTree":
        working_frame = features.reset_index(drop=True).copy()
        working_frame[LABEL_COLUMN] = list(labels)
        self.feature_names = [column for column in features.columns]
        self.root = self._build_tree(
            working_frame,
            available_features=self.feature_names.copy(),
            depth=0,
        )
        return self

    def _build_tree(
        self,
        frame: pd.DataFrame,
        *,
        available_features: list[str],
        depth: int,
    ) -> TreeNode:
        label_series = frame[LABEL_COLUMN]
        counts = Counter(int(value) for value in label_series)
        majority_class = int(max(counts.items(), key=lambda item: (item[1], item[0]))[0])
        sample_count = int(len(frame))
        positive_prob = counts.get(1, 0) / sample_count if sample_count else 0.0
        node = TreeNode(
            depth=depth,
            sample_count=sample_count,
            class_counts={int(key): int(value) for key, value in counts.items()},
            majority_class=majority_class,
            positive_prob=float(positive_prob),
        )

        if len(counts) == 1:
            return node
        if not available_features:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if sample_count < self.min_samples_split:
            return node

        candidate_features = available_features
        if (
            self.feature_subsample_size is not None
            and self.feature_subsample_size < len(available_features)
        ):
            sampled = self._rng.choice(
                np.asarray(available_features, dtype=object),
                size=self.feature_subsample_size,
                replace=False,
            )
            candidate_features = [str(value) for value in sampled.tolist()]

        best_feature = None
        best_gain = float("-inf")
        for feature_name in candidate_features:
            gain = information_gain(frame, feature_name)
            if gain > best_gain:
                best_feature = feature_name
                best_gain = gain

        if best_feature is None or best_gain <= self.min_gain:
            return node

        node.feature_name = best_feature
        node.split_gain = float(best_gain)
        remaining_features = [feature for feature in available_features if feature != best_feature]
        for feature_value, subset in frame.groupby(best_feature, sort=False):
            child_frame = subset.drop(columns=[best_feature]).reset_index(drop=True)
            node.children[str(feature_value)] = self._build_tree(
                child_frame,
                available_features=remaining_features,
                depth=depth + 1,
            )
        return node

    def _require_root(self) -> TreeNode:
        if self.root is None:
            raise ValueError("The decision tree has not been fitted yet.")
        return self.root

    def predict_one(self, row: pd.Series | dict[str, object]) -> tuple[int, float]:
        node = self._require_root()
        while node.feature_name is not None:
            if isinstance(row, pd.Series):
                raw_value = row[node.feature_name]
            else:
                raw_value = row[node.feature_name]
            next_node = node.children.get(str(raw_value))
            if next_node is None:
                break
            node = next_node
        return int(node.majority_class), float(node.positive_prob)

    def predict_proba(self, features: pd.DataFrame) -> list[float]:
        probabilities = []
        for _, row in features.iterrows():
            _, positive_prob = self.predict_one(row)
            probabilities.append(float(positive_prob))
        return probabilities

    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> list[int]:
        probabilities = self.predict_proba(features)
        return [1 if probability >= threshold else 0 for probability in probabilities]

    def feature_usage_counts(self) -> dict[str, int]:
        root = self._require_root()
        counts: Counter[str] = Counter()

        def walk(node: TreeNode) -> None:
            if node.feature_name is None:
                return
            counts[node.feature_name] += 1
            for child in node.children.values():
                walk(child)

        walk(root)
        return dict(counts)

    def text_summary(self, max_depth: int | None = 4, max_lines: int = 80) -> str:
        root = self._require_root()
        lines: list[str] = []

        def walk(node: TreeNode, prefix: str = "") -> None:
            if len(lines) >= max_lines:
                return
            if node.feature_name is None or (max_depth is not None and node.depth >= max_depth):
                lines.append(
                    (
                        f"{prefix}Leaf(depth={node.depth}, samples={node.sample_count}, "
                        f"majority={node.majority_class}, p1={node.positive_prob:.3f})"
                    )
                )
                return

            lines.append(
                (
                    f"{prefix}Node(depth={node.depth}, samples={node.sample_count}, "
                    f"split={node.feature_name}, gain={node.split_gain:.6f}, "
                    f"majority={node.majority_class}, p1={node.positive_prob:.3f})"
                )
            )
            for feature_value, child in node.children.items():
                if len(lines) >= max_lines:
                    return
                lines.append(f"{prefix}  -> {node.feature_name} = {feature_value}")
                walk(child, prefix + "    ")

        walk(root)
        if len(lines) >= max_lines:
            lines.append("... summary truncated ...")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        root = self._require_root()
        return {
            "model_type": "id3_tree",
            "params": {
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_gain": self.min_gain,
                "feature_subsample_size": self.feature_subsample_size,
                "random_state": self.random_state,
            },
            "feature_names": self.feature_names,
            "root": root.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ID3DecisionTree":
        params = dict(payload["params"])
        model = cls(
            max_depth=params.get("max_depth"),
            min_samples_split=int(params["min_samples_split"]),
            min_gain=float(params["min_gain"]),
            feature_subsample_size=(
                int(params["feature_subsample_size"])
                if params.get("feature_subsample_size") is not None
                else None
            ),
            random_state=int(params["random_state"]) if params.get("random_state") is not None else None,
        )
        model.feature_names = [str(value) for value in payload.get("feature_names", [])]
        model.root = TreeNode.from_dict(dict(payload["root"]))
        return model
