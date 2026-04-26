from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adult_lab.config import Config, load_config
from adult_lab.data import RawDatasetBundle, ensure_data_files, ensure_directories, load_raw_dataset
from adult_lab.metrics import evaluate_binary_classification
from adult_lab.models import ID3DecisionTree, RandomForestID3
from adult_lab.preprocessing import ProcessedDatasetBundle, apply_preprocessing
from adult_lab.reporting import (
    generate_markdown_report,
    plot_confusion_matrices,
    plot_feature_usage,
    plot_forest_grid_heatmap,
    plot_label_distribution,
    plot_metrics_comparison,
    plot_missing_summary,
    plot_processed_feature_distributions,
    plot_roc_curves,
    plot_tree_train_test_metrics,
)


CORRECTIONS = [
    "测试集标签带有句点时统一去掉尾部 `.`，否则会导致标签编码与评估错误。",
    "将 `capital-loss` 与 `capital-gain` 一致地按 `>0` 和 `=0` 二值化，不复刻附录中的误用转换函数问题。",
    "`workclass` 仅将政府与自雇两组做规则合并，其余合法类别保持原值，不机械照搬 OCR 失真的类别描述。",
    "`adult.test` 首行说明文本在读取时显式跳过，避免把元信息误读为样本。",
]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _processed_train_path(config: Config) -> Path:
    return config.paths.processed_dir / "train_processed.csv"


def _processed_test_path(config: Config) -> Path:
    return config.paths.processed_dir / "test_processed.csv"


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def persist_processed_data(config: Config, processed: ProcessedDatasetBundle) -> None:
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    processed.train.to_csv(_processed_train_path(config), index=False)
    processed.test.to_csv(_processed_test_path(config), index=False)


def build_profile_summary(
    raw_bundle: RawDatasetBundle,
    processed: ProcessedDatasetBundle,
    config: Config,
) -> dict[str, Any]:
    return {
        "data_source": raw_bundle.data_source,
        "preprocess_profile": config.experiment.preprocess_profile,
        "missing_summary": raw_bundle.missing_summary,
        "processed": {
            "train_rows": int(len(processed.train)),
            "test_rows": int(len(processed.test)),
            "train_label_distribution": {
                str(key): int(value)
                for key, value in processed.train["income"].value_counts().sort_index().to_dict().items()
            },
            "test_label_distribution": {
                str(key): int(value)
                for key, value in processed.test["income"].value_counts().sort_index().to_dict().items()
            },
        },
    }


def prepare_dataset(config: Config) -> tuple[RawDatasetBundle, ProcessedDatasetBundle, dict[str, Any]]:
    ensure_directories(config)
    raw_bundle = load_raw_dataset(config)
    processed = apply_preprocessing(raw_bundle, config.experiment.preprocess_profile)
    persist_processed_data(config, processed)
    profile_summary = build_profile_summary(raw_bundle, processed, config)
    save_json(config.paths.profile_summary_path, profile_summary)
    return raw_bundle, processed, profile_summary


def stratified_split(
    frame: pd.DataFrame,
    *,
    label_column: str,
    validation_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []

    for _, group in frame.groupby(label_column, sort=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        validation_count = int(round(len(indices) * validation_ratio))
        if len(indices) > 1:
            validation_count = max(1, min(len(indices) - 1, validation_count))
        else:
            validation_count = 0
        validation_indices = indices[:validation_count]
        train_indices = indices[validation_count:]
        validation_parts.append(frame.loc[validation_indices])
        train_parts.append(frame.loc[train_indices])

    train_split = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    validation_split = (
        pd.concat(validation_parts).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
        if validation_parts
        else frame.iloc[0:0].copy()
    )
    return train_split, validation_split


def train_tree_from_processed(
    processed: ProcessedDatasetBundle,
    config: Config,
) -> tuple[ID3DecisionTree, dict[str, Any]]:
    model = ID3DecisionTree(
        max_depth=config.tree.max_depth,
        min_samples_split=config.tree.min_samples_split,
        min_gain=config.tree.min_gain,
        random_state=config.experiment.seed,
    )
    model.fit(processed.train[processed.feature_columns], processed.train[processed.label_column])
    metadata = {
        "trained_at": _timestamp(),
        "feature_columns": processed.feature_columns,
        "feature_usage_counts": model.feature_usage_counts(),
        "tree_summary": model.text_summary(),
        "params": {
            "max_depth": config.tree.max_depth,
            "min_samples_split": config.tree.min_samples_split,
            "min_gain": config.tree.min_gain,
        },
    }
    config.paths.tree_summary_path.write_text(metadata["tree_summary"], encoding="utf-8")
    save_json(
        config.paths.tree_model_path,
        {
            "metadata": metadata,
            "model": model.to_dict(),
        },
    )
    return model, metadata


def _forest_score_key(result: dict[str, Any]) -> tuple[float, float, int, int]:
    return (
        float(result["auc"]),
        float(result["accuracy"]),
        -int(result["n_estimators"]),
        -int(result["max_features"]),
    )


def train_forest_from_processed(
    processed: ProcessedDatasetBundle,
    config: Config,
) -> tuple[RandomForestID3, dict[str, Any]]:
    feature_count = len(processed.feature_columns)
    search_train, validation = stratified_split(
        processed.train,
        label_column=processed.label_column,
        validation_ratio=config.experiment.validation_ratio,
        seed=config.experiment.seed,
    )
    search_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_model_params: dict[str, int] | None = None

    max_feature_candidates = sorted({min(value, feature_count) for value in config.forest.max_features_grid})
    n_estimators_candidates = sorted(set(config.forest.n_estimators_grid))

    for max_features in max_feature_candidates:
        for n_estimators in n_estimators_candidates:
            candidate_model = RandomForestID3(
                n_estimators=n_estimators,
                max_features=max_features,
                max_depth=config.forest.max_depth,
                min_samples_split=config.forest.min_samples_split,
                min_gain=config.forest.min_gain,
                random_state=config.experiment.seed + n_estimators * 31 + max_features * 101,
            )
            candidate_model.fit(
                search_train[processed.feature_columns],
                search_train[processed.label_column],
            )
            validation_probabilities = candidate_model.predict_proba(validation[processed.feature_columns])
            validation_metrics = evaluate_binary_classification(
                validation[processed.label_column].tolist(),
                validation_probabilities,
            )
            result = {
                "n_estimators": n_estimators,
                "max_features": max_features,
                "accuracy": validation_metrics["accuracy"],
                "f1": validation_metrics["f1"],
                "auc": validation_metrics["auc"],
            }
            search_results.append(result)
            if best_result is None or _forest_score_key(result) > _forest_score_key(best_result):
                best_result = result
                best_model_params = {
                    "n_estimators": n_estimators,
                    "max_features": max_features,
                }

    if best_result is None or best_model_params is None:
        raise RuntimeError("Random forest grid search did not produce any candidate.")

    final_model = RandomForestID3(
        n_estimators=best_model_params["n_estimators"],
        max_features=best_model_params["max_features"],
        max_depth=config.forest.max_depth,
        min_samples_split=config.forest.min_samples_split,
        min_gain=config.forest.min_gain,
        random_state=config.experiment.seed,
    )
    final_model.fit(processed.train[processed.feature_columns], processed.train[processed.label_column])
    metadata = {
        "trained_at": _timestamp(),
        "feature_columns": processed.feature_columns,
        "search_train_rows": int(len(search_train)),
        "validation_rows": int(len(validation)),
        "search_results": search_results,
        "best_params": best_model_params,
        "best_validation_metrics": best_result,
        "feature_usage_counts": final_model.feature_usage_counts(),
        "params": {
            "max_depth": config.forest.max_depth,
            "min_samples_split": config.forest.min_samples_split,
            "min_gain": config.forest.min_gain,
        },
    }
    save_json(
        config.paths.forest_model_path,
        {
            "metadata": metadata,
            "model": final_model.to_dict(),
        },
    )
    return final_model, metadata


def evaluate_model_on_frame(
    model: ID3DecisionTree | RandomForestID3,
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_column: str,
) -> dict[str, Any]:
    probabilities = model.predict_proba(frame[feature_columns])
    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    metrics = evaluate_binary_classification(frame[label_column].tolist(), probabilities)
    return {
        "metrics": metrics,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def load_tree_model(path: Path) -> tuple[ID3DecisionTree, dict[str, Any]]:
    payload = load_json(path)
    return ID3DecisionTree.from_dict(payload["model"]), payload["metadata"]


def load_forest_model(path: Path) -> tuple[RandomForestID3, dict[str, Any]]:
    payload = load_json(path)
    return RandomForestID3.from_dict(payload["model"]), payload["metadata"]


def download_data(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_directories(config)
    source = ensure_data_files(config)
    return {
        "train_path": str(config.paths.raw_train_path),
        "test_path": str(config.paths.raw_test_path),
        "source": source,
    }


def profile_data(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    raw_bundle, processed, profile_summary = prepare_dataset(config)
    figure_paths = {
        "label_distribution": config.paths.figures_dir / "label_distribution.png",
        "missing_summary": config.paths.figures_dir / "missing_summary.png",
        "processed_feature_distributions": config.paths.figures_dir / "processed_feature_distributions.png",
    }
    plot_label_distribution(processed, figure_paths["label_distribution"])
    plot_missing_summary(raw_bundle.missing_summary, figure_paths["missing_summary"])
    plot_processed_feature_distributions(processed, figure_paths["processed_feature_distributions"])
    return {
        "profile_summary_path": str(config.paths.profile_summary_path),
        "figure_paths": {key: str(value) for key, value in figure_paths.items()},
        "profile_summary": profile_summary,
    }


def train_model(config_path: str | Path, model_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    _, processed, _ = prepare_dataset(config)
    if model_name == "tree":
        _, metadata = train_tree_from_processed(processed, config)
        return {
            "model": model_name,
            "model_path": str(config.paths.tree_model_path),
            "metadata": metadata,
        }
    if model_name == "forest":
        _, metadata = train_forest_from_processed(processed, config)
        return {
            "model": model_name,
            "model_path": str(config.paths.forest_model_path),
            "metadata": metadata,
        }
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_model(config_path: str | Path, model_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    _, processed, profile_summary = prepare_dataset(config)

    if model_name == "tree":
        model, metadata = load_tree_model(config.paths.tree_model_path)
    elif model_name == "forest":
        model, metadata = load_forest_model(config.paths.forest_model_path)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    train_evaluation = evaluate_model_on_frame(
        model,
        processed.train,
        feature_columns=processed.feature_columns,
        label_column=processed.label_column,
    )
    test_evaluation = evaluate_model_on_frame(
        model,
        processed.test,
        feature_columns=processed.feature_columns,
        label_column=processed.label_column,
    )
    payload = {
        "generated_at": _timestamp(),
        "profile": profile_summary,
        model_name: {
            "metadata": metadata,
            "train": train_evaluation["metrics"],
            "test": test_evaluation["metrics"],
        },
    }
    save_json(config.paths.metrics_path, payload)
    return payload


def run_all(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    raw_bundle, processed, profile_summary = prepare_dataset(config)

    tree_model, tree_metadata = train_tree_from_processed(processed, config)
    forest_model, forest_metadata = train_forest_from_processed(processed, config)

    tree_train = evaluate_model_on_frame(
        tree_model,
        processed.train,
        feature_columns=processed.feature_columns,
        label_column=processed.label_column,
    )
    tree_test = evaluate_model_on_frame(
        tree_model,
        processed.test,
        feature_columns=processed.feature_columns,
        label_column=processed.label_column,
    )
    forest_train = evaluate_model_on_frame(
        forest_model,
        processed.train,
        feature_columns=processed.feature_columns,
        label_column=processed.label_column,
    )
    forest_test = evaluate_model_on_frame(
        forest_model,
        processed.test,
        feature_columns=processed.feature_columns,
        label_column=processed.label_column,
    )

    predictions = processed.test[processed.feature_columns + [processed.label_column]].copy()
    predictions.insert(0, "row_id", range(len(predictions)))
    predictions = predictions.rename(columns={processed.label_column: "y_true"})
    predictions["tree_pred"] = tree_test["predictions"]
    predictions["tree_proba"] = tree_test["probabilities"]
    predictions["forest_pred"] = forest_test["predictions"]
    predictions["forest_proba"] = forest_test["probabilities"]
    predictions.to_csv(config.paths.predictions_path, index=False)

    figure_paths = {
        "label_distribution": config.paths.figures_dir / "label_distribution.png",
        "missing_summary": config.paths.figures_dir / "missing_summary.png",
        "processed_feature_distributions": config.paths.figures_dir / "processed_feature_distributions.png",
        "tree_train_test_metrics": config.paths.figures_dir / "tree_train_test_metrics.png",
        "confusion_matrices": config.paths.figures_dir / "confusion_matrices.png",
        "roc_curves": config.paths.figures_dir / "roc_curves.png",
        "metrics_comparison": config.paths.figures_dir / "metrics_comparison.png",
        "forest_grid_heatmap": config.paths.figures_dir / "forest_grid_heatmap.png",
        "feature_usage": config.paths.figures_dir / "feature_usage.png",
    }
    plot_label_distribution(processed, figure_paths["label_distribution"])
    plot_missing_summary(raw_bundle.missing_summary, figure_paths["missing_summary"])
    plot_processed_feature_distributions(processed, figure_paths["processed_feature_distributions"])
    plot_tree_train_test_metrics(
        {"train": tree_train["metrics"], "test": tree_test["metrics"]},
        figure_paths["tree_train_test_metrics"],
    )
    plot_confusion_matrices(tree_test["metrics"], forest_test["metrics"], figure_paths["confusion_matrices"])
    plot_roc_curves(tree_test["metrics"], forest_test["metrics"], figure_paths["roc_curves"])
    plot_metrics_comparison(tree_test["metrics"], forest_test["metrics"], figure_paths["metrics_comparison"])
    plot_forest_grid_heatmap(forest_metadata["search_results"], figure_paths["forest_grid_heatmap"])
    plot_feature_usage(
        processed.feature_columns,
        tree_metadata["feature_usage_counts"],
        forest_metadata["feature_usage_counts"],
        figure_paths["feature_usage"],
    )

    metrics_payload = {
        "generated_at": _timestamp(),
        "profile": profile_summary,
        "tree": {
            "metadata": tree_metadata,
            "train": tree_train["metrics"],
            "test": tree_test["metrics"],
        },
        "forest": {
            "metadata": forest_metadata,
            "train": forest_train["metrics"],
            "test": forest_test["metrics"],
        },
    }
    save_json(config.paths.metrics_path, metrics_payload)

    generate_markdown_report(
        report_path=config.paths.report_path,
        profile_summary=profile_summary,
        figure_paths=figure_paths,
        tree_metadata=tree_metadata,
        tree_metrics={"train": tree_train["metrics"], "test": tree_test["metrics"]},
        forest_metadata=forest_metadata,
        forest_metrics={"train": forest_train["metrics"], "test": forest_test["metrics"]},
        corrections=CORRECTIONS,
        metrics_path=config.paths.metrics_path,
        predictions_path=config.paths.predictions_path,
        tree_model_path=config.paths.tree_model_path,
        forest_model_path=config.paths.forest_model_path,
    )

    return {
        "metrics_path": str(config.paths.metrics_path),
        "predictions_path": str(config.paths.predictions_path),
        "tree_model_path": str(config.paths.tree_model_path),
        "forest_model_path": str(config.paths.forest_model_path),
        "report_path": str(config.paths.report_path),
        "figure_paths": {key: str(value) for key, value in figure_paths.items()},
    }
