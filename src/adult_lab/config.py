from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(slots=True)
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    train_file: str
    test_file: str

    @property
    def raw_train_path(self) -> Path:
        return self.raw_dir / self.train_file

    @property
    def raw_test_path(self) -> Path:
        return self.raw_dir / self.test_file

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    @property
    def predictions_path(self) -> Path:
        return self.artifacts_dir / "predictions.csv"

    @property
    def tree_model_path(self) -> Path:
        return self.artifacts_dir / "model_tree.json"

    @property
    def forest_model_path(self) -> Path:
        return self.artifacts_dir / "model_forest.json"

    @property
    def profile_summary_path(self) -> Path:
        return self.artifacts_dir / "profile_summary.json"

    @property
    def tree_summary_path(self) -> Path:
        return self.artifacts_dir / "tree_summary.txt"

    @property
    def figures_dir(self) -> Path:
        return self.artifacts_dir / "figures"

    @property
    def report_path(self) -> Path:
        return self.reports_dir / "experiment_report.md"


@dataclass(slots=True)
class DownloadConfig:
    zip_urls: list[str]
    raw_train_urls: list[str]
    raw_test_urls: list[str]
    timeout_seconds: int


@dataclass(slots=True)
class ExperimentConfig:
    seed: int
    validation_ratio: float
    preprocess_profile: str


@dataclass(slots=True)
class TreeConfig:
    max_depth: int | None
    min_samples_split: int
    min_gain: float


@dataclass(slots=True)
class ForestConfig:
    n_estimators_grid: list[int]
    max_features_grid: list[int]
    max_depth: int | None
    min_samples_split: int
    min_gain: float


@dataclass(slots=True)
class Config:
    root_dir: Path
    config_path: Path
    paths: PathsConfig
    download: DownloadConfig
    experiment: ExperimentConfig
    tree: TreeConfig
    forest: ForestConfig


def _resolve_path(base_dir: Path, raw_value: str) -> Path:
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(config_path: str | Path) -> Config:
    resolved_config_path = Path(config_path).resolve()
    with resolved_config_path.open("rb") as file:
        payload = tomllib.load(file)

    base_dir = resolved_config_path.parent
    root_dir = base_dir.resolve()

    paths_data = payload["paths"]
    download_data = payload["download"]
    experiment_data = payload["experiment"]
    tree_data = payload["tree"]
    forest_data = payload["forest"]

    paths = PathsConfig(
        raw_dir=_resolve_path(base_dir, paths_data["raw_dir"]),
        processed_dir=_resolve_path(base_dir, paths_data["processed_dir"]),
        artifacts_dir=_resolve_path(base_dir, paths_data["artifacts_dir"]),
        reports_dir=_resolve_path(base_dir, paths_data["reports_dir"]),
        train_file=paths_data["train_file"],
        test_file=paths_data["test_file"],
    )
    download = DownloadConfig(
        zip_urls=list(download_data["zip_urls"]),
        raw_train_urls=list(download_data["raw_train_urls"]),
        raw_test_urls=list(download_data["raw_test_urls"]),
        timeout_seconds=int(download_data["timeout_seconds"]),
    )
    experiment = ExperimentConfig(
        seed=int(experiment_data["seed"]),
        validation_ratio=float(experiment_data["validation_ratio"]),
        preprocess_profile=str(experiment_data["preprocess_profile"]),
    )
    tree = TreeConfig(
        max_depth=int(tree_data["max_depth"]) if tree_data["max_depth"] is not None else None,
        min_samples_split=int(tree_data["min_samples_split"]),
        min_gain=float(tree_data["min_gain"]),
    )
    forest = ForestConfig(
        n_estimators_grid=[int(value) for value in forest_data["n_estimators_grid"]],
        max_features_grid=[int(value) for value in forest_data["max_features_grid"]],
        max_depth=int(forest_data["max_depth"]) if forest_data["max_depth"] is not None else None,
        min_samples_split=int(forest_data["min_samples_split"]),
        min_gain=float(forest_data["min_gain"]),
    )
    return Config(
        root_dir=root_dir,
        config_path=resolved_config_path,
        paths=paths,
        download=download,
        experiment=experiment,
        tree=tree,
        forest=forest,
    )
