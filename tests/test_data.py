from __future__ import annotations

from adult_lab.config import load_config
from adult_lab.data import load_raw_dataset


def test_load_raw_dataset_handles_missing_values_and_test_header(mini_project) -> None:
    config = load_config(mini_project)
    bundle = load_raw_dataset(config)

    assert len(bundle.train_raw) == 12
    assert len(bundle.train) == 10
    assert len(bundle.test_raw) == 4
    assert len(bundle.test) == 4
    assert bundle.missing_summary["train"]["rows_removed"] == 2
    assert bundle.test["income"].tolist() == [">50K", "<=50K", ">50K", "<=50K"]
