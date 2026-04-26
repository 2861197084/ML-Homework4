from __future__ import annotations

from pathlib import Path

import pytest


TRAIN_ROWS = [
    "39, State-gov,77516,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States,<=50K",
    "50, Self-emp-not-inc,83311,Bachelors,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,60,United-States,>50K",
    "38, Private,215646,HS-grad,9,Divorced,Handlers-cleaners,Not-in-family,White,Male,0,0,40,United-States,<=50K",
    "53, Private,234721,11th,7,Married-civ-spouse,Handlers-cleaners,Husband,Black,Male,0,0,40,United-States,<=50K",
    "28, Private,338409,Bachelors,13,Married-civ-spouse,Prof-specialty,Wife,Black,Female,0,0,40,Cuba,>50K",
    "37, Private,284582,Masters,14,Married-civ-spouse,Exec-managerial,Wife,White,Female,0,0,40,United-States,>50K",
    "49, Private,160187,9th,5,Married-spouse-absent,Other-service,Not-in-family,Black,Female,0,0,16,Jamaica,<=50K",
    "52, Self-emp-inc,209642,HS-grad,9,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,45,United-States,>50K",
    "31, Federal-gov,45781,Masters,14,Never-married,Prof-specialty,Not-in-family,White,Female,14084,0,50,United-States,>50K",
    "42, Private,159449,Bachelors,13,Married-civ-spouse,Sales,Husband,White,Male,5178,0,40,United-States,>50K",
    "23, ?,122272,HS-grad,9,Never-married,Adm-clerical,Own-child,White,Female,0,0,30,United-States,<=50K",
    "45, Private,386940,Masters,14,Divorced,?,Unmarried,White,Female,0,0,45,United-States,<=50K",
]

TEST_ROWS = [
    "|1x3 Cross validator",
    "34, Private,198693,Bachelors,13,Married-civ-spouse,Prof-specialty,Husband,White,Male,0,0,60,United-States,>50K.",
    "29, Private,227026,HS-grad,9,Never-married,Other-service,Own-child,Black,Female,0,0,35,United-States,<=50K.",
    "46, Local-gov,75666,Masters,14,Married-civ-spouse,Protective-serv,Husband,White,Male,0,0,45,United-States,>50K.",
    "27, Private,112321,Some-college,10,Divorced,Sales,Not-in-family,White,Female,0,0,38,Canada,<=50K.",
]


def create_test_project(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "adult.data").write_text("\n".join(TRAIN_ROWS), encoding="utf-8")
    (raw_dir / "adult.test").write_text("\n".join(TEST_ROWS), encoding="utf-8")

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[paths]
raw_dir = "raw"
processed_dir = "processed"
artifacts_dir = "artifacts"
reports_dir = "reports"
train_file = "adult.data"
test_file = "adult.test"

[download]
zip_urls = []
raw_train_urls = []
raw_test_urls = []
timeout_seconds = 5

[experiment]
seed = 42
validation_ratio = 0.4
preprocess_profile = "pdf_default"

[tree]
max_depth = 6
min_samples_split = 2
min_gain = 1e-6

[forest]
n_estimators_grid = [3, 5]
max_features_grid = [2, 4]
max_depth = 6
min_samples_split = 2
min_gain = 1e-6
""".strip(),
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def mini_project(tmp_path: Path) -> Path:
    return create_test_project(tmp_path)
