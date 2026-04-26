from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from adult_lab.data import RawDatasetBundle


KEY_FEATURES = [
    "age",
    "workclass",
    "occupation",
    "marital-status",
    "hours-per-week",
    "native-country",
]

FEATURE_COLUMNS = [
    "age",
    "workclass",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]


@dataclass(slots=True)
class ProcessedDatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    label_column: str
    profile_name: str


def _bucket_age(value: int) -> str:
    if value < 25:
        return "0-24"
    if value < 50:
        return "25-49"
    if value < 75:
        return "50-74"
    return "75+"


def _binary_nonzero(value: int) -> str:
    return "1" if value > 0 else "0"


def _bucket_hours(value: int) -> str:
    if value < 40:
        return "<40"
    if value == 40:
        return "40"
    return ">40"


def _map_country(value: str) -> str:
    return "USA" if value == "United-States" else "notUSA"


def _map_workclass(value: str) -> str:
    if value in {"Federal-gov", "Local-gov", "State-gov"}:
        return "Government"
    if value in {"Self-emp-not-inc", "Self-emp-inc"}:
        return "Proprietor"
    return value


def _bucket_education_num(value: int) -> str:
    if value < 5:
        return "0-4"
    if value < 10:
        return "5-9"
    return "10+"


def _map_marital_status(value: str) -> str:
    if value in {"Divorced", "Never-married", "Separated", "Widowed"}:
        return "not-married"
    return "married"


def _map_occupation(value: str) -> str:
    if value in {"Prof-specialty", "Exec-managerial"}:
        return "High"
    if value in {
        "Tech-support",
        "Transport-moving",
        "Protective-serv",
        "Sales",
        "Craft-repair",
        "Armed-Forces",
    }:
        return "Med"
    return "Low"


def _map_relationship(value: str) -> str:
    if value in {"Husband", "Wife"}:
        return value
    return "Other"


def _map_race(value: str) -> str:
    return "White" if value == "White" else "Other"


def _encode_income(value: str) -> int:
    normalized = value.strip()
    if normalized == ">50K":
        return 1
    if normalized == "<=50K":
        return 0
    raise ValueError(f"Unexpected income label: {value}")


def _transform_frame(frame: pd.DataFrame) -> pd.DataFrame:
    transformed = frame.copy()
    transformed = transformed.drop(columns=["fnlwgt", "education"])
    transformed["age"] = transformed["age"].map(_bucket_age)
    transformed["capital-gain"] = transformed["capital-gain"].map(_binary_nonzero)
    transformed["capital-loss"] = transformed["capital-loss"].map(_binary_nonzero)
    transformed["hours-per-week"] = transformed["hours-per-week"].map(_bucket_hours)
    transformed["native-country"] = transformed["native-country"].map(_map_country)
    transformed["workclass"] = transformed["workclass"].map(_map_workclass)
    transformed["education-num"] = transformed["education-num"].map(_bucket_education_num)
    transformed["marital-status"] = transformed["marital-status"].map(_map_marital_status)
    transformed["occupation"] = transformed["occupation"].map(_map_occupation)
    transformed["relationship"] = transformed["relationship"].map(_map_relationship)
    transformed["race"] = transformed["race"].map(_map_race)
    transformed["income"] = transformed["income"].map(_encode_income)
    return transformed.reset_index(drop=True)


def apply_preprocessing(bundle: RawDatasetBundle, profile_name: str) -> ProcessedDatasetBundle:
    if profile_name != "pdf_default":
        raise ValueError(f"Unsupported preprocessing profile: {profile_name}")

    train = _transform_frame(bundle.train)
    test = _transform_frame(bundle.test)
    return ProcessedDatasetBundle(
        train=train,
        test=test,
        feature_columns=FEATURE_COLUMNS.copy(),
        label_column="income",
        profile_name=profile_name,
    )
