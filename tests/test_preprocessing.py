from __future__ import annotations

import pandas as pd

from adult_lab.data import RawDatasetBundle
from adult_lab.preprocessing import apply_preprocessing


def test_pdf_default_preprocessing_maps_columns_as_expected() -> None:
    frame = pd.DataFrame(
        [
            {
                "age": 52,
                "workclass": "State-gov",
                "fnlwgt": 1,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Divorced",
                "occupation": "Prof-specialty",
                "relationship": "Not-in-family",
                "race": "Black",
                "sex": "Female",
                "capital-gain": 1,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": "Canada",
                "income": ">50K",
            }
        ]
    )
    bundle = RawDatasetBundle(
        train_raw=frame,
        test_raw=frame,
        train=frame,
        test=frame,
        missing_summary={},
        data_source="unit-test",
    )

    processed = apply_preprocessing(bundle, "pdf_default")
    row = processed.train.iloc[0]

    assert row["age"] == "50-74"
    assert row["workclass"] == "Government"
    assert row["education-num"] == "10+"
    assert row["marital-status"] == "not-married"
    assert row["occupation"] == "High"
    assert row["relationship"] == "Other"
    assert row["race"] == "Other"
    assert row["capital-gain"] == "1"
    assert row["capital-loss"] == "0"
    assert row["hours-per-week"] == ">40"
    assert row["native-country"] == "notUSA"
    assert row["income"] == 1
