from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import urllib.error
import urllib.request
from zipfile import ZipFile

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from adult_lab.config import Config


COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
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
    "income",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


@dataclass(slots=True)
class RawDatasetBundle:
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    missing_summary: dict[str, object]
    data_source: str


def ensure_directories(config: Config) -> None:
    for directory in [
        config.paths.raw_dir,
        config.paths.processed_dir,
        config.paths.artifacts_dir,
        config.paths.reports_dir,
        config.paths.figures_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def _download_bytes(url: str, timeout_seconds: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "adult-lab/0.1 (+https://archive.ics.uci.edu/dataset/2/adult)"
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def _try_zip_download(config: Config) -> str | None:
    for url in config.download.zip_urls:
        try:
            payload = _download_bytes(url, config.download.timeout_seconds)
            with ZipFile(BytesIO(payload)) as archive:
                member_lookup = {Path(name).name: name for name in archive.namelist()}
                for file_name in [config.paths.train_file, config.paths.test_file]:
                    member_name = member_lookup.get(file_name)
                    if member_name is None:
                        raise FileNotFoundError(f"{file_name} not found in {url}")
                    target_path = config.paths.raw_dir / file_name
                    target_path.write_bytes(archive.read(member_name))
            return url
        except Exception:
            continue
    return None


def _try_file_download(urls: list[str], target_path: Path, timeout_seconds: int) -> str | None:
    for url in urls:
        try:
            target_path.write_bytes(_download_bytes(url, timeout_seconds))
            return url
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    return None


def ensure_data_files(config: Config) -> str:
    ensure_directories(config)
    if config.paths.raw_train_path.exists() and config.paths.raw_test_path.exists():
        return "existing-local-files"

    zip_source = _try_zip_download(config)
    if config.paths.raw_train_path.exists() and config.paths.raw_test_path.exists():
        return zip_source or "zip-download"

    train_source = None
    test_source = None
    if not config.paths.raw_train_path.exists():
        train_source = _try_file_download(
            config.download.raw_train_urls,
            config.paths.raw_train_path,
            config.download.timeout_seconds,
        )
    if not config.paths.raw_test_path.exists():
        test_source = _try_file_download(
            config.download.raw_test_urls,
            config.paths.raw_test_path,
            config.download.timeout_seconds,
        )

    if config.paths.raw_train_path.exists() and config.paths.raw_test_path.exists():
        return test_source or train_source or "raw-file-download"

    missing = []
    if not config.paths.raw_train_path.exists():
        missing.append(str(config.paths.raw_train_path))
    if not config.paths.raw_test_path.exists():
        missing.append(str(config.paths.raw_test_path))
    raise FileNotFoundError(
        "Failed to download Adult dataset files. Missing: " + ", ".join(missing)
    )


def _load_single_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        comment="|",
        skip_blank_lines=True,
        skipinitialspace=True,
        na_values=["?"],
    )
    frame = frame.dropna(how="all")
    for column in frame.columns:
        if is_object_dtype(frame[column]) or is_string_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    frame["income"] = frame["income"].str.rstrip(".")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _split_missing_summary(raw_frame: pd.DataFrame, clean_frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows_before_cleaning": int(len(raw_frame)),
        "rows_after_cleaning": int(len(clean_frame)),
        "rows_with_missing": int(raw_frame.isna().any(axis=1).sum()),
        "rows_removed": int(len(raw_frame) - len(clean_frame)),
        "missing_per_column": {
            column: int(value)
            for column, value in raw_frame.isna().sum().to_dict().items()
            if int(value) > 0
        },
    }


def load_raw_dataset(config: Config) -> RawDatasetBundle:
    data_source = ensure_data_files(config)
    train_raw = _load_single_split(config.paths.raw_train_path)
    test_raw = _load_single_split(config.paths.raw_test_path)

    train_clean = train_raw.dropna().reset_index(drop=True)
    test_clean = test_raw.dropna().reset_index(drop=True)
    missing_summary = {
        "train": _split_missing_summary(train_raw, train_clean),
        "test": _split_missing_summary(test_raw, test_clean),
    }

    return RawDatasetBundle(
        train_raw=train_raw,
        test_raw=test_raw,
        train=train_clean,
        test=test_clean,
        missing_summary=missing_summary,
        data_source=data_source,
    )
