"""Data discovery and train/test splitting utilities.

The code is intentionally flexible because Kaggle image datasets are sometimes
extracted with slightly different top-level folder names. The only hard
requirement is that images are eventually inside class folders such as
``WithMask`` and ``WithoutMask``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_LABEL_ALIASES = {
    "withmask": "WithMask",
    "mask": "WithMask",
    "masked": "WithMask",
    "wearingmask": "WithMask",
    "withoutmask": "WithoutMask",
    "nomask": "WithoutMask",
    "unmasked": "WithoutMask",
    "notmask": "WithoutMask",
}

_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "validation",
    "val": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}


def _normalize_name(name: str) -> str:
    """Normalize folder names for robust label/split detection."""
    return (
        name.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


def infer_label(path: Path) -> Optional[str]:
    """Infer the class label from an image path's parent folders."""
    for parent in [path.parent, *path.parents]:
        key = _normalize_name(parent.name)
        if key in _LABEL_ALIASES:
            return _LABEL_ALIASES[key]
    return None


def infer_split(path: Path) -> str:
    """Infer split name from parent folders; return ``all`` if not present."""
    for parent in [path.parent, *path.parents]:
        key = _normalize_name(parent.name)
        if key in _SPLIT_ALIASES:
            return _SPLIT_ALIASES[key]
    return "all"


def find_image_records(data_dir: str | Path) -> pd.DataFrame:
    """Return a DataFrame with image paths, labels, and split names.

    Parameters
    ----------
    data_dir:
        Folder containing the extracted dataset. Usually ``data/raw``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``path``, ``label``, and ``split``.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}\n"
            "Create it and place the extracted Kaggle dataset inside it."
        )

    rows = []
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = infer_label(path)
        if label is None:
            continue
        rows.append({"path": str(path), "label": label, "split": infer_split(path)})

    records = pd.DataFrame(rows)
    if records.empty:
        raise ValueError(
            "No labeled images were found. Expected folders such as:\n"
            "  data/raw/Face Mask Dataset/Train/WithMask/*.png\n"
            "  data/raw/Face Mask Dataset/Train/WithoutMask/*.png\n"
            "See data/readme_data.txt for setup instructions."
        )

    records = records.drop_duplicates(subset=["path"]).reset_index(drop=True)
    return records


def limit_per_class(
    records: pd.DataFrame,
    max_per_class: Optional[int],
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Optionally cap records per class for quick experiments."""
    if max_per_class is None or max_per_class <= 0:
        return records.reset_index(drop=True)

    limited = (
        records.groupby("label", group_keys=False)
        .apply(lambda df: df.sample(min(len(df), max_per_class), random_state=random_state))
        .reset_index(drop=True)
    )
    return limited.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def prepare_splits(
    records: pd.DataFrame,
    test_size: float = 0.20,
    validation_size: float = 0.10,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare train, validation, and test splits.

    If the dataset already has ``Train``, ``Validation``, and ``Test`` folders,
    those splits are used directly. Otherwise, this function creates stratified
    splits from all available labeled images.
    """
    records = records.reset_index(drop=True)
    split_values = set(records["split"].unique())

    if "train" in split_values and "test" in split_values:
        train = records[records["split"] == "train"].copy()
        validation = records[records["split"] == "validation"].copy()
        test = records[records["split"] == "test"].copy()

        if validation.empty and validation_size > 0 and len(train) > 4:
            train, validation = train_test_split(
                train,
                test_size=validation_size,
                stratify=train["label"],
                random_state=random_state,
            )
        return (
            train.reset_index(drop=True),
            validation.reset_index(drop=True),
            test.reset_index(drop=True),
        )

    # No explicit split folders. Make new stratified splits.
    train, test = train_test_split(
        records,
        test_size=test_size,
        stratify=records["label"],
        random_state=random_state,
    )

    if validation_size > 0 and len(train) > 4:
        adjusted_val_size = validation_size / (1.0 - test_size)
        train, validation = train_test_split(
            train,
            test_size=adjusted_val_size,
            stratify=train["label"],
            random_state=random_state,
        )
    else:
        validation = pd.DataFrame(columns=records.columns)

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def summarize_records(records: pd.DataFrame) -> str:
    """Create a readable count summary by split and label."""
    if records.empty:
        return "No records found."
    counts = records.groupby(["split", "label"]).size().unstack(fill_value=0)
    return counts.to_string()
