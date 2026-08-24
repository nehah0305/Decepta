import pandas as pd

from .config import REQUIRED_COLUMNS, VALID_SPLITS


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Check that all required columns exist.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def validate_labels(df: pd.DataFrame) -> None:
    """
    Ensure labels contain only 0 and 1.
    """

    valid_labels = {0, 1}

    labels = set(df["label"].dropna().unique())

    invalid_labels = labels - valid_labels

    if invalid_labels:
        raise ValueError(
            f"Invalid labels found: {invalid_labels}. "
            f"Allowed labels are 0 = REAL and 1 = FAKE."
        )


def validate_splits(df: pd.DataFrame) -> None:
    """
    Validate split names if the split column already exists.
    """

    if "split" not in df.columns:
        return

    splits = set(df["split"].dropna().unique())

    invalid_splits = splits - VALID_SPLITS

    if invalid_splits:
        raise ValueError(
            f"Invalid split values found: {invalid_splits}"
        )


def validate_missing_values(df: pd.DataFrame) -> None:
    """
    Check important fields for missing values.
    """

    important_columns = [
        "sample_id",
        "dataset",
        "video_path",
        "source_id",
        "split_group_id",
        "manipulation",
        "label",
    ]

    missing = df[important_columns].isnull().sum()

    problems = missing[missing > 0]

    if not problems.empty:

        raise ValueError(
            "Missing values detected:\n"
            f"{problems}"
        )


def validate_duplicate_sample_ids(df: pd.DataFrame) -> None:
    """
    Ensure sample IDs are unique.
    """

    duplicates = df[
        df["sample_id"].duplicated(keep=False)
    ]

    if not duplicates.empty:

        duplicate_ids = duplicates["sample_id"].unique()

        raise ValueError(
            f"Duplicate sample IDs detected: "
            f"{duplicate_ids[:20]}"
        )


def validate_manifest_schema(df: pd.DataFrame) -> None:
    """
    Run all basic schema validations.
    """

    print("\nChecking manifest schema...")

    validate_required_columns(df)

    print("✓ Required columns present")

    validate_missing_values(df)

    print("✓ No missing values in required fields")

    validate_labels(df)

    print("✓ Labels valid")

    validate_splits(df)

    print("✓ Split values valid")

    validate_duplicate_sample_ids(df)

    print("✓ Sample IDs unique")

    print("✓ Manifest schema validation passed")