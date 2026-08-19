import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

from .config import (
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


def split_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if abs(
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
        - 1.0
    ) > 1e-6:

        raise ValueError(
            "Train, validation and test ratios "
            "must sum to 1."
        )

    df = df.copy()

    # --------------------------------------------------------
    # FIRST SPLIT
    # Train vs temporary
    # --------------------------------------------------------

    train_ratio = TRAIN_RATIO

    temporary_ratio = (
        VALIDATION_RATIO
        + TEST_RATIO
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=temporary_ratio,
        random_state=RANDOM_SEED,
    )

    train_indices, temporary_indices = next(
        splitter.split(
            df,
            groups=df["subject_id"]
        )
    )

    df["split"] = "temporary"

    df.loc[
        train_indices,
        "split"
    ] = "train"

    # --------------------------------------------------------
    # SECOND SPLIT
    # Validation vs Test
    # --------------------------------------------------------

    temporary_df = df[
        df["split"] == "temporary"
    ].copy()

    validation_fraction = (
        VALIDATION_RATIO
        /
        (
            VALIDATION_RATIO
            + TEST_RATIO
        )
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=1 - validation_fraction,
        random_state=RANDOM_SEED,
    )

    validation_indices, test_indices = next(
        splitter.split(
            temporary_df,
            groups=temporary_df["subject_id"]
        )
    )

    temporary_df.loc[
        temporary_df.index[validation_indices],
        "split"
    ] = "validation"

    temporary_df.loc[
        temporary_df.index[test_indices],
        "split"
    ] = "test"

    df.loc[
        temporary_df.index,
        "split"
    ] = temporary_df["split"]

    return df


def print_split_statistics(
    df: pd.DataFrame,
) -> None:

    print("\n================================")
    print("       SPLIT STATISTICS")
    print("================================")

    print(
        "\nSamples per split:"
    )

    print(
        df["split"].value_counts()
    )

    print(
        "\nLabels per split:"
    )

    print(
        pd.crosstab(
            df["split"],
            df["label"]
        )
    )