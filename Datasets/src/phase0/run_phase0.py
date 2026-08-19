import pandas as pd

from .config import (
    MASTER_MANIFEST,
    TRAIN_MANIFEST,
    VALIDATION_MANIFEST,
    TEST_MANIFEST,
    REPORTS_DIR,
)

from .dataset_schema import (
    validate_manifest_schema,
)

from .split_dataset import (
    split_dataset,
    print_split_statistics,
)

from .leakage_checker import (
    check_subject_leakage,
    check_source_leakage,
    print_leakage_report,
)


def save_split_manifests(
    df: pd.DataFrame,
) -> None:

    df[
        df["split"] == "train"
    ].to_csv(
        TRAIN_MANIFEST,
        index=False
    )

    df[
        df["split"] == "validation"
    ].to_csv(
        VALIDATION_MANIFEST,
        index=False
    )

    df[
        df["split"] == "test"
    ].to_csv(
        TEST_MANIFEST,
        index=False
    )

    print(
        "\nSplit manifests saved."
    )


def generate_statistics(
    df: pd.DataFrame,
) -> None:

    statistics = {
        "total_samples": len(df),

        "real_samples": int(
            (df["label"] == 0).sum()
        ),

        "fake_samples": int(
            (df["label"] == 1).sum()
        ),

        "train_samples": int(
            (df["split"] == "train").sum()
        ),

        "validation_samples": int(
            (df["split"] == "validation").sum()
        ),

        "test_samples": int(
            (df["split"] == "test").sum()
        ),
    }

    statistics_df = pd.DataFrame(
        [statistics]
    )

    output_path = (
        REPORTS_DIR
        / "dataset_statistics.csv"
    )

    statistics_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nStatistics saved to:\n"
        f"{output_path}"
    )


def main():

    print("\n")
    print("=" * 60)
    print("       DEEPFAKE DETECTION - PHASE 0")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD MASTER MANIFEST
    # --------------------------------------------------------

    print(
        f"\nLoading manifest:\n"
        f"{MASTER_MANIFEST}"
    )

    if not MASTER_MANIFEST.exists():

        raise FileNotFoundError(
            "\nMaster manifest does not exist.\n"
            "Create datasets/metadata/master.csv "
            "before running Phase 0."
        )

    df = pd.read_csv(
        MASTER_MANIFEST
    )

    print(
        f"Loaded {len(df)} samples."
    )

    # --------------------------------------------------------
    # VALIDATE SCHEMA
    # --------------------------------------------------------

    validate_manifest_schema(df)

    # --------------------------------------------------------
    # SPLIT DATASET
    # --------------------------------------------------------

    print(
        "\nCreating train/validation/test splits..."
    )

    df = split_dataset(df)

    # --------------------------------------------------------
    # SAVE UPDATED MASTER MANIFEST
    # --------------------------------------------------------

    df.to_csv(
        MASTER_MANIFEST,
        index=False
    )

    print(
        "\nUpdated master manifest saved."
    )

    # --------------------------------------------------------
    # PRINT STATISTICS
    # --------------------------------------------------------

    print_split_statistics(df)

    # --------------------------------------------------------
    # SAVE SPLITS
    # --------------------------------------------------------

    save_split_manifests(df)

    # --------------------------------------------------------
    # CHECK LEAKAGE
    # --------------------------------------------------------

    subject_results = check_subject_leakage(df)

    source_results = check_source_leakage(df)

    leakage_passed = print_leakage_report(
        subject_results,
        source_results,
    )

    # --------------------------------------------------------
    # GENERATE STATISTICS
    # --------------------------------------------------------

    generate_statistics(df)

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)

    if leakage_passed:

        print(
            "PHASE 0 COMPLETED SUCCESSFULLY"
        )

    else:

        print(
            "PHASE 0 FAILED - DATA LEAKAGE DETECTED"
        )

    print("=" * 60)


if __name__ == "__main__":

    main()