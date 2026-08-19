import pandas as pd


def find_set_overlap(
    first_set: set,
    second_set: set,
) -> set:

    return first_set.intersection(second_set)


def check_subject_leakage(
    df: pd.DataFrame,
) -> dict:

    results = {}

    train_subjects = set(
        df[df["split"] == "train"]["subject_id"]
    )

    validation_subjects = set(
        df[df["split"] == "validation"]["subject_id"]
    )

    test_subjects = set(
        df[df["split"] == "test"]["subject_id"]
    )

    results["train_validation"] = find_set_overlap(
        train_subjects,
        validation_subjects
    )

    results["train_test"] = find_set_overlap(
        train_subjects,
        test_subjects
    )

    results["validation_test"] = find_set_overlap(
        validation_subjects,
        test_subjects
    )

    return results


def check_source_leakage(
    df: pd.DataFrame,
) -> dict:

    results = {}

    train_sources = set(
        df[df["split"] == "train"]["source_id"]
    )

    validation_sources = set(
        df[df["split"] == "validation"]["source_id"]
    )

    test_sources = set(
        df[df["split"] == "test"]["source_id"]
    )

    results["train_validation"] = find_set_overlap(
        train_sources,
        validation_sources
    )

    results["train_test"] = find_set_overlap(
        train_sources,
        test_sources
    )

    results["validation_test"] = find_set_overlap(
        validation_sources,
        test_sources
    )

    return results


def print_leakage_report(
    subject_results: dict,
    source_results: dict,
) -> bool:

    print("\n================================")
    print("       LEAKAGE REPORT")
    print("================================")

    leakage_found = False

    print("\nSubject leakage:")

    for pair, overlap in subject_results.items():

        if overlap:

            leakage_found = True

            print(
                f"❌ {pair}: "
                f"{len(overlap)} overlapping subjects"
            )

        else:

            print(
                f"✓ {pair}: no leakage"
            )

    print("\nSource leakage:")

    for pair, overlap in source_results.items():

        if overlap:

            leakage_found = True

            print(
                f"❌ {pair}: "
                f"{len(overlap)} overlapping sources"
            )

        else:

            print(
                f"✓ {pair}: no leakage"
            )

    return not leakage_found