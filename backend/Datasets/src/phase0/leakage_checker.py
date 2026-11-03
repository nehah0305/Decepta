# What: This file implements the leakage checker module in backend/Datasets/src/phase0/leakage_checker.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import pandas as pd


def find_set_overlap(
    first_set: set,
    second_set: set,
) -> set:

    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: first_set, second_set.
    Output: The returned value, or side effects performed by the function.
    """
    return first_set.intersection(second_set)


def check_subject_leakage(
    df: pd.DataFrame,
) -> dict:

    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: df.
    Output: The returned value, or side effects performed by the function.
    """
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

    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: df.
    Output: The returned value, or side effects performed by the function.
    """
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


def check_split_group_leakage(
    df: pd.DataFrame,
) -> dict:

    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: df.
    Output: The returned value, or side effects performed by the function.
    """
    split_groups = {
        split: set(
            df[df["split"] == split]["split_group_id"]
        )
        for split in ("train", "validation", "test")
    }

    return {
        "train_validation": find_set_overlap(
            split_groups["train"],
            split_groups["validation"],
        ),
        "train_test": find_set_overlap(
            split_groups["train"],
            split_groups["test"],
        ),
        "validation_test": find_set_overlap(
            split_groups["validation"],
            split_groups["test"],
        ),
    }


def print_leakage_report(
    subject_results: dict,
    source_results: dict,
) -> bool:

    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: subject_results, source_results.
    Output: The returned value, or side effects performed by the function.
    """
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