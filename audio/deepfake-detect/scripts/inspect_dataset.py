# What: This file implements the inspect dataset module in audio/deepfake-detect/scripts/inspect_dataset.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

"""Verifies the real LAV-DF metadata schema before indexing.

Run this immediately after downloading. The upstream schema isn't publicly
documented, so this reports what fields actually exist and whether the
indexer resolved the ones it needs -- catching a mismatch here takes
seconds, whereas a silent mismatch produces an empty manifest that only
surfaces hours later during training.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.lavdf import index_lavdf, inspect_metadata


def main() -> None:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: none.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    parser = argparse.ArgumentParser(description="Inspect LAV-DF metadata schema.")
    parser.add_argument("--root", help="Dataset root containing metadata.json")
    parser.add_argument("--metadata", help="Path directly to metadata.json")
    parser.add_argument("--full-index", action="store_true", help="Also build the manifest")
    args = parser.parse_args()

    metadata_path = args.metadata
    if not metadata_path:
        if not args.root:
            raise SystemExit("Pass --root or --metadata")
        matches = list(Path(args.root).rglob("metadata.json"))
        if not matches:
            raise SystemExit(f"No metadata.json found under {args.root}")
        metadata_path = str(matches[0])

    report = inspect_metadata(metadata_path)

    print(f"metadata.json: {metadata_path}")
    print(f"entries: {report['num_entries']}")
    print(f"\nobserved keys: {report['observed_keys']}")
    print(f"\nresolved fields: {json.dumps(report['resolved'], indent=2)}")

    if report["unresolved"]:
        print(f"\nUNRESOLVED: {report['unresolved']}")
        print(
            "Add the real spelling to FIELD_ALIASES in src/preprocessing/lavdf.py. "
            "Required fields are file, modify_video, modify_audio."
        )
    else:
        print("\nAll fields resolved.")

    print(f"\nexample entry:\n{json.dumps(report.get('example_entry', {}), indent=2)[:1200]}")

    if args.full_index and args.root:
        manifest = index_lavdf(args.root)
        print(f"\nindexed {len(manifest)} rows")
        print(manifest["label"].value_counts().to_string())
        print(manifest["manipulated_modality"].value_counts().to_string())
        print(f"unique identities: {manifest['identity_id'].nunique()}")


if __name__ == "__main__":
    main()
