from pathlib import Path
import pandas as pd


class FaceForensicsAdapter:
    """
    Converts FaceForensics++ C23 metadata into
    our project's standard manifest format.
    """

    def __init__(self, dataset_root, metadata_file):

        self.dataset_root = Path(dataset_root)
        self.metadata_file = Path(metadata_file)

    def load_metadata(self):

        if not self.metadata_file.exists():

            raise FileNotFoundError(
                f"Metadata file not found:\n"
                f"{self.metadata_file}"
            )

        df = pd.read_csv(self.metadata_file)

        print("\nMetadata loaded successfully.")

        print(f"Number of records: {len(df)}")

        print("\nColumns:")
        print(df.columns.tolist())

        return df

    def create_manifest(self):

        df = self.load_metadata()

        required_columns = [
            "File Path",
            "Label",
            "Frame Count",
            "Width",
            "Height",
            "Codec",
            "File Size(MB)",
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        manifest = pd.DataFrame()

        # --------------------------------------------------
        # Basic identifiers
        # --------------------------------------------------

        manifest["sample_id"] = [
            f"FFPP_{i:07d}"
            for i in range(len(df))
        ]

        manifest["dataset"] = "FaceForensics++"

        # --------------------------------------------------
        # Video path
        # --------------------------------------------------

        manifest["video_path"] = df[
            "File Path"
        ].astype(str)

        # --------------------------------------------------
        # Label
        # REAL = 0
        # FAKE = 1
        # --------------------------------------------------

        manifest["label"] = (
            df["Label"]
            .astype(str)
            .str.upper()
            .map({
                "REAL": 0,
                "FAKE": 1,
            })
        )

        # --------------------------------------------------
        # Video metadata
        # --------------------------------------------------

        manifest["frame_count"] = df[
            "Frame Count"
        ]

        manifest["width"] = df[
            "Width"
        ]

        manifest["height"] = df[
            "Height"
        ]

        manifest["codec"] = df[
            "Codec"
        ]

        manifest["file_size_mb"] = df[
            "File Size(MB)"
        ]

        # --------------------------------------------------
        # Determine manipulation type
        # --------------------------------------------------

        manifest["manipulation"] = (
            manifest["video_path"]
            .apply(self.detect_manipulation)
        )

        # --------------------------------------------------
        # Generator
        # --------------------------------------------------

        manifest["generator"] = (
            manifest["manipulation"]
        )

        manifest.loc[
            manifest["label"] == 0,
            "generator"
        ] = "None"

        # --------------------------------------------------
        # These will be populated later
        # after we inspect FF++ relationships.
        # --------------------------------------------------

        manifest["subject_id"] = None

        manifest["source_id"] = None

        manifest["original_video_id"] = None

        provenance = manifest.apply(
            lambda row: self.infer_provenance(
                row["video_path"],
                row["manipulation"],
            ),
            axis=1,
        )

        manifest["source_id"] = provenance.apply(
            lambda value: value[0]
        )

        manifest["original_video_id"] = provenance.apply(
            lambda value: value[1]
        )

        pair_groups = {}

        for source_id, original_video_id in provenance:
            if (
                source_id
                and original_video_id
                and source_id != original_video_id
                and source_id.startswith("ORIG_")
                and original_video_id.startswith("ORIG_")
            ):
                pair = tuple(sorted((source_id, original_video_id)))
                group_id = f"PAIR_{pair[0]}_{pair[1]}"
                pair_groups[source_id] = group_id
                pair_groups[original_video_id] = group_id

        manifest["split_group_id"] = provenance.apply(
            lambda value: pair_groups.get(
                value[1] or value[0],
                value[0],
            )
        )

        manifest["audio_path"] = None

        manifest["split"] = None

        return manifest

    @staticmethod
    def detect_manipulation(video_path):

        path = video_path.lower()

        if path.startswith("original/"):
            return "Original"

        if path.startswith("deepfakedetection/"):
            return "DeepFakeDetection"

        if path.startswith("deepfakes/"):
            return "Deepfakes"

        if path.startswith("face2face/"):
            return "Face2Face"

        if path.startswith("faceshifter/"):
            return "FaceShifter"

        if path.startswith("faceswap/"):
            return "FaceSwap"

        if path.startswith("neuraltextures/"):
            return "NeuralTextures"

        return "Unknown"

    @staticmethod
    def infer_provenance(video_path, manipulation):

        path = Path(video_path)
        stem = path.stem

        if manipulation == "Original" and stem.isdigit():
            original_id = f"ORIG_{int(stem):03d}"
            return original_id, original_id

        if manipulation in {
            "Deepfakes",
            "Face2Face",
            "FaceShifter",
            "FaceSwap",
            "NeuralTextures",
        }:
            parts = stem.split("_")

            if len(parts) == 2 and all(part.isdigit() for part in parts):
                original_id, source_id = parts
                return (
                    f"ORIG_{int(source_id):03d}",
                    f"ORIG_{int(original_id):03d}",
                )

        if manipulation == "DeepFakeDetection":
            parts = stem.split("__")

            if len(parts) == 3 and parts[2].isalnum():
                return f"DFD_{parts[2]}", None

        return None, None


def main():

    project_root = Path(__file__).resolve().parents[3]

    dataset_root = (
        project_root
        / "raw"
        / "faceforensicspp"
        / "c23"
    )

    metadata_file = (
        dataset_root
        / "csv"
        / "FF++_Metadata.csv"
    )

    adapter = FaceForensicsAdapter(
        dataset_root=dataset_root,
        metadata_file=metadata_file,
    )

    manifest = adapter.create_manifest()

    output_path = (
        project_root
        / "metadata"
        / "faceforensicspp.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(
        output_path,
        index=False
    )

    print("\n======================================")
    print("FaceForensics++ manifest created")
    print("======================================")

    print(f"\nOutput:")
    print(output_path)

    print("\nSamples:")
    print(len(manifest))

    print("\nLabels:")
    print(
        manifest["label"]
        .value_counts(dropna=False)
    )

    print("\nManipulations:")
    print(
        manifest["manipulation"]
        .value_counts(dropna=False)
    )

    print("\nFirst 5 rows:")
    print(
        manifest.head().to_string()
    )


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()