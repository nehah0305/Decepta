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


def main():

    project_root = Path(__file__).resolve().parents[3]

    dataset_root = (
        project_root
        / "datasets"
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
        / "datasets"
        / "metadata"
        / "faceforensicspp.csv"
    )

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