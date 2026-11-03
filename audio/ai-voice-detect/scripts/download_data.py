# What: This file implements the download data module in audio/ai-voice-detect/scripts/download_data.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

"""Fetch the AI-vs-human voice corpus from the Hugging Face Hub.

garystafford/deepfake-audio-detection, CC-BY-4.0, ungated: 1,866 clips of
16 kHz mono FLAC, 933 human (YouTube) and 933 AI across six TTS platforms.
Already at WavLM's native sample rate, so nothing is resampled on the way in.

Downloads are deliberately single-threaded with retries. The Hub returns 429
on this repo under parallel fan-out, and a rate-limited snapshot_download
exits *successfully* having silently skipped whole directories -- which is
worse than failing, because the next script sees a half-corpus and trains on
it. `verify()` is what makes that impossible.
"""

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "garystafford/deepfake-audio-detection"
EXPECTED = {"real": 933, "fake": 933}


def verify(local_dir: str) -> dict:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: local_dir.
    Output: The returned value, or side effects performed by the function.
    """
    root = Path(local_dir)
    return {name: len(list((root / name).glob("*.flac"))) for name in EXPECTED}


def download(local_dir: str, max_retries: int = 8) -> None:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: local_dir, max_retries.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    for attempt in range(1, max_retries + 1):
        counts = verify(local_dir)
        if all(counts[name] >= EXPECTED[name] for name in EXPECTED):
            print(f"complete: {counts}")
            return

        print(f"attempt {attempt}/{max_retries}  have={counts}", flush=True)
        try:
            snapshot_download(
                REPO_ID,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=["real/*", "fake/*", "README.md"],
                max_workers=1,          # 429s start above this
                resume_download=True,
            )
        except Exception as error:  # noqa: BLE001 - retry transient Hub errors
            wait = min(30, 2**attempt)
            print(f"  {type(error).__name__}: {error}\n  retrying in {wait}s", flush=True)
            time.sleep(wait)

    counts = verify(local_dir)
    if not all(counts[name] >= EXPECTED[name] for name in EXPECTED):
        sys.exit(f"incomplete after {max_retries} attempts: {counts} (expected {EXPECTED})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    download(args.out)
