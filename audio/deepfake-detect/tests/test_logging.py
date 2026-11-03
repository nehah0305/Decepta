# What: This file implements the test logging module in audio/deepfake-detect/tests/test_logging.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

from pathlib import Path

from src.utils.logging import get_logger


def test_get_logger_creates_log_dir(tmp_path: Path):
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: tmp_path.
    Output: Side effects performed by the function; no value is explicitly returned.
    """
    log_dir = tmp_path / "run1"
    logger = get_logger("test_run", log_dir=log_dir)

    logger.info("hello")
    logger.scalar("dummy/metric", 1.0, step=0)
    logger.close()

    assert (log_dir / "run.log").exists()
