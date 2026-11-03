# What: This file implements the logging module in audio/audio-deepfake-detect/src/utils/logging.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

"""Single logging helper shared by every training/eval/infer script.

Wraps Python's stdlib logging plus a TensorBoard SummaryWriter. Both are
returned from get_logger() so callers have one call site to set up
console + file + scalar logging consistently across scripts.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """
    What: This class provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Constructor arguments and class attributes supplied by callers.
    Output: Instances exposing the class attributes and methods.
    """
    def __init__(self, name: str, log_dir: Path):
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self, name, log_dir.
        Output: Side effects performed by the function; no value is explicitly returned.
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(fmt)
            self.logger.addHandler(console)

            file_handler = logging.FileHandler(self.log_dir / "run.log")
            file_handler.setFormatter(fmt)
            self.logger.addHandler(file_handler)

        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

    def info(self, msg: str) -> None:
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self, msg.
        Output: Side effects performed by the function; no value is explicitly returned.
        """
        self.logger.info(msg)

    def scalar(self, tag: str, value: float, step: int) -> None:
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self, tag, value, step.
        Output: Side effects performed by the function; no value is explicitly returned.
        """
        self.writer.add_scalar(tag, value, step)

    def scalars(self, tags_to_values: dict, step: int) -> None:
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self, tags_to_values, step.
        Output: Side effects performed by the function; no value is explicitly returned.
        """
        for tag, value in tags_to_values.items():
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self.
        Output: Side effects performed by the function; no value is explicitly returned.
        """
        self.writer.close()


def get_logger(name: str, log_dir: Optional[Path] = None) -> ExperimentLogger:
    """
    What: This function provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Arguments: name, log_dir.
    Output: The returned value, or side effects performed by the function.
    """
    log_dir = log_dir or Path("outputs") / name
    return ExperimentLogger(name=name, log_dir=Path(log_dir))
