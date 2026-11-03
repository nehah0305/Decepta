# What: This file implements the   init   module in backend/model/utils/__init__.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

"""
Utilities Module for Visual Deepfake Detection System.
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .logging import FrameCoverageReport, save_frame_metadata_csv_json

__all__ = [
    "FrameCoverageReport",
    "save_frame_metadata_csv_json",
    "save_checkpoint",
    "load_checkpoint",
]
