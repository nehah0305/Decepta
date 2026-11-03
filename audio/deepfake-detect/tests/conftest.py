# What: This file implements the conftest module in audio/deepfake-detect/tests/conftest.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
