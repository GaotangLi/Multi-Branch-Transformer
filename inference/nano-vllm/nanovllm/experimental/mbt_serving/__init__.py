# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Version-2, single-GPU inference for Standard and 2–4-branch MBT exports.

The legacy ``wide_serving`` TP engine remains unchanged. Import the config
without importing CUDA, Triton, or model weights.
"""

from .config import MBTConfig

__all__ = ["MBTConfig"]
