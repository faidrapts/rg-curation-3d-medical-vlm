"""
Text preprocessing utilities for radiology reports.
"""

from __future__ import annotations

import pandas as pd
import psutil


def truncate_text(text: str, max_tokens: int = 77) -> str:
    """Truncate a string to at most *max_tokens* whitespace-delimited words.

    Word-level truncation is used as a lightweight proxy for token count
    before the actual tokenizer is applied.  CLIP's context window is 77
    tokens (default).  Pass ``max_tokens=4096`` for Clinical Longformer,
    which handles precise truncation internally.

    Empty or ``NaN`` inputs are replaced with a placeholder.  Truncated
    strings are suffixed with ``'...'`` to indicate they were clipped.

    Args:
        text: Input text string.
        max_tokens: Maximum number of words to retain (default: 77).

    Returns:
        Truncated string (with ``'...'`` appended if clipped).
    """
    if not text or pd.isna(text):
        return "No impressions available"

    words = str(text).split()
    if len(words) <= max_tokens:
        return text

    return " ".join(words[:max_tokens]) + "..."


def get_memory_usage() -> float:
    """Return the current process RSS memory usage in megabytes.

    Returns:
        Memory usage in MB (float).
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)
