"""Parse legacy set/list string fields from CSV exports."""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd


def parse_collection(value: Any, *, as_strings: bool = True) -> list:
    """Parse `{a, b}` / `[a, b]` literals or pipe-separated strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (set, list, tuple)):
                items = list(parsed)
                return [str(x) for x in items] if as_strings else items
        except (ValueError, SyntaxError):
            pass
    if "|" in text:
        return [p for p in text.split("|") if p]
    return [text] if text else []


def collection_len(value: Any) -> int:
    return len(parse_collection(value))
