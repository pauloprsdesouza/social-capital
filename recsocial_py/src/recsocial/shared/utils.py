"""Shared utilities."""

from __future__ import annotations

import ast
import re
from typing import Any

import pandas as pd


def parse_mentions(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("{"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (set, list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
    return [text]


def parse_topic(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text.startswith("{"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, set):
                return next(iter(parsed), None)
            if isinstance(parsed, (list, tuple)) and parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass
    return text or None


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_str_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return str(int(value))
    return str(value).strip()


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
