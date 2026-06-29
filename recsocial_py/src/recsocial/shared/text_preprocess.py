"""Text preprocessing (SDD §9)."""

from __future__ import annotations

import re
import string

from recsocial.shared.config_models import TextConfig
from recsocial.shared.utils import HASHTAG_PATTERN, MENTION_PATTERN, URL_PATTERN

_STOPWORDS: set[str] | None = None


def _get_stopwords(language: str) -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            import nltk
            from nltk.corpus import stopwords

            nltk.download("stopwords", quiet=True)
            _STOPWORDS = set(stopwords.words(language))
        except Exception:
            _STOPWORDS = set()
    return _STOPWORDS


def preprocess_text(text: str, config: TextConfig) -> str:
    if not text:
        return ""
    out = text
    if config.lowercase:
        out = out.lower()
    if config.remove_urls:
        out = URL_PATTERN.sub(" ", out)
    if config.remove_mentions:
        out = MENTION_PATTERN.sub(" ", out)
    if config.remove_hashtags_symbol:
        out = HASHTAG_PATTERN.sub(r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    if config.remove_stopwords:
        stops = _get_stopwords(config.language)
        tokens = [t for t in out.split() if t not in stops]
        out = " ".join(tokens)
    return out
