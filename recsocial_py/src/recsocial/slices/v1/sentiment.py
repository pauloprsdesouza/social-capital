"""Sentiment analysis backends (SDD §11)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from recsocial.slices.v1.config import SentimentConfig

SentimentLabel = Literal["positive", "negative", "neutral", "mixed", "unknown"]


@dataclass
class SentimentResult:
    label: SentimentLabel
    score: float | None = None


class SentimentAnalyzer(Protocol):
    def classify(self, text: str) -> SentimentResult: ...


class DummySentimentAnalyzer:
    def classify(self, text: str) -> SentimentResult:
        return SentimentResult(label="neutral", score=0.0)


class VaderSentimentAnalyzer:
    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def classify(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(label="unknown", score=0.0)
        scores = self._analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.35:
            label: SentimentLabel = "positive"
        elif compound <= -0.35:
            label = "negative"
        elif scores["pos"] > 0 and scores["neg"] > 0:
            label = "mixed"
        else:
            label = "neutral"
        return SentimentResult(label=label, score=compound)


class OracleSentimentAnalyzer:
    """Map stored float sentiment scores to labels for validation."""

    def __init__(self, score_by_news_id: dict[str, float]) -> None:
        self._scores = score_by_news_id

    def classify(self, text: str) -> SentimentResult:
        return SentimentResult(label="neutral", score=0.0)

    def classify_news(self, news_id: str) -> SentimentResult:
        score = float(self._scores.get(news_id, 0.0))
        if score >= 0.35:
            label: SentimentLabel = "positive"
        elif score <= -0.35:
            label = "negative"
        elif abs(score) < 0.05:
            label = "neutral"
        else:
            label = "mixed"
        return SentimentResult(label=label, score=score)


def sentiment_weight(label: str, config: SentimentConfig) -> float:
    weights = config.weights.model_dump()
    return float(weights.get(label, weights["unknown"]))


def build_sentiment_analyzer(
    backend: str,
    oracle_scores: dict[str, float] | None = None,
) -> SentimentAnalyzer:
    if backend == "dummy":
        return DummySentimentAnalyzer()
    if backend == "vader":
        return VaderSentimentAnalyzer()
    if backend == "oracle" and oracle_scores is not None:
        return OracleSentimentAnalyzer(oracle_scores)
    return VaderSentimentAnalyzer()
