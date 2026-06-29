"""Shared module tests."""

from __future__ import annotations

import pandas as pd

from recsocial.shared.parsing import collection_len, parse_collection
from recsocial.shared.reranking import build_trial_rerank_bundle, rerank_by_score
from recsocial.shared.session_metrics import (
    EvaluationSettings,
    evaluate_recommendations_by_session,
)


def test_parse_collection_set_literal():
    assert set(parse_collection("{'a', 'b'}")) == {"a", "b"}


def test_collection_len_empty():
    assert collection_len(None) == 0


def test_rerank_by_score():
    trial = pd.DataFrame({"news_id": ["1", "2"], "user_id": [1, 1], "rating": [3, 5]})
    features = pd.DataFrame({"news_id": ["1", "2"], "score": [0.2, 0.8]})
    out = rerank_by_score(trial, features, "score", top_k=2)
    assert out.iloc[0]["news_id"] == "2"


def test_evaluate_recommendations_by_session():
    recs = pd.DataFrame({
        "user_id": [1, 1],
        "algorithm": ["A", "A"],
        "news_id": ["x", "y"],
        "ranking": [1, 2],
        "rating": [5, 2],
    })
    settings = EvaluationSettings(relevance_threshold=4, precision_k_values=(1, 2))
    detail = evaluate_recommendations_by_session(recs, settings)
    assert detail.iloc[0]["mrr"] == 1.0
    assert detail.iloc[0]["precision_at_1"] == 1.0


def test_build_trial_rerank_bundle_state_art():
    ratings = pd.DataFrame({
        "user_id": [1, 1],
        "algorithm": ["B1", "B1"],
        "news_id": ["a", "b"],
        "position": [1, 2],
        "rating": [5, 3],
    })
    features = pd.DataFrame({
        "news_id": ["a", "b"],
        "state_art_score": [0.1, 0.9],
        "scsa_plus_score": [0.5, 0.5],
    })
    out = build_trial_rerank_bundle(
        ratings,
        features,
        base_algorithms=["B1"],
        rerank_suffixes={"state_art": "STATE_ART", "scsa_plus": "SCSA_PLUS"},
        scsa_score_specs=[("scsa_plus", "scsa_plus_score")],
    )
    assert set(out["algorithm"]) == {"B1-STATE_ART", "B1-SCSA_PLUS"}
