"""Influence scores per paper Equations 1–3 and Algorithm 1."""

from __future__ import annotations

import math

import numpy as np


def popularity_score(followers_count: int, lambda_: float = 1.0) -> float:
    return 1.0 - math.exp(-lambda_ * followers_count)


def reputation_score(followers_count: int, lists_count: int) -> float:
    if lists_count != 0:
        return followers_count / lists_count
    return float(followers_count)


def influence_score(
    followers_count: int,
    received_likes_count: int,
    published_news_count: int,
    lists_count: int = 0,
    is_verified: bool = False,
    *,
    mode: str = "paper_pseudocode",
    max_followers: int = 1,
    beta_lists_fallback: float = 1.0,
    verified_bonus_theta: float = 1.0,
    lambda_: float = 1.0,
) -> float:
    ts = followers_count
    tls = lists_count
    tc = received_likes_count
    tnp = published_news_count

    if mode == "paper_pseudocode":
        if ts == 0:
            ts = max_followers
        if tls == 0:
            tls = beta_lists_fallback
        rep = reputation_score(ts, tls)
        if is_verified:
            rep += verified_bonus_theta
    else:
        rep = reputation_score(ts, tls)

    if rep == 0:
        return 0.0

    pscore = popularity_score(ts, lambda_=lambda_)
    return (pscore + tc + tnp) / rep
