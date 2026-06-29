# Software Design Document — SCSA_PLUS_V3 Reproduction System

## 1. Document Title

**Software Design Document for Reproducing “Unlocking the Power of Social Capital: Advanced Strategies for Enhanced Personalized Recommendations in Online Social Networks”**

---

## 2. Document Purpose

This Software Design Document defines a Python implementation to reproduce and extend the recommender system proposed in the AMCIS 2024 paper **“Unlocking the Power of Social Capital: Advanced Strategies for Enhanced Personalized Recommendations in Online Social Networks.”**

The implementation focuses on reproducing the paper’s offline evaluation and its main proposal, here referred to as **SCSA_PLUS_V3**, using the extracted chart values from Figures 3–10 as reference targets.

The system shall support:

1. Loading Twitter/X-like social-network data.
2. Preprocessing tweet text and metadata.
3. Engineering tweet and user features.
4. Computing Social Capital components.
5. Ranking candidate tweets/news.
6. Comparing SCSA_PLUS_V3 against previous and baseline models.
7. Reproducing reported MRR, MAP@10, NDCG@10, and Precision@N values.
8. Generating structured reports and validation outputs.

---

## 3. Problem Statement

Online social networks generate massive amounts of content. Users follow accounts for many reasons, but not all posts from those accounts are relevant to their interests. Traditional recommender systems based only on content similarity or collaborative filtering may not fully capture the social dynamics that influence relevance.

The article proposes a Social Capital-based recommendation model that uses the social value of tweets/news to improve personalized recommendations. The model considers multiple dimensions of social interaction, including:

* Sentiment impact.
* Engagement.
* Content relevance.
* Network influence.
* Author influence.
* Content virality.
* Recency.
* Diversity.
* Context.

The goal of this implementation is to reproduce the proposed approach and evaluate it against baselines using the same ranking metrics reported in the article.

---

## 4. System Name

Recommended project name:

```text
social-capital-recsys-v3
```

Recommended Python package name:

```text
sc_recsys_v3
```

---

## 5. Scope

## 5.1 In Scope

The implementation shall include:

* Dataset loading from CSV, Parquet, or JSONL.
* Tweet/news preprocessing.
* Sentiment analysis.
* Text feature extraction using TF-IDF.
* Optional dimensionality reduction.
* Social Capital component calculation.
* Recommendation ranking.
* Offline evaluation.
* Chart-value reproduction.
* Algorithmic reproduction.
* Ablation studies.
* Report generation.

The implementation shall reproduce or approximate:

* SCSA_PLUS_V3.
* SCSA_PLUS.
* STATE_ART.
* B1.
* CS.
* SC.
* SCSA.

---

## 5.2 Out of Scope for Version 1

The first version shall not include:

* Live Twitter/X API integration.
* Production serving.
* Real-time user feedback.
* Online A/B testing.
* Click-through-rate tracking.
* Dwell-time tracking.
* Community-detection ranking.
* Misinformation detection.
* Bot detection.
* Multi-platform social-network support.

These are future extensions.

---

## 6. Architectural Overview

The system shall follow this pipeline:

```text
Raw Twitter/X-like data
   ↓
Data validation
   ↓
Feature engineering
   ↓
Social Capital component scoring
   ↓
Recommendation model
   ↓
Ranking by Social Capital
   ↓
Offline evaluation
   ↓
Reference-result validation
   ↓
Reproduction report
```

---

## 7. Recommended Repository Structure

```text
social-capital-recsys-v3/
  pyproject.toml
  README.md

  configs/
    default.yaml
    reproduction.yaml
    ablation.yaml
    reference_results.yaml

  data/
    raw/
    processed/
    interim/

  reports/
    tables/
    figures/
    reproduction_report.md
    reproduction_report.html

  notebooks/
    01_dataset_exploration.ipynb
    02_feature_engineering.ipynb
    03_scoring_validation.ipynb
    04_reproduce_figures_3_to_10.ipynb
    05_ablation_analysis.ipynb

  src/
    sc_recsys_v3/
      __init__.py
      config.py
      schemas.py
      data_loader.py
      preprocessing.py
      text_features.py
      sentiment.py
      engagement.py
      content_relevance.py
      network_influence.py
      author_influence.py
      content_virality.py
      recency.py
      diversity.py
      context.py
      social_capital.py
      recommenders.py
      baselines.py
      evaluation.py
      reference_results.py
      validation.py
      ablation.py
      experiment.py
      reporting.py
      utils.py

  tests/
    test_preprocessing.py
    test_sentiment.py
    test_engagement.py
    test_content_relevance.py
    test_network_influence.py
    test_author_influence.py
    test_content_virality.py
    test_social_capital.py
    test_recommenders.py
    test_metrics.py
    test_reference_results.py
    test_validation.py
```

---

## 8. Technology Stack

## 8.1 Required Python Libraries

```text
python >= 3.11
pandas
numpy
scipy
scikit-learn
pydantic
pyyaml
matplotlib
tqdm
```

## 8.2 NLP Libraries

```text
nltk
spacy
regex
emoji
unidecode
```

## 8.3 Optional NLP Libraries

```text
transformers
sentence-transformers
vaderSentiment
torch
```

## 8.4 Testing and Quality

```text
pytest
pytest-cov
ruff
mypy
black
```

---

## 9. Domain Model

## 9.1 Tweet

```python
class Tweet(BaseModel):
    tweet_id: str
    author_id: str
    text: str

    created_at: datetime | None = None

    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    impression_count: int = 0

    mentions: list[str] = []
    urls: list[str] = []
    hashtags: list[str] = []

    media_count: int = 0
    has_image: bool = False
    has_video: bool = False

    topic: str | None = None
    subtopic: str | None = None
```

---

## 9.2 User

```python
class User(BaseModel):
    user_id: str
    username: str | None = None

    followers_count: int = 0
    following_count: int = 0
    listed_count: int = 0
    tweet_count: int = 0

    verified: bool = False
```

---

## 9.3 Rating

```python
class Rating(BaseModel):
    user_id: str
    tweet_id: str
    algorithm: str
    rating: int
    round_id: int | None = None
    position: int | None = None
```

---

## 9.4 Recommendation Record

```python
class RecommendationRecord(BaseModel):
    user_id: str
    tweet_id: str
    algorithm: str
    rank: int
    score: float

    sentiment_impact: float | None = None
    engagement_score: float | None = None
    content_relevance: float | None = None
    network_influence: float | None = None
    author_influence: float | None = None
    content_virality: float | None = None
    recency_score: float | None = None
    diversity_score: float | None = None
    context_score: float | None = None
```

---

## 9.5 Social Capital Breakdown

```python
class SocialCapitalBreakdown(BaseModel):
    tweet_id: str

    sentiment_impact: float
    engagement_score: float
    content_relevance: float
    network_influence: float
    author_influence: float
    content_virality: float

    recency_score: float | None = None
    diversity_score: float | None = None
    context_score: float | None = None

    social_capital_score: float
```

---

## 10. Dataset Contract

The system shall support the following canonical input files.

## 10.1 `tweets.csv`

```text
tweet_id,
author_id,
text,
created_at,
like_count,
retweet_count,
reply_count,
quote_count,
impression_count,
topic,
subtopic,
media_count,
has_image,
has_video
```

## 10.2 `users.csv`

```text
user_id,
username,
followers_count,
following_count,
listed_count,
tweet_count,
verified
```

## 10.3 `tweet_mentions.csv`

```text
tweet_id,
mentioned_user_id
```

## 10.4 `tweet_urls.csv`

```text
tweet_id,
url
```

## 10.5 `tweet_hashtags.csv`

```text
tweet_id,
hashtag
```

## 10.6 `ratings.csv`

```text
user_id,
tweet_id,
algorithm,
rating,
round_id,
position
```

## 10.7 `candidate_sets.csv`

```text
user_id,
round_id,
tweet_id
```

This file is optional, but strongly recommended for exact reproduction.

---

## 11. Feature Engineering Pipeline

## 11.1 Text Cleaning

The system shall clean tweet text using the following steps:

1. Normalize Unicode.
2. Convert text to lowercase.
3. Remove URLs from text.
4. Remove mentions from text.
5. Normalize hashtags.
6. Remove punctuation.
7. Remove repeated whitespace.

```python
def clean_text(text: str, config: TextConfig) -> str:
    text = normalize_unicode(text)

    if config.lowercase:
        text = text.lower()

    if config.remove_urls:
        text = remove_urls(text)

    if config.remove_mentions:
        text = remove_mentions(text)

    if config.remove_hashtag_symbol:
        text = normalize_hashtags(text)

    if config.remove_punctuation:
        text = remove_punctuation(text)

    return normalize_whitespace(text)
```

---

## 11.2 Tokenization

```python
def tokenize(text: str) -> list[str]:
    return nltk.word_tokenize(text)
```

---

## 11.3 Stopword Removal

```python
def remove_stopwords(tokens: list[str], language: str = "english") -> list[str]:
    stop_words = set(stopwords.words(language))
    return [token for token in tokens if token not in stop_words]
```

---

## 11.4 Lemmatization

```python
def lemmatize(tokens: list[str]) -> list[str]:
    return [lemmatizer.lemmatize(token) for token in tokens]
```

---

## 11.5 Disambiguation

```python
def disambiguate(tokens: list[str], context: str) -> list[str]:
    ...
```

The first implementation may use a no-op disambiguation function if WordNet-based disambiguation is not yet stable.

---

## 11.6 Synonym Expansion

```python
def expand_synonyms(tokens: list[str]) -> list[str]:
    ...
```

This feature should be configurable because synonym expansion may introduce noise.

---

## 11.7 TF-IDF Vectorization

```python
def build_tfidf(corpus: list[str], config: TextConfig):
    vectorizer = TfidfVectorizer(
        min_df=config.min_df,
        max_df=config.max_df,
        ngram_range=tuple(config.ngram_range)
    )
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer
```

---

## 11.8 Dimensionality Reduction

Although the paper mentions PCA, the Python implementation should use TruncatedSVD for sparse TF-IDF matrices.

```python
def reduce_dimensions(matrix, n_components: int = 100):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(matrix)
    return reduced, svd
```

---

## 12. Social Capital Components

## 12.1 Sentiment Impact — SI

Sentiment Impact measures the polarity or emotional impact of tweet text.

```python
def sentiment_impact(text: str, analyzer: SentimentAnalyzer) -> float:
    sentiment = analyzer.score(text)
    return normalize_sentiment_to_0_1(sentiment)
```

Suggested normalization:

```text
compound score range: [-1, 1]
normalized score = (compound + 1) / 2
```

Suggested categorical mapping:

```text
positive = 1.00
neutral  = 0.50
mixed    = 0.50
negative = 0.00
unknown  = 0.50
```

---

## 12.2 Engagement Score — ES

Engagement Score measures direct interaction with the tweet.

```python
def engagement_score(tweet: Tweet, scaler: FeatureScaler) -> float:
    raw = np.mean([
        tweet.like_count,
        tweet.retweet_count,
        tweet.reply_count,
        tweet.quote_count,
    ])
    return scaler.transform_value("engagement_score", raw)
```

Alternative raw mode:

```python
ES = mean(like_count, retweet_count, reply_count, quote_count)
```

---

## 12.3 Content Relevance — CR

Content Relevance measures the similarity between the tweet content and the active user profile or target topic.

```python
def content_relevance(
    tweet_vector,
    user_profile_vector,
) -> float:
    if user_profile_vector is None:
        return 0.0

    return cosine_similarity(tweet_vector, user_profile_vector)
```

Cold-start fallback:

```text
CR = normalized mean TF-IDF value of tweet
```

---

## 12.4 Network Influence — NI

Network Influence captures tweet connectivity using mentions and URLs.

```python
def network_influence(tweet: Tweet, scaler: FeatureScaler) -> float:
    raw = len(tweet.mentions) + len(tweet.urls)
    return scaler.transform_value("network_influence", raw)
```

---

## 12.5 Author Influence — AI

The paper-compatible mode treats Author Influence as a count-based signal.

```python
def author_influence_paper_compatible(tweet: Tweet, scaler: FeatureScaler) -> float:
    raw = len(tweet.mentions)
    return scaler.transform_value("author_influence", raw)
```

The research mode may use an explicit user-strength score.

```python
def author_influence_user_strength(author_id: str, user_strengths: dict[str, float]) -> float:
    return user_strengths.get(author_id, 0.0)
```

---

## 12.6 Content Virality — CV

Content Virality measures propagation through retweets and quotes.

```python
def content_virality(tweet: Tweet, scaler: FeatureScaler) -> float:
    raw = tweet.retweet_count + tweet.quote_count
    return scaler.transform_value("content_virality", raw)
```

---

## 12.7 Recency Score

The 2024 article uses an exponential recency score.

```text
RecencyScore(t) = exp(-lambda_decay * elapsed_days)
```

Default:

```text
lambda_decay = 0.03
```

Python implementation:

```python
def recency_score(
    created_at: datetime,
    max_created_at: datetime,
    lambda_decay: float = 0.03,
) -> float:
    elapsed_days = max((max_created_at - created_at).days, 0)
    return math.exp(-lambda_decay * elapsed_days)
```

---

## 12.8 Diversity Score

Diversity may be computed as lexical diversity or as a ranking-level dissimilarity score.

Paper-compatible default:

```python
def lexical_diversity(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)
```

Research mode:

```python
def ranking_diversity(candidate_vector, selected_vectors) -> float:
    if not selected_vectors:
        return 1.0

    similarities = cosine_similarity(candidate_vector, selected_vectors)
    return 1 - float(np.mean(similarities))
```

---

## 12.9 Context Score

Context Score measures topic alignment.

```python
def context_score(
    tweet_text: str,
    topic_keyword_documents: list[str],
    vectorizer: TfidfVectorizer,
) -> float:
    docs = [tweet_text] + topic_keyword_documents
    matrix = vectorizer.fit_transform(docs)

    tweet_vector = matrix[0]
    topic_vectors = matrix[1:]

    similarities = cosine_similarity(tweet_vector, topic_vectors)[0]

    if len(similarities) == 0:
        return 0.0

    return float(similarities.max())
```

---

## 13. Main Social Capital Formula

## 13.1 Paper-Compatible Formula

```text
SC = α * SI + β * ES + γ * CR + σ * NI + ε * AI + ζ * CV
```

Default weights:

```text
SI = 0.20
ES = 0.20
CR = 0.20
NI = 0.20
AI = 0.10
CV = 0.10
```

The weights must sum to 1.0.

---

## 13.2 Extended Research Formula

```text
SC_EXT =
    α * SI
  + β * ES
  + γ * CR
  + σ * NI
  + ε * AI
  + ζ * CV
  + ρ * RecencyScore
  + δ * DiversityScore
  + κ * ContextScore
```

This mode is not the primary reproduction mode. It is intended for future experiments.

---

## 13.3 Python Scorer

```python
class SocialCapitalScorer:
    def __init__(
        self,
        config: SocialCapitalConfig,
        sentiment_analyzer: SentimentAnalyzer,
        feature_scaler: FeatureScaler,
        vectorizer: TfidfVectorizer,
    ):
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.feature_scaler = feature_scaler
        self.vectorizer = vectorizer

    def score(self, tweet: Tweet, user_profile_vector=None) -> SocialCapitalBreakdown:
        si = sentiment_impact(tweet.text, self.sentiment_analyzer)
        es = engagement_score(tweet, self.feature_scaler)
        cr = content_relevance(tweet.vector, user_profile_vector)
        ni = network_influence(tweet, self.feature_scaler)
        ai = author_influence_paper_compatible(tweet, self.feature_scaler)
        cv = content_virality(tweet, self.feature_scaler)

        weights = self.config.weights

        score = (
            weights.sentiment_impact * si
            + weights.engagement_score * es
            + weights.content_relevance * cr
            + weights.network_influence * ni
            + weights.author_influence * ai
            + weights.content_virality * cv
        )

        return SocialCapitalBreakdown(
            tweet_id=tweet.tweet_id,
            sentiment_impact=si,
            engagement_score=es,
            content_relevance=cr,
            network_influence=ni,
            author_influence=ai,
            content_virality=cv,
            social_capital_score=score,
        )
```

---

## 14. Recommendation Algorithms

## 14.1 SCSA_PLUS_V3

This is the paper’s proposed model.

```python
score = social_capital_score(tweet, user)
```

Rank descending by `score`.

---

## 14.2 SCSA_PLUS

Represents the previous Social Capital with Sentiment Analysis approach.

```python
score = previous_social_capital_score(tweet, user) * sentiment_weight(tweet.text)
```

---

## 14.3 STATE_ART

Represents the state-of-the-art baseline used in the article.

Preferred reproduction mode:

```text
Use imported baseline rankings from the original dataset, if available.
```

Fallback approximation:

```text
STATE_ART = content_relevance + normalized_engagement + recency
```

---

## 14.4 B1

Represents the B1 baseline.

Preferred reproduction mode:

```text
Use imported B1 ranking from original evaluated results, if available.
```

Fallback approximation:

```text
B1 = normalized(like_count + retweet_count + reply_count + quote_count)
```

---

## 14.5 CS

Cosine Similarity baseline.

```python
score = cosine_similarity(user_profile_vector, tweet_vector)
```

---

## 14.6 SC

Social Capital baseline from previous work.

```text
SC = interaction-oriented social capital without full V3 component set
```

---

## 14.7 SCSA

Social Capital with Sentiment Analysis baseline.

```text
SCSA = SC + sentiment analysis
```

---

## 15. Recommender Interface

```python
class Recommender:
    def recommend(
        self,
        user_id: str,
        candidate_tweets: list[Tweet],
        algorithm: str,
        top_k: int = 10,
    ) -> list[RecommendationRecord]:
        ...
```

Supported algorithms:

```text
SCSA_PLUS_V3
SCSA_PLUS
STATE_ART
B1
CS
SC
SCSA
```

---

## 16. Offline Evaluation Design

## 16.1 Experiment Flow

```python
for user_id in users:
    candidates = get_candidate_tweets(user_id)

    for algorithm in algorithms:
        ranked = recommender.recommend(
            user_id=user_id,
            candidate_tweets=candidates,
            algorithm=algorithm,
            top_k=10,
        )

        metrics = evaluator.evaluate(
            ranked_items=ranked,
            ground_truth=ratings,
        )

        save_metrics(user_id, algorithm, metrics)
```

---

## 16.2 Relevance Threshold

```text
relevant = rating >= 4
```

---

## 16.3 Required Metrics

```text
MRR
MAP@10
NDCG@10
Precision@1
Precision@2
Precision@3
Precision@4
Precision@5
```

---

## 17. Metric Definitions

## 17.1 MRR

```python
def mrr(ranked_ids: list[str], ratings: dict[str, int], threshold: int = 4) -> float:
    for idx, item_id in enumerate(ranked_ids, start=1):
        if ratings.get(item_id, 0) >= threshold:
            return 1 / idx
    return 0.0
```

---

## 17.2 Precision@K

```python
def precision_at_k(
    ranked_ids: list[str],
    ratings: dict[str, int],
    k: int,
    threshold: int = 4,
) -> float:
    top_k = ranked_ids[:k]
    relevant = sum(1 for item_id in top_k if ratings.get(item_id, 0) >= threshold)
    return relevant / k
```

---

## 17.3 MAP@K

```python
def average_precision_at_k(
    ranked_ids: list[str],
    ratings: dict[str, int],
    k: int,
    threshold: int = 4,
) -> float:
    hits = 0
    precision_sum = 0.0

    for idx, item_id in enumerate(ranked_ids[:k], start=1):
        if ratings.get(item_id, 0) >= threshold:
            hits += 1
            precision_sum += hits / idx

    if hits == 0:
        return 0.0

    return precision_sum / hits
```

---

## 17.4 NDCG@K

```python
def ndcg_at_k(
    ranked_ids: list[str],
    ratings: dict[str, int],
    k: int,
) -> float:
    def dcg(items: list[str]) -> float:
        return sum(
            ratings.get(item_id, 0) / math.log2(idx + 1)
            for idx, item_id in enumerate(items[:k], start=1)
        )

    ideal_ids = sorted(
        ranked_ids,
        key=lambda item_id: ratings.get(item_id, 0),
        reverse=True,
    )

    ideal = dcg(ideal_ids)

    if ideal == 0:
        return 0.0

    return dcg(ranked_ids) / ideal
```

---

## 18. Reference Result Values Extracted from Figures 3–10

The following values are the structured reproduction targets extracted from the result charts.

---

## 18.1 Figure 3 — B1 Comparison

| Algorithm Variant |   MRR | MAP@10 | NDCG@10 |
| ----------------- | ----: | -----: | ------: |
| B1-SCSA_PLUS      | 0.793 |  0.777 |   0.788 |
| B1-STATE_ART      | 0.710 |  0.434 |   0.813 |
| SCSA_PLUS_V3      | 0.665 |  0.454 |   0.659 |

Winner summary:

```text
MRR winner: B1-SCSA_PLUS
MAP@10 winner: B1-SCSA_PLUS
NDCG@10 winner: B1-STATE_ART
```

---

## 18.2 Figure 4 — CS Comparison

| Algorithm Variant |   MRR | MAP@10 | NDCG@10 |
| ----------------- | ----: | -----: | ------: |
| CS-SCSA_PLUS      | 0.641 |  0.669 |   0.702 |
| CS-STATE_ART      | 0.680 |  0.415 |   0.817 |
| SCSA_PLUS_V3      | 0.630 |  0.440 |   0.660 |

Winner summary:

```text
MRR winner: CS-STATE_ART
MAP@10 winner: CS-SCSA_PLUS
NDCG@10 winner: CS-STATE_ART
```

---

## 18.3 Figure 5 — SC Comparison

| Algorithm Variant |   MRR | MAP@10 | NDCG@10 |
| ----------------- | ----: | -----: | ------: |
| SC-SCSA_PLUS      | 0.778 |  0.786 |   0.792 |
| SC-STATE_ART      | 0.739 |  0.508 |   0.836 |
| SCSA_PLUS_V3      | 0.695 |  0.505 |   0.676 |

Winner summary:

```text
MRR winner: SC-SCSA_PLUS
MAP@10 winner: SC-SCSA_PLUS
NDCG@10 winner: SC-STATE_ART
```

---

## 18.4 Figure 6 — SCSA Comparison

| Algorithm Variant |   MRR | MAP@10 | NDCG@10 |
| ----------------- | ----: | -----: | ------: |
| SCSA-SCSA_PLUS    | 0.756 |  0.735 |   0.753 |
| SCSA-STATE_ART    | 0.704 |  0.411 |   0.811 |
| SCSA_PLUS_V3      | 0.665 |  0.427 |   0.656 |

Winner summary:

```text
MRR winner: SCSA-SCSA_PLUS
MAP@10 winner: SCSA-SCSA_PLUS
NDCG@10 winner: SCSA-STATE_ART
```

---

## 18.5 Figure 7 — B1 Precision Comparison

| Algorithm Variant |   P@1 |   P@2 |   P@3 |   P@4 |   P@5 |
| ----------------- | ----: | ----: | ----: | ----: | ----: |
| B1-SCSA_PLUS      | 0.592 | 0.634 | 0.624 | 0.623 | 0.606 |
| B1-STATE_ART      | 0.570 | 0.549 | 0.560 | 0.570 | 0.560 |
| SCSA_PLUS_V3      | 0.577 | 0.577 | 0.558 | 0.549 | 0.557 |

Winner summary:

```text
P@1 winner: B1-SCSA_PLUS
P@2 winner: B1-SCSA_PLUS
P@3 winner: B1-SCSA_PLUS
P@4 winner: B1-SCSA_PLUS
P@5 winner: B1-SCSA_PLUS
```

---

## 18.6 Figure 8 — CS Precision Comparison

| Algorithm Variant |   P@1 |   P@2 |   P@3 |   P@4 |   P@5 |
| ----------------- | ----: | ----: | ----: | ----: | ----: |
| CS-SCSA_PLUS      | 0.408 | 0.437 | 0.474 | 0.496 | 0.518 |
| CS-STATE_ART      | 0.530 | 0.520 | 0.563 | 0.556 | 0.567 |
| SCSA_PLUS_V3      | 0.535 | 0.535 | 0.053 | 0.528 | 0.549 |

Winner summary:

```text
P@1 winner: SCSA_PLUS_V3
P@2 winner: SCSA_PLUS_V3
P@3 winner: CS-STATE_ART
P@4 winner: CS-STATE_ART
P@5 winner: CS-STATE_ART
```

Important anomaly:

```text
SCSA_PLUS_V3 at P@3 = 0.053
```

This value appears visually anomalous, but it must be preserved in chart-label reproduction mode.

---

## 18.7 Figure 9 — SC Precision Comparison

| Algorithm Variant |   P@1 |   P@2 |   P@3 |   P@4 |   P@5 |
| ----------------- | ----: | ----: | ----: | ----: | ----: |
| SC-SCSA_PLUS      | 0.630 | 0.630 | 0.607 | 0.620 | 0.638 |
| SC-STATE_ART      | 0.616 | 0.610 | 0.602 | 0.599 | 0.594 |
| SCSA_PLUS_V3      | 0.616 | 0.609 | 0.611 | 0.609 | 0.605 |

Winner summary:

```text
P@1 winner: SC-SCSA_PLUS
P@2 winner: SC-SCSA_PLUS
P@3 winner: SCSA_PLUS_V3
P@4 winner: SC-SCSA_PLUS
P@5 winner: SC-SCSA_PLUS
```

---

## 18.8 Figure 10 — SCSA Precision Comparison

| Algorithm Variant |   P@1 |   P@2 |   P@3 |   P@4 |   P@5 |
| ----------------- | ----: | ----: | ----: | ----: | ----: |
| SCSA-SCSA_PLUS    | 0.606 | 0.585 | 0.568 | 0.546 | 0.555 |
| SCSA-STATE_ART    | 0.600 | 0.605 | 0.549 | 0.510 | 0.510 |
| SCSA_PLUS_V3      | 0.605 | 0.521 | 0.507 | 0.521 | 0.512 |

Winner summary:

```text
P@1 winner: SCSA-SCSA_PLUS
P@2 winner: SCSA-STATE_ART
P@3 winner: SCSA-SCSA_PLUS
P@4 winner: SCSA-SCSA_PLUS
P@5 winner: SCSA-SCSA_PLUS
```

---

## 19. Reference Results YAML

The following YAML block shall be stored in:

```text
configs/reference_results.yaml
```

```yaml
reference_results:
  ranking_metrics:
    figure_3_b1:
      B1_SCSA_PLUS:
        MRR: 0.793
        MAP_10: 0.777
        NDCG_10: 0.788
      B1_STATE_ART:
        MRR: 0.710
        MAP_10: 0.434
        NDCG_10: 0.813
      SCSA_PLUS_V3:
        MRR: 0.665
        MAP_10: 0.454
        NDCG_10: 0.659

    figure_4_cs:
      CS_SCSA_PLUS:
        MRR: 0.641
        MAP_10: 0.669
        NDCG_10: 0.702
      CS_STATE_ART:
        MRR: 0.680
        MAP_10: 0.415
        NDCG_10: 0.817
      SCSA_PLUS_V3:
        MRR: 0.630
        MAP_10: 0.440
        NDCG_10: 0.660

    figure_5_sc:
      SC_SCSA_PLUS:
        MRR: 0.778
        MAP_10: 0.786
        NDCG_10: 0.792
      SC_STATE_ART:
        MRR: 0.739
        MAP_10: 0.508
        NDCG_10: 0.836
      SCSA_PLUS_V3:
        MRR: 0.695
        MAP_10: 0.505
        NDCG_10: 0.676

    figure_6_scsa:
      SCSA_SCSA_PLUS:
        MRR: 0.756
        MAP_10: 0.735
        NDCG_10: 0.753
      SCSA_STATE_ART:
        MRR: 0.704
        MAP_10: 0.411
        NDCG_10: 0.811
      SCSA_PLUS_V3:
        MRR: 0.665
        MAP_10: 0.427
        NDCG_10: 0.656

  precision_metrics:
    figure_7_b1:
      B1_SCSA_PLUS:
        P_1: 0.592
        P_2: 0.634
        P_3: 0.624
        P_4: 0.623
        P_5: 0.606
      B1_STATE_ART:
        P_1: 0.570
        P_2: 0.549
        P_3: 0.560
        P_4: 0.570
        P_5: 0.560
      SCSA_PLUS_V3:
        P_1: 0.577
        P_2: 0.577
        P_3: 0.558
        P_4: 0.549
        P_5: 0.557

    figure_8_cs:
      CS_SCSA_PLUS:
        P_1: 0.408
        P_2: 0.437
        P_3: 0.474
        P_4: 0.496
        P_5: 0.518
      CS_STATE_ART:
        P_1: 0.530
        P_2: 0.520
        P_3: 0.563
        P_4: 0.556
        P_5: 0.567
      SCSA_PLUS_V3:
        P_1: 0.535
        P_2: 0.535
        P_3: 0.053
        P_4: 0.528
        P_5: 0.549

    figure_9_sc:
      SC_SCSA_PLUS:
        P_1: 0.630
        P_2: 0.630
        P_3: 0.607
        P_4: 0.620
        P_5: 0.638
      SC_STATE_ART:
        P_1: 0.616
        P_2: 0.610
        P_3: 0.602
        P_4: 0.599
        P_5: 0.594
      SCSA_PLUS_V3:
        P_1: 0.616
        P_2: 0.609
        P_3: 0.611
        P_4: 0.609
        P_5: 0.605

    figure_10_scsa:
      SCSA_SCSA_PLUS:
        P_1: 0.606
        P_2: 0.585
        P_3: 0.568
        P_4: 0.546
        P_5: 0.555
      SCSA_STATE_ART:
        P_1: 0.600
        P_2: 0.605
        P_3: 0.549
        P_4: 0.510
        P_5: 0.510
      SCSA_PLUS_V3:
        P_1: 0.605
        P_2: 0.521
        P_3: 0.507
        P_4: 0.521
        P_5: 0.512

validation:
  chart_label_tolerance: 0.001
  algorithmic_reproduction_tolerance: 0.03
  relaxed_reproduction_tolerance: 0.05
  preserve_chart_anomalies: true
```

---

## 20. Main Configuration File

The following shall be stored in:

```text
configs/reproduction.yaml
```

```yaml
random_seed: 42

data:
  input_dir: data/raw
  processed_dir: data/processed
  candidate_sets_required: false

text:
  language: en
  lowercase: true
  remove_urls_from_text: true
  keep_url_count: true
  remove_mentions_from_text: true
  keep_mentions_count: true
  remove_hashtag_symbol: true
  keep_hashtag_tokens: true
  remove_punctuation: true
  remove_stopwords: true
  lemmatize: true
  disambiguation_enabled: true
  synonym_expansion_enabled: true
  min_df: 2
  max_df: 0.95
  ngram_range: [1, 2]

dimensionality_reduction:
  enabled: true
  method: truncated_svd
  n_components: 100
  random_state: 42

sentiment:
  enabled: true
  backend: vader
  normalize_to_0_1: true

recency:
  enabled: true
  lambda_decay: 0.03
  unit: days
  formula: exponential

social_capital:
  mode: paper_compatible
  normalize_components: true
  weights:
    sentiment_impact: 0.20
    engagement_score: 0.20
    content_relevance: 0.20
    network_influence: 0.20
    author_influence: 0.10
    content_virality: 0.10

recommendation:
  top_k: 10
  relevance_threshold: 4
  algorithms:
    - B1_SCSA_PLUS
    - B1_STATE_ART
    - CS_SCSA_PLUS
    - CS_STATE_ART
    - SC_SCSA_PLUS
    - SC_STATE_ART
    - SCSA_SCSA_PLUS
    - SCSA_STATE_ART
    - SCSA_PLUS_V3

evaluation:
  ranking_metrics:
    - MRR
    - MAP_10
    - NDCG_10
  precision_cutoffs: [1, 2, 3, 4, 5]

validation:
  chart_label_tolerance: 0.001
  algorithmic_reproduction_tolerance: 0.03
  relaxed_reproduction_tolerance: 0.05
  preserve_chart_anomalies: true
```

---

## 21. Reference Result Loader

```python
class ReferenceResultsRepository:
    def __init__(self, path: str):
        self.path = path
        self.data = self._load_yaml(path)

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def get_ranking_metrics(self) -> dict:
        return self.data["reference_results"]["ranking_metrics"]

    def get_precision_metrics(self) -> dict:
        return self.data["reference_results"]["precision_metrics"]

    def get_validation_config(self) -> dict:
        return self.data["validation"]
```

---

## 22. Validation Engine

## 22.1 Chart-Label Reproduction Mode

This mode validates imported or manually reconstructed result tables against the extracted figure values.

```text
Tolerance: ±0.001
```

```python
def validate_chart_label_reproduction(
    actual: float,
    expected: float,
    tolerance: float = 0.001,
) -> bool:
    return abs(actual - expected) <= tolerance
```

---

## 22.2 Algorithmic Reproduction Mode

This mode validates recomputed results from raw data.

```text
Tolerance: ±0.03
```

```python
def validate_algorithmic_reproduction(
    actual: float,
    expected: float,
    tolerance: float = 0.03,
) -> bool:
    return abs(actual - expected) <= tolerance
```

---

## 22.3 Relaxed Reproduction Mode

This mode is used when:

* Candidate sets are unavailable.
* The original sentiment backend is unknown.
* Baseline implementations are approximated.
* Preprocessing differs from the paper.

```text
Tolerance: ±0.05
```

```python
def validate_relaxed_reproduction(
    actual: float,
    expected: float,
    tolerance: float = 0.05,
) -> bool:
    return abs(actual - expected) <= tolerance
```

---

## 22.4 Anomaly Preservation Rule

The system shall preserve the following chart value:

```text
Figure 8
Algorithm: SCSA_PLUS_V3
Metric: P@3
Value: 0.053
```

This value must not be automatically corrected to `0.53` unless the user explicitly enables a data-cleaning mode.

```python
def validate_anomaly_preservation(reference_results: dict) -> bool:
    value = (
        reference_results["precision_metrics"]
        ["figure_8_cs"]
        ["SCSA_PLUS_V3"]
        ["P_3"]
    )
    return value == 0.053
```

---

## 23. Reporting Requirements

The reproduction report shall include:

1. Dataset summary.
2. Feature-engineering configuration.
3. Sentiment backend.
4. Social Capital formula.
5. Component weights.
6. Algorithm list.
7. Figures 3–6 reproduction tables.
8. Figures 7–10 reproduction tables.
9. Winner summary for every metric group.
10. Difference between reproduced and extracted values.
11. Chart-label reproduction pass/fail.
12. Algorithmic reproduction pass/fail.
13. Relaxed reproduction pass/fail, when applicable.
14. Notes about baseline assumptions.
15. Notes about candidate-set availability.
16. Notes about the Figure 8 P@3 anomaly.
17. Recommendations for future improvement.

---

## 24. Report Output Example

```text
reports/
  reproduction_report.md
  reproduction_report.html
  tables/
    figure_3_b1_ranking_metrics.csv
    figure_4_cs_ranking_metrics.csv
    figure_5_sc_ranking_metrics.csv
    figure_6_scsa_ranking_metrics.csv
    figure_7_b1_precision_metrics.csv
    figure_8_cs_precision_metrics.csv
    figure_9_sc_precision_metrics.csv
    figure_10_scsa_precision_metrics.csv
    validation_summary.csv
```

---

## 25. CLI Design

## 25.1 Preprocess

```bash
sc-recsys-v3 preprocess \
  --config configs/reproduction.yaml \
  --input data/raw \
  --output data/processed
```

---

## 25.2 Score Tweets

```bash
sc-recsys-v3 score \
  --config configs/reproduction.yaml \
  --input data/processed \
  --output data/interim/scored_tweets.parquet
```

---

## 25.3 Recommend

```bash
sc-recsys-v3 recommend \
  --config configs/reproduction.yaml \
  --user-id USER_ID \
  --top-k 10
```

---

## 25.4 Run Experiment

```bash
sc-recsys-v3 experiment \
  --config configs/reproduction.yaml \
  --reference configs/reference_results.yaml \
  --output reports/reproduction
```

---

## 25.5 Validate Results

```bash
sc-recsys-v3 validate \
  --actual reports/reproduction/metrics.csv \
  --reference configs/reference_results.yaml \
  --mode chart_label
```

Supported validation modes:

```text
chart_label
algorithmic
relaxed
```

---

## 25.6 Run Ablation

```bash
sc-recsys-v3 ablation \
  --config configs/ablation.yaml \
  --output reports/ablation
```

---

## 26. Ablation Studies

The implementation shall support the following ablation experiments:

```text
SCSA_PLUS_V3 without sentiment impact
SCSA_PLUS_V3 without engagement score
SCSA_PLUS_V3 without content relevance
SCSA_PLUS_V3 without network influence
SCSA_PLUS_V3 without author influence
SCSA_PLUS_V3 without content virality
SCSA_PLUS_V3 without recency
SCSA_PLUS_V3 without diversity
SCSA_PLUS_V3 without context
SCSA_PLUS_V3 with normalized features
SCSA_PLUS_V3 with raw features
SCSA_PLUS_V3 with TF-IDF context
SCSA_PLUS_V3 with embedding context
SCSA_PLUS_V3 with lexical diversity
SCSA_PLUS_V3 with semantic diversity
```

---

## 27. Testing Strategy

## 27.1 Unit Tests

Required unit tests:

1. Text cleaning removes URLs, mentions, and punctuation.
2. Tokenization returns expected tokens.
3. Sentiment score is normalized to 0–1.
4. Engagement score is calculated correctly.
5. Network influence counts mentions and URLs.
6. Content virality sums retweets and quotes.
7. Social Capital weights sum to 1.0.
8. Social Capital score increases when a positive component increases.
9. Recommender returns items in descending score order.
10. MRR matches known toy data.
11. MAP@10 matches known toy data.
12. NDCG@10 matches known toy data.
13. Precision@K matches known toy data.

---

## 27.2 Reference Result Tests

### Test 1 — Ranking Metric Group Count

```python
def test_ranking_metric_group_count(reference_repo):
    ranking = reference_repo.get_ranking_metrics()
    assert len(ranking) == 4
```

---

### Test 2 — Precision Metric Group Count

```python
def test_precision_metric_group_count(reference_repo):
    precision = reference_repo.get_precision_metrics()
    assert len(precision) == 4
```

---

### Test 3 — Figure 3 Values

```python
def test_figure_3_b1_scsa_plus_values(reference_repo):
    fig3 = reference_repo.get_ranking_metrics()["figure_3_b1"]
    assert fig3["B1_SCSA_PLUS"]["MRR"] == 0.793
    assert fig3["B1_SCSA_PLUS"]["MAP_10"] == 0.777
    assert fig3["B1_SCSA_PLUS"]["NDCG_10"] == 0.788
```

---

### Test 4 — Figure 8 Anomaly Preservation

```python
def test_figure_8_anomaly_is_preserved(reference_repo):
    fig8 = reference_repo.get_precision_metrics()["figure_8_cs"]
    assert fig8["SCSA_PLUS_V3"]["P_3"] == 0.053
```

---

### Test 5 — Chart-Label Validation

```python
def test_chart_label_validation():
    assert validate_chart_label_reproduction(
        actual=0.793,
        expected=0.793,
        tolerance=0.001,
    )
```

---

## 27.3 Integration Tests

Create a synthetic dataset with:

```text
5 users
30 tweets
20 mentions
10 URLs
10 hashtags
10 ratings
```

Expected behavior:

1. Tweets with high engagement should rank higher in engagement-heavy mode.
2. Tweets with high content relevance should rank higher in content-heavy mode.
3. Tweets with high network influence should rank higher in network-heavy mode.
4. SCSA_PLUS_V3 should produce a complete top-k recommendation list.
5. Evaluation should produce MRR, MAP@10, NDCG@10, and Precision@1–5.
6. Validation should compare results against reference tables.

---

## 28. Known Ambiguities and Assumptions

The implementation must explicitly document the following:

1. The exact original baseline implementations may not be fully available.
2. The exact original candidate sets may not be fully available.
3. The exact sentiment backend and configuration may differ from the paper.
4. The exact topic-keyword sets for context scoring may require reconstruction.
5. SCSA_PLUS_V3 underperforms the previous SCSA_PLUS variants in several reported charts.
6. STATE_ART often performs better on NDCG@10.
7. SCSA_PLUS often performs better on MRR and MAP@10.
8. The Figure 8 P@3 value for SCSA_PLUS_V3 appears anomalous.
9. Chart-label reproduction and algorithmic reproduction are different validation goals.
10. Reproducing exact figures may require importing previous ranking outputs rather than recomputing them from scratch.

---

## 29. Future Improvements

Future versions should investigate:

1. Online evaluation.
2. Real-time ranking.
3. Click-through-rate optimization.
4. Dwell-time optimization.
5. Community-aware recommendation.
6. Knowledge-enhanced user profiles.
7. Graph neural network baselines.
8. Transformer-based content relevance.
9. Semantic diversity.
10. Bot and misinformation filtering.
11. Multi-platform generalization.
12. Learning Social Capital weights automatically.
13. Re-ranking using diversity and novelty constraints.
14. Hybridizing SCSA_PLUS_V3 with CS and STATE_ART strengths.

---

## 30. Definition of Done

The implementation shall be considered complete when:

1. Dataset loading works.
2. Schema validation passes.
3. Feature engineering runs successfully.
4. Social Capital components are computed.
5. SCSA_PLUS_V3 recommendations are generated.
6. Baseline algorithms are executable or importable.
7. MRR, MAP@10, NDCG@10, and Precision@1–5 are computed.
8. Figures 3–6 are reproduced as structured result tables.
9. Figures 7–10 are reproduced as structured precision tables.
10. Extracted chart values are stored in YAML.
11. Chart-label validation passes.
12. Algorithmic reproduction validation is available.
13. The Figure 8 P@3 anomaly is preserved and documented.
14. A full reproduction report is generated.
15. All assumptions and deviations from the paper are documented.
16. Unit and integration tests pass.
