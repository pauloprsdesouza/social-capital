import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from sklearn.metrics import label_ranking_average_precision_score, ndcg_score, precision_score


# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
data = pd.read_csv('output_recommendations.csv')

# Preprocessing
def clean_text(text):
    return re.sub(r'http\S+', '', text).lower().strip()

# Apply cleaning function
data['clean_text'] = data['text'].apply(clean_text)

# Creating a composite key of 'user_id' and 'algorithm'
data['user_algorithm'] = data['user_id'].astype(str) + "_" + data['algorithm']

# Splitting data into training and testing
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
additional_features = ['like_count', 'retweet_count', 'reply_count', 'quote_count']
X_additional_train = scaler.fit_transform(train_data[additional_features])
X_additional_test = scaler.transform(test_data[additional_features])

# Feature Engineering with TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_data['clean_text']).toarray()
X_test = vectorizer.transform(test_data['clean_text']).toarray()

# Neural Network Architecture
tweet_input = Input(shape=(1000,))
additional_input = Input(shape=(X_additional_train.shape[1],), name='additional_input')
combined = Concatenate()([tweet_input, additional_input])
dense_1 = Dense(128, activation='relu')(combined)
dense_2 = Dense(64, activation='relu')(dense_1)
predictions = Dense(1, activation='linear')(dense_2)

# Compile the model
model = Model(inputs=[tweet_input, additional_input], outputs=predictions)
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit([X_train, X_additional_train], train_data['rating'], epochs=5, batch_size=32, validation_split=0.1)

# Making predictions
predicted_scores = np.round(model.predict([X_test, X_additional_test]).flatten())

def mean_reciprocal_rank(relevance_scores):
    """Calculate the mean reciprocal rank."""
    ranks = np.argwhere(np.asarray(relevance_scores) == 1)[:, 0] + 1  # Find ranks of relevant items
    return np.mean(1 / ranks) if len(ranks) > 0 else 0

def calculate_metrics(grouped_data, threshold=4):
    # Sort true rankings and predicted ratings for each group
    grouped_data['sorted_predicted_ratings'] = grouped_data['predicted_ratings'].apply(lambda x: sorted(x, reverse=True))
    grouped_data['sorted_binary_relevance'] = grouped_data.apply(
        lambda row: [row['binary_relevance'][i] for i in np.argsort(row['true_rankings'])], axis=1
    )

    # Calculate the metrics for each group
    metrics_results = grouped_data.apply(
        lambda row: pd.Series({
            'MAP@10': label_ranking_average_precision_score([row['sorted_binary_relevance'][:10]], [row['sorted_predicted_ratings'][:10]]),
            'NDCG@10': ndcg_score([row['sorted_binary_relevance'][:10]], [row['sorted_predicted_ratings'][:10]]),
            'Precision@5': precision_score(row['sorted_binary_relevance'][:5], [1 if x >= threshold else 0 for x in row['sorted_predicted_ratings'][:5]], zero_division=0),
            'MRR': mean_reciprocal_rank(row['sorted_binary_relevance'])
        }), axis=1
    )

    return metrics_results.mean()

scsa_plus_data = data[data['algorithm'].str.contains('SCSA_PLUS')]

# Preparing the data for evaluation
grouped_data = scsa_plus_data.groupby(['user_id', 'algorithm']).apply(
    lambda x: pd.Series({
        'predicted_ratings': list(x['rating']),
        'true_rankings': list(x['ranking']),
        'binary_relevance': [1 if rating >= 4 else 0 for rating in x['rating']]  # Assuming a rating of 4 or higher is considered relevant
    })
).reset_index()

# Calculate the metrics
average_metrics = calculate_metrics(grouped_data)

# Print the average metrics
print("Average Metrics for SCSA_PLUS Algorithm:")
print(average_metrics)