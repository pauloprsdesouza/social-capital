# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
data = pd.read_csv('output_recommendations.csv')

# Preprocessing
# Define a simple text cleaner
import re
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # Remove special characters and numbers
    text = text.lower().strip()  # Lowercase and strip
    return text

# Apply cleaning function
data['clean_text'] = data['text'].apply(clean_text)

# Splitting data into training and testing
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Select additional features
additional_features = train_data[['like_count', 'retweet_count', 'reply_count', 'quote_count']]

# Scale the features
scaler = StandardScaler()
X_additional_train = scaler.fit_transform(additional_features)
X_additional_test = scaler.transform(test_data[['like_count', 'retweet_count', 'reply_count', 'quote_count']])


# Feature Engineering
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_data['clean_text']).toarray()
X_test = vectorizer.transform(test_data['clean_text']).toarray()

# Convert rating to a float type if it's not already
train_data['rating'] = train_data['rating'].astype(float)
test_data['rating'] = test_data['rating'].astype(float)

# Neural Network Architecture
tweet_input = Input(shape=(1000,), dtype='float32')
additional_input = Input(shape=(X_additional_train.shape[1],), dtype='float32', name='additional_input')  # Additional features

combined = Concatenate()([tweet_input, additional_input])

# Dense layers
dense_1 = Dense(128, activation='relu')(combined)
dense_2 = Dense(64, activation='relu')(dense_1)
predictions = Dense(1, activation='linear')(dense_2)

# Create and compile the model
model = Model(inputs=[tweet_input, additional_input], outputs=predictions)
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit([X_train, X_additional_train], train_data['rating'], epochs=5, batch_size=32, validation_split=0.1)

# Making predictions
predictions = model.predict([X_test, X_additional_test]).flatten()

# Reshaping the test ratings for evaluation
test_ratings = test_data['rating'].values

# Calculate Precision@K
def precision_at_k(true_labels, predicted_scores, k=10):
    correct_predictions = 0
    for true_item, scores in zip(true_labels, predicted_scores):
        top_k_indices = np.argsort(-scores)[:k]  # Indices of top k scores
        if true_item in top_k_indices:  # Check if true item is in top k predictions
            correct_predictions += 1
    return correct_predictions / len(true_labels)

# Calculate Mean Reciprocal Rank
def mean_reciprocal_rank(true_labels, predicted_scores):
    mrr = 0
    for true_item, scores in zip(true_labels, predicted_scores):
        sorted_indices = np.argsort(-scores)  # Indices of scores sorted in descending order
        for rank, index in enumerate(sorted_indices, start=1):
            if true_item == index:  # Check if true item is at this rank
                mrr += 1 / rank
                break
    return mrr / len(true_labels)

def average_precision_at_k(true_item, predicted_scores, k=10):
    # Ensure predicted_scores is a 1D array of scores
    predicted_scores = np.array(predicted_scores).flatten()
    
    # Get the indices of the scores, sorted by descending order
    sorted_indices = np.argsort(-predicted_scores)
    
    # Initialize variables to calculate average precision
    relevant = 0
    average_precision = 0
    
    # Loop through the sorted list up to k elements
    for i in range(min(k, len(predicted_scores))):
        if sorted_indices[i] == true_item:  # Check if the sorted item is the true item
            relevant += 1
            average_precision += relevant / (i + 1)
            break  # Stop if we found the true item
    
    return average_precision / min(k, relevant) if relevant > 0 else 0

def map_at_k(true_labels, predicted_scores, k=10):
    # Ensure the inputs are arrays
    true_labels = np.array(true_labels)
    predicted_scores = np.array(predicted_scores)

    # Check if predicted_scores is a 2D array
    if predicted_scores.ndim == 1:
        predicted_scores = predicted_scores.reshape(-1, 1)
    
    # Calculate MAP@K for each set of true labels and predicted scores
    all_aps = [average_precision_at_k(true_labels[i], predicted_scores[i], k) 
               for i in range(len(true_labels))]
    
    # Return the mean of all average precisions
    return np.mean(all_aps)

# Assuming 'item_id' represents the item and each user has only one relevant item
true_labels = test_data['item_id'].values  # Array of relevant item indices

predicted_scores = predictions  # Array of predicted scores
print(predictions)

# Calculate metrics
print(f"MAP@10: {map_at_k(true_labels, predicted_scores, k=10)}")
print(f"MRR: {mean_reciprocal_rank(true_labels, predicted_scores)}")
print(f"Precision@5: {precision_at_k(true_labels, predicted_scores, k=5)}")
print(f"NDCG@10: {ndcg_score(np.asarray([test_ratings]), np.asarray([predictions]), k=10)}")

# Save the model for future use
#model.save('/path/to/save/your/model')