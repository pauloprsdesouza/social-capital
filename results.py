import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('updated_recommendations.csv') 

# Display the first few rows of the dataset to understand its structure
data.head()

def compute_mrr(ranked_list):
    """
    Compute the Mean Reciprocal Rank (MRR) given a ranked list of ratings.
    """
    for idx, rating in enumerate(ranked_list):
        if rating > 3:  # Assuming a rating greater than 3 is considered as relevant
            return 1.0 / (idx + 1)
    return 0

# Group by user and algorithm, then apply the MRR function to the ratings
mrr_values = data.groupby(['user_id', 'algorithm'])['rating'].apply(list).reset_index()
mrr_values['mrr'] = mrr_values['rating'].apply(compute_mrr)

# Compute the mean MRR for each algorithm
mrr_per_algorithm = mrr_values.groupby('algorithm')['mrr'].mean()
print(mrr_per_algorithm)

def compute_average_precision(ranked_list):
    """
    Compute the Average Precision given a ranked list of ratings.
    """
    relevant_items = 0
    cum_precision = 0
    for idx, rating in enumerate(ranked_list[:10]):
        if rating > 3:  # Assuming a rating greater than 3 is considered as relevant
            relevant_items += 1
            cum_precision += relevant_items / (idx + 1)
    return cum_precision / min(len(ranked_list), 10)

# Compute the Average Precision for each user and algorithm
mrr_values['avg_precision'] = mrr_values['rating'].apply(compute_average_precision)

# Compute the mean Average Precision for each algorithm (MAP@10)
map_per_algorithm = mrr_values.groupby('algorithm')['avg_precision'].mean()

print(map_per_algorithm)


def compute_average_precision(dataframe, k=10, relevant_threshold=4):
    dataframe = dataframe.sort_values(by=["user_id", "algorithm", "ranking"])
    
    dataframe["is_relevant"] = dataframe["rating"] >= relevant_threshold

    dataframe["precision_at_k"] = dataframe.groupby(["user_id", "algorithm"])["is_relevant"].cumsum() / (dataframe["ranking"].astype(int))

    dataframe = dataframe[dataframe["ranking"] <= k]

    avg_precision = dataframe[dataframe["is_relevant"] == True].groupby(["user_id", "algorithm"])["precision_at_k"].mean()

    mrr_values['avg_precision'] = avg_precision.groupby(level="algorithm").mean()

    return avg_precision.groupby(level="algorithm").mean()

map_per_algorithm = compute_average_precision(data)
print(map_per_algorithm)

def compute_ndcg(ranked_list):
    """
    Compute the NDCG@10 given a ranked list of ratings.
    """
    dcg = sum([(2**rating - 1) / np.log2(idx + 2) for idx, rating in enumerate(ranked_list[:10])])
    # Compute ideal DCG (i.e., perfect ranking)
    sorted_list = sorted(ranked_list, reverse=True)
    idcg = sum([(2**rating - 1) / np.log2(idx + 2) for idx, rating in enumerate(sorted_list[:10])])
    return dcg / idcg if idcg > 0 else 0

# Compute NDCG@10 for each user and algorithm
mrr_values['ndcg'] = mrr_values['rating'].apply(compute_ndcg)

# Compute the mean NDCG@10 for each algorithm
ndcg_per_algorithm = mrr_values.groupby('algorithm')['ndcg'].mean()

print(ndcg_per_algorithm)


def compute_precision_at_k(ranked_list, k=5):
    """
    Compute the precision at k given a ranked list of ratings.
    """
    return sum([1 for rating in ranked_list[:k] if rating > 3]) / k

# Compute Precision@10 for each user and algorithm
mrr_values['precision_at_5'] = mrr_values['rating'].apply(compute_precision_at_k)

# Compute the mean Precision@10 for each algorithm
precision_at_5_per_algorithm = mrr_values.groupby('algorithm')['precision_at_5'].mean()

print(precision_at_5_per_algorithm)
print()

import scipy.stats as stats

# ANOVA test results
anova_results = {}

metrics = ['mrr', 'avg_precision', 'ndcg', 'precision_at_5']

for metric in metrics:
    f_val, p_val = stats.f_oneway(
        mrr_values[mrr_values['algorithm'] == 'B1'][metric],
        mrr_values[mrr_values['algorithm'] == 'B1-SCSA_PLUS'][metric],
        mrr_values[mrr_values['algorithm'] == 'CS'][metric],
        mrr_values[mrr_values['algorithm'] == 'CS-SCSA_PLUS'][metric],
        mrr_values[mrr_values['algorithm'] == 'SC'][metric],
        mrr_values[mrr_values['algorithm'] == 'SC-SCSA_PLUS'][metric],
        mrr_values[mrr_values['algorithm'] == 'SCSA'][metric],
        mrr_values[mrr_values['algorithm'] == 'SCSA-SCSA_PLUS'][metric],
        mrr_values[mrr_values['algorithm'] == 'B1-SCSA_PLUS'][metric],
        mrr_values[mrr_values['algorithm'] == 'B1-STATE_ART'][metric],
        mrr_values[mrr_values['algorithm'] == 'CS-SCSA_PLUS'][metric],
        mrr_values[mrr_values['algorithm'] == 'CS-STATE_ART'][metric],
        mrr_values[mrr_values['algorithm'] == 'SC-SCSA_PLUS'][metric],
        mrr_values[mrr_values['algorithm'] == 'SC-STATE_ART'][metric],
        mrr_values[mrr_values['algorithm'] == 'SCSA-SCSA_PLUS'][metric],
        mrr_values[mrr_values['algorithm'] == 'SCSA-STATE_ART'][metric]
    )
    anova_results[metric] = {'F-value': f_val, 'p-value': p_val}

print(anova_results)
