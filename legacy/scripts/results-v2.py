import pandas as pd
import numpy as np
import scipy.stats as stats
import itertools


# Load the dataset
data = pd.read_csv('./output.csv')

# Functions to compute various metrics
def compute_mrr(ranked_list):
    for idx, rating in enumerate(ranked_list):
        if rating > 3:
            return 1.0 / (idx + 1)
    return 0

def compute_average_precision(ranked_list, k=10):
    relevant_items = 0
    cum_precision = 0
    for idx, rating in enumerate(ranked_list[:k]):
        if rating > 3:
            relevant_items += 1
            cum_precision += relevant_items / (idx + 1)
    return cum_precision / min(len(ranked_list), k)

def compute_ndcg(ranked_list, k=10):
    """
    Compute the NDCG@10 given a ranked list of ratings.
    """
    ranked_list = [float(r) for r in ranked_list]  # Ensure all ratings are float
    ranked_list_sorted = sorted(ranked_list, reverse=True)[:k]
    dcg = sum([(2**ranked_list[idx] - 1) / np.log2(idx + 2) for idx in range(k)])
    idcg = sum([(2**rating - 1) / np.log2(idx + 2) for idx, rating in enumerate(ranked_list_sorted)])
    return dcg / idcg if idcg > 0 else 0

def compute_precision_at_k(ranked_list, k=5):
    return sum([1 for rating in ranked_list[:k] if rating > 3]) / k

# Group data and apply metrics computation
mrr_values = data.groupby(['user_id', 'algorithm'])['rating'].apply(list).reset_index()
mrr_values['mrr'] = mrr_values['rating'].apply(compute_mrr)
mrr_values['avg_precision'] = mrr_values['rating'].apply(compute_average_precision)
mrr_values['ndcg'] = mrr_values['rating'].apply(compute_ndcg)
mrr_values['precision_at_5'] = mrr_values['rating'].apply(lambda x: compute_precision_at_k(x, 5))

# ANOVA test results
anova_results = {}

metrics = ['mrr', 'avg_precision', 'ndcg', 'precision_at_5']
algorithms = data['algorithm'].unique()

alpha = 0.05


for metric in metrics:
    anova_results[metric] = {}
    print(f"\nANOVA Results for {metric}:")
    for algo_pair in itertools.combinations(algorithms, 2):
        # Prepare data for the two algorithms
        data1 = mrr_values[(mrr_values['algorithm'] == algo_pair[0])][metric].dropna()
        data2 = mrr_values[(mrr_values['algorithm'] == algo_pair[1])][metric].dropna()
        
        # Perform ANOVA
        f_val, p_val = stats.f_oneway(data1, data2)
        
        if p_val < alpha:
        # Store results
            anova_results[metric][f"{algo_pair[0]} vs {algo_pair[1]}"] = {'F-value': f_val, 'p-value': p_val}
            
            # Print results
            print(f"{algo_pair[0]} vs {algo_pair[1]}: F-value = {f_val:.4f}, p-value = {p_val:.4g} \n")
