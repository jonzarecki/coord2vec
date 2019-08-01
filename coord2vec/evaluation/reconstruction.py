import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr

def reconstruction_correlation(embedding: np.ndarray,
                               features: np.ndarray) -> float:
    """
    computer the self-distance matrix correlation between two datasets
    Args:
        embedding: a 2-d array
        features: a 2-d array, with same shape as the embedding

    Returns:
        pearson correlation
    """
    features_distance_matrix = pairwise_distances(features)
    embedding_distance_matrix = pairwise_distances(embedding)
    corr_coefficient, p_value = pearsonr(features_distance_matrix.flatten(), embedding_distance_matrix.flatten())
    return corr_coefficient