import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr


def reconstruction_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    computer the self-distance matrix correlation between two datasets
    Args:
        x: an array of shape (n_samples, x_dim)
        y: an array of shape (n_samples, y_dim)

    Returns:
        pearson correlation
    """
    x_distance_matrix = pairwise_distances(x)
    y_distance_matrix = pairwise_distances(y)
    corr_coefficient, p_value = pearsonr(x_distance_matrix.flatten(), y_distance_matrix.flatten())
    return corr_coefficient
