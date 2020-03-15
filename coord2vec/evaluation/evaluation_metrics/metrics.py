import logging
from typing import Tuple, List

import sklearn
from sklearn.metrics import precision_recall_fscore_support, make_scorer
import numpy as np


def soft_precision_recall_fscore(y_true, y_proba, threshold: float = 0.5) -> Tuple[float, float, float, float]:
    """
    calculate evaluation metrics with respect to the true geometries
    Args:
        y_true: possibly soft labels to be used for this function
        y_proba: the probability assigned to the geometries by the model
        threshold: threshold on the probabilities

    Returns:
        soft-labeled precision, recall and f_score
    """
    y_pred = y_proba >= threshold

    # calculate precision and recall
    if (y_true * y_pred).sum() == 0.:
        logging.warning("(y_true * y_pred).sum() == 0. All probs are 0.")
    precision = (y_true * y_pred).sum() / y_pred.sum()
    recall = (y_true * y_pred).sum() / y_true.sum()
    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score, len(y_true)


def soft_precision_recall_curve(y_true, y_proba) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    like sklearn's precision recall curve, only using the soft labels
    Args:
        y_true: true soft labels [0,1]
        y_proba: label probability [0,1]

    Returns:
        precision array, recall array, threshold array
    """
    precisions, recalls, thresholds = [], [], []
    for threshold in np.linspace(0, 1, 201, endpoint=True):
        precision, recall, _, _ = soft_precision_recall_fscore(y_true, y_proba, threshold=threshold)
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(threshold)
    return np.nan_to_num(np.array(precisions)), np.array(recalls), np.array(thresholds)

def soft_auc(y_true, y_proba) -> float:
    """
    #TODO fill this
    Args:
        y_true:
        y_proba:

    Returns:

    """
    precision, recall, _ = soft_precision_recall_curve(y_true, y_proba)
    return sklearn.metrics.auc(recall, precision)

def get_soft_auc_scorer():
    return make_scorer(soft_auc, needs_proba=True)
