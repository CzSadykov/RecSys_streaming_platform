import numpy as np


def ndcg_at_k(y_pred, y_true, k: int = 10):
    """Calculate NDCG at k

    Parameters
    ----------
    y_pred : array-like
        Predicted values
    y_true : array-like
        True values
    k : int, optional
        Number of items to consider, by default 10

    Returns
    -------
    ndcg : float
        Normalized Discounted Cumulative Gain at k
    """
    if not y_pred or not y_true:
        return 0.0

    ideal_relevance = np.sort(y_pred)[::-1][:k]
    relevance = np.array(y_true[:k])
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))
    return dcg / (idcg + 1e-42)


def map_at_k(y_pred, y_true, k: int = 10):
    """Calculate MAP at k

    Parameters
    ----------
    y_pred : array-like
        Predicted values
    y_true : array-like
        True values
    k : int, optional
        Number of items to consider, by default 10

    Returns
    -------
    map : float
        Mean Average Precision at k
    """
    if not y_pred or not y_true:
        return 0.0

    if len(y_pred) > k:
        y_pred = y_pred[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_pred):
        if p in y_true and p not in y_pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(y_true), k)
