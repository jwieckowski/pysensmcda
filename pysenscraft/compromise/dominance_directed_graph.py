import numpy as np
from numpy.linalg import matrix_power
from scipy.stats import rankdata

def dominance_directed_graph(rankings: np.ndarray):
    """
    Calculates compromised ranking using dominance directed graph.

    Parameters
    -----------
        rankings : ndarray
            Two-dimensional matrix containing different rankings in columns.

    Example
    ----------
        >>> rankings = np.array([[3, 2, 3],
        >>>                     [4, 4, 4],
        >>>                     [2, 3, 2],
        >>>                     [1, 1, 1]])
        >>> compromised_ranking = dominance_directed_graph(rankings)

    Returns
    --------
        ndarray
            Numpy array containing compromised ranking.
    """

    alt_num, method_num = rankings.shape

    final_points = np.zeros((alt_num, alt_num))
    for method_idx in range(method_num):
        ranking = rankings[:, method_idx]
        A = np.zeros((alt_num, alt_num))
        for idx in range(alt_num):
            ind_better = np.where(ranking > ranking[idx])[0]
            A[idx, ind_better] += 1
        enhanced_dominant_matrix = A + matrix_power(A, 2)
        final_points += enhanced_dominant_matrix
    
    return rankdata(-np.sum(final_points, axis=1))