# Copyright (C) 2024 Bartosz Paradowski, Jakub Więckowski

import numpy as np
from ..validator import Validator
from ..utils import memory_guard

def __welsch_minimizer__(s: float, sigma: float) -> float:
    return np.exp(-(s**2)/(2*sigma**2))

def __euc_dist__(R_1: np.ndarray, R_2: np.ndarray) -> float:
    return np.sqrt(np.sum((R_1 - R_2)**2))

def __pdf__(mu: float, sigma: float, x: float) -> float:
    return (1/(np.sqrt(2*np.pi*(sigma**2)))) * (np.e**(-((x-mu)**2)/(2*(sigma**2))))

def __indicators__(rankings: np.ndarray, R_avg: np.ndarray, sigma: float, w: np.ndarray) -> tuple[float, float]:
    consensus = 0
    trust = 0
    K = rankings.shape[0]
    M = rankings.shape[1]
    for m in range(M):
        R = rankings[:, m]
        s = 0
        for k in range(K):
            s += __pdf__(0, sigma, R[k]-R_avg[k])/__pdf__(0, sigma, 0)
        consensus += s
        trust += w[m]*s
    return consensus/(K * M), trust/K

@memory_guard
def HQ_compromise(rankings: np.ndarray, max_iters:int = 1000, tol:float = 10e-10) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
    """
    Perform HQ (Half-Quadratic) compromise on a set of rankings.

    Parameters
    ----------
    rankings : np.ndarray
        2D array representing the rankings. Each row corresponds to an alternative,
        and each column corresponds to a criterion.

    max_iters : int, optional, default=1000
        Maximum number of iterations for the compromise algorithm.

    tol : float, optional, default=10e-10
        Tolerance for convergence.

    Returns
    -------
        tuple
            Tuple that contains (consensus_index, trust_index), weights of rankings, compromised ranking
    
    Examples
    --------
    ### Example 1
        >>> rankings = np.array([[2, 2, 2],
        >>>                     [3, 4, 5],
        >>>                     [1, 1, 4],
        >>>                     [4, 3, 1],
        >>>                     [7, 5, 7],
        >>>                     [8, 8, 8],
        >>>                     [5, 6, 3],
        >>>                     [6, 7, 6]])
        >>> HQ_compromise(rankings)
    
    ### Example 2 - set tolerance on convergance and change maximum number of iterations
        >>> rankings = np.array([[2, 2, 2],
        >>>                     [3, 4, 5],
        >>>                     [1, 1, 4],
        >>>                     [4, 3, 1],
        >>>                     [7, 5, 7],
        >>>                     [8, 8, 8],
        >>>                     [5, 6, 3],
        >>>                     [6, 7, 6]])
        >>> HQ_compromise(rankings, max_iters=100, tol=10e-6)

    """

    Validator.is_type_valid(rankings, np.ndarray)
    Validator.is_type_valid(max_iters, int)
    Validator.is_positive_value(max_iters)
    Validator.is_type_valid(tol, float)
    Validator.is_positive_value(tol)

    R = rankings
    M = R.shape[1]

    R_star = 1/M * np.sum(R, axis=1)

    alpha = np.zeros(R.shape[1])
    iter = 0

    while iter < max_iters:
        sigma = 0
        for i in range(M):
            sigma += __euc_dist__(R[:, i], R_star)**2
        sigma = (sigma) / (2 * (M**2))

        old_alpha = alpha.copy()
        for i in range(M):
            alpha[i] = __welsch_minimizer__(__euc_dist__(R[:, i], R_star), sigma)

        w = alpha/np.sum(alpha)

        R_star_old = R_star.copy()
        R_star = np.sum(R*w, axis=1)
        
        iter += 1
        if __euc_dist__(R_star_old, R_star) <= tol and __euc_dist__(old_alpha, alpha) <= tol:
            break
        
    if iter == max_iters:
        print(f'Compromise not obtained in {max_iters} iterations.')

    consensus, trust = __indicators__(R, R_star, sigma, w)
    return ((consensus, trust), w, R_star)