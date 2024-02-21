import numpy as np

def welsch_minimizer(s, sigma):
    # minimizer Table 1
    return np.exp(-(s**2)/(sigma**2))

def euc_dist(R_1, R_2):
    return np.sqrt(np.sum((R_1 - R_2)**2))

def pdf(mu, sigma, x):
    return (1/(np.sqrt(2*np.pi*(sigma**2)))) * (np.e**(-((x-mu)**2)/(2*(sigma**2))))

def indicators(rankings, R_avg, sigma, w):
    # consensus Eq (14)
    # trust Eq (15)
    consensus = 0
    trust = 0
    K = rankings.shape[0]
    M = rankings.shape[1]
    for m in range(M):
        R = rankings[:, m]
        s = 0
        for k in range(K):
            s += pdf(0, sigma, R[k]-R_avg[k])/pdf(0, sigma, 0)
        consensus += s
        trust += w[m]*s
    return consensus/(K * M), trust/K

def HQ_compromise(R):
    """
    Test case 1:
    >>> rankings = np.array([[2, 2, 2],
    >>>                     [3, 4, 5],
    >>>                     [1, 1, 4],
    >>>                     [4, 3, 1],
    >>>                     [7, 5, 7],
    >>>                     [8, 8, 8],
    >>>                     [5, 6, 3],
    >>>                     [6, 7, 6]])
    Should return: ((0.85, 0.95), array([0.4997, 0.4946, 0.0057]), array([2, 4, 1, 3, 6, 8, 5, 7]))

    Test case 2:
    >>> rankings = np.array([[1, 1, 1],
    >>>                     [2, 2, 2],
    >>>                     [7, 6, 6],
    >>>                     [3, 3, 4],
    >>>                     [6, 7, 7],
    >>>                     [4, 4, 3],
    >>>                     [5, 5, 5]])
    Should return: ((0.8, 1.0), array([0, 1, 0]), array([1, 2, 6, 3, 7, 4, 5]))
    
    Test case 3:
    >>> rankings = np.array([[6, 5, 5],
    >>>                     [1, 1, 1],
    >>>                     [7, 7, 6],
    >>>                     [2, 2, 4],
    >>>                     [3, 6, 7],
    >>>                     [5, 4, 3],
    >>>                     [4, 3, 2]])
    Should return: ((0.8, 0.98), array([0.0056, 0.9502, 0.0442]), array([5, 1, 7, 2, 6, 4, 3]))
    """
    K = R.shape[0]
    M = R.shape[1]

    # R_avg = R*; Eq (6)
    R_avg = 1/M * np.sum(R, axis=1)

    alpha = np.zeros(R.shape[1])
    change = 1
    while change != 0:
        # Eq (13)
        sigma = 0
        for i in range(M):
            sigma += euc_dist(R[:, i], R_avg)**2
        sigma = (sigma) / (2 * (K**2))

        # minimizer Table 1
        # alpha Eq (9)
        for i in range(M):
            alpha[i] = welsch_minimizer(euc_dist(R[:, i], R_avg), sigma)
        # w Eq (10)
        w = alpha/np.sum(alpha)
        change = np.sum(R_avg - np.sum(w*R, axis=1))
        # R_avg Eq (10)
        R_avg = np.sum(w*R, axis=1)

    consensus, trust = indicators(R, R_avg, sigma, w)
    return ((consensus, trust), w, R_avg)

rankings = np.array([[6, 5, 5],
                    [1, 1, 1],
                    [7, 7, 6],
                    [2, 2, 4],
                    [3, 6, 7],
                    [5, 4, 3],
                    [4, 3, 2]])


HQ_compromise(rankings)