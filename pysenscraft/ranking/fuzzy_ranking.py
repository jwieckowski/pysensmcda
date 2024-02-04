# Copyright (C) 2024 Jakub Więckowski

import numpy as np
import pandas as pd

def fuzzy_ranking(ranks, variant:str = None):

    ALT = len(ranks[0])

    columns_labels = [f'A{i+1}' for i in range(ALT)]

    pd_rank = pd.DataFrame(ranks, columns=columns_labels)
    rank_prob = np.zeros((ALT, ALT))  

    for row, col in enumerate(pd_rank.columns):
        for pos in range(ALT):
            rank_prob[pos, row] = len(pd_rank[pd_rank[col] == pos+1])

    rank_prob = np.round(rank_prob / len(pd_rank), 4)
    rank_prob = pd.DataFrame(rank_prob, columns=columns_labels)

    if variant == 'rank':
        M = rank_prob.to_numpy()
        return np.round(M / np.max(M, axis=1), 4)
    elif variant == 'alt':
        M = rank_prob.to_numpy()
        return np.round(M / np.max(M, axis=0), 4)
    else:
        return rank_prob.to_numpy()


ranks = np.array([
    [1, 2, 3, 4, 5],
    [2, 1, 5, 3, 4],
    [4, 3, 2, 5, 1],
    [3, 2, 1, 4, 5],
])

fuzzy_rank = fuzzy_ranking(ranks, variant='alt')
print(fuzzy_rank)


