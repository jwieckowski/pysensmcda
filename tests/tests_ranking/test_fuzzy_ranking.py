# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.ranking import fuzzy_ranking

def test_fuzzy_ranking():
    rankings = np.array([
        [1, 2, 3, 4, 5],
        [2, 1, 5, 3, 4],
        [4, 3, 2, 5, 1],
        [3, 2, 1, 4, 5],
    ])
    fuzzy_rank = fuzzy_ranking(rankings, normalization_axis=0)
    assert fuzzy_rank.shape == (5, 5)
    
def test_fuzzy_ranking_error():
    rankings = 1
    with raises(TypeError):
        fuzzy_ranking(rankings, normalization_axis=0)

def test_fuzzy_ranking_normalization_axis_error():
    rankings = 1
    with raises(TypeError):
        fuzzy_ranking(rankings, normalization_axis=2)
    
