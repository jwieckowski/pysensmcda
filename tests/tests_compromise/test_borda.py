# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.compromise import borda

def test_borda():
    rankings = np.array([[3, 2, 3],
                        [4, 4, 4],
                        [2, 3, 2],
                        [1, 1, 1]])
    compromised_ranking = borda(rankings)
    assert np.array_equal(compromised_ranking, np.array([3, 4, 2, 1]))


def test_borda_invalid_rankings():
    # This will raise a TypeError in the function
    rankings = "invalid_ranking"  

    with raises(TypeError):
        borda(rankings)
