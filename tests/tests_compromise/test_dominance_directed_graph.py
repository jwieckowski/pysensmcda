# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysensmcdacompromise import dominance_directed_graph

def test_dominance_directed_graph():
    rankings = np.array([[3, 2, 3],
                        [4, 4, 4],
                        [2, 3, 2],
                        [1, 1, 1]])
    compromised_ranking = dominance_directed_graph(rankings)
    assert np.array_equal(compromised_ranking, np.array([3, 4, 2, 1]))


def test_dominance_directed_graph_invalid_rankings():
    # This will raise a TypeError in the function
    rankings = "invalid_ranking"  

    with raises(TypeError):
        dominance_directed_graph(rankings)
