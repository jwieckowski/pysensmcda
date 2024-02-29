# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.criteria import generate_weights_scenarios

def test_generate_weights_scenarios_parallel_with_array_return():
    scenarios = generate_weights_scenarios(4, 0.1, 3, return_array=True)
    assert isinstance(scenarios, np.ndarray)
    assert len(scenarios[0]) == 4

def test_generate_weights_scenarios_sequential():
    scenarios = generate_weights_scenarios(4, 0.1, 3, sequential=True, return_array=True)
    assert isinstance(scenarios, np.ndarray)
    assert len(scenarios[0]) == 4

def test_generate_weights_scenarios_error():
    with raises(TypeError):
        generate_weights_scenarios(4.5, 0.1, 3, return_array=True)