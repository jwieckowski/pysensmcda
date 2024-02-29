# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.criteria.random_distribution import *

def test_chisquare_distribution_default():
    weights = chisquare_distribution(3)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

def test_chisquare_distribution_explicit():
    weights = chisquare_distribution(3, 5)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

def test_laplace_distribution_default():
    weights = laplace_distribution(3)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

def test_laplace_distribution_explicit():
    weights = laplace_distribution(3, 5, 2)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

def test_normal_distribution_default():
    weights = normal_distribution(3)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

def test_normal_distribution_explicit():
    weights = normal_distribution(3, 5, 2)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

def test_random_distribution_default():
    weights = random_distribution(3)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

def test_triangular_distribution_default():
    weights = triangular_distribution(3)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

def test_triangular_distribution_explicit():
    weights = triangular_distribution(3, 2, 5, 6)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

def test_uniform_distribution_default():
    weights = uniform_distribution(3)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

def test_uniform_distribution_explicit():
    weights = uniform_distribution(3, 2, 5)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

def test_distribution_error():
    with raises(TypeError):
        random_distribution('wrong type parameter')