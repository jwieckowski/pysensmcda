# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.validator import Validator

def test_is_type_valid():
    # Should not raise an exception
    Validator.is_type_valid(42, int)
    # Should raise an exception
    with raises(TypeError):
        Validator.is_type_valid(42, str)

def test_is_callable():
    # Should not raise an exception
    Validator.is_callable(print)
    # Should raise an exception
    with raises(TypeError):
        Validator.is_callable(42)

def test_is_dimension_valid():
    # Should not raise an exception
    Validator.is_dimension_valid(np.array([1, 2, 3]), 1)
    # Should raise an exception
    with raises(ValueError):
        Validator.is_dimension_valid(np.array([[1, 2], [3, 4]]), 1)

def test_is_sum_valid():
    # Should not raise an exception
    Validator.is_sum_valid(np.array([0.3, 0.4, 0.3]), 1)
    # Should raise an exception
    with raises(ValueError):
        Validator.is_sum_valid(np.array([1, 2, 3]), 5)

def test_is_shape_equal():
    # Should not raise an exception
    Validator.is_shape_equal(2, 2)
    # Should raise an exception
    with raises(ValueError):
        Validator.is_shape_equal(2, 5)

def test_are_indexes_valid():
    # Should not raise an exception
    Validator.are_indexes_valid(np.array([0, 1, 2]), 3)
    Validator.are_indexes_valid(np.array([[0, 2], 1, 2], dtype='object'), 3)
    # Should raise an exception
    with raises(IndexError):
        Validator.are_indexes_valid(np.array([0, 1, 2]), 2)

def test_is_positive_value():
    # Should not raise an exception
    Validator.is_positive_value(1)
    Validator.is_positive_value(0, -1)
    # Should raise an exception
    with raises(ValueError):
        Validator.is_positive_value(0)

def test_is_in_range():
    # Should not raise an exception
    Validator.is_in_range(3, 1, 5)
    # Should raise an exception
    with raises(ValueError):
        Validator.is_in_range(8, 1, 5)

def test_is_in_list():
    # Should not raise an exception
    Validator.is_in_list([1, 2, 3], [1, 2, 3, 4, 5])
    # Should raise an exception
    with raises(ValueError):
        Validator.is_in_list([1, 6], [1, 2, 3, 4, 5])

def test_is_key_in_dict():
    # Should not raise an exception
    Validator.is_key_in_dict(['key1', 'key2'], {'key1': 42, 'key2': 24})
    # Should raise an exception
    with raises(ValueError):
        Validator.is_key_in_dict(['key1', 'key3'], {'key1': 42, 'key2': 24})

def test_is_array_2D_3D():
    # Should not raise an exception
    # 2D numpy array dtype object
    discrete_values = np.array([[2, 3, 4], [1, 5, 6], [3, 4]], dtype='object')
    matrix = np.array([
        [4, 1, 6],
        [2, 6, 3],
        [9, 5, 7],
    ])
    Validator.is_array_2D_3D(discrete_values, matrix)

    # 3D numpy array dtype object
    discrete_values = np.array([
        [[5, 6], [2, 4], [5, 8]],
        [[3, 5.5], [4], [3.5, 4.5]],
        [[7, 8], [6], [8, 9]],
    ], dtype='object')
    matrix = np.array([
        [4, 1, 6],
        [2, 6, 3],
        [9, 5, 7],
    ])
    Validator.is_array_2D_3D(discrete_values, matrix)
    # Should raise an exception
    with raises(TypeError):
        Validator.is_array_2D_3D(np.array([[[0, 1], 1], [1, 2, 3]], dtype='object'), matrix)

def test_is_type_in_dict_valid():
    # Should not raise an exception
    test_dict = {'key': 42}
    Validator.is_type_in_dict_valid('key', test_dict, int)  # Should not raise an exception
    # Should raise an exception
    with raises(TypeError):
        Validator.is_type_in_dict_valid('key', test_dict, str)
