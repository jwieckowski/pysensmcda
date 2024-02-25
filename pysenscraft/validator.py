# Copyright (C) 2023 - 2024 Jakub Więckowski

import numpy as np

def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name

class Validator:
    @staticmethod
    def is_type_valid(var, type):
        if not isinstance(var, type):
            raise TypeError(f"'{get_var_name(var)}' should be given as {type}")

    @staticmethod
    def is_dimension_valid(var, size):
        if var.ndim != size:
            raise ValueError(f"'{get_var_name(var)}' should be given as {size}D vector")
    
    @staticmethod
    def is_sum_valid(var, sum, precision=3):
        if np.round(np.sum(var), precision) != sum:
            raise ValueError(f"'{get_var_name(var)}' should sum up to {sum}")


# examples
range_values = 1
print(Validator.is_type_valid(range_values, float))