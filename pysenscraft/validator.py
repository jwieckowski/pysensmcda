# Copyright (C) 2023 - 2024 Jakub Więckowski

import numpy as np
import inspect

def get_var_name(var):
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    args = string[string.find('(') + 1:-1].split(',')
    
    name = None
    for i in args:
        if i.find('=') != -1:
            name = i.split('=')[1].strip()        
        else:
            name = i
    return name

class Validator:
    @staticmethod
    def is_type_valid(var, type, custom_message = None):
        if not isinstance(var, type):
            if custom_message:
                raise TypeError(custom_message)
            else:
                raise TypeError(f"'{get_var_name(var)}' should be given as {type}")
        return True

    @staticmethod
    def is_callable(var, custom_message = None):
        if isinstance(var, (list, np.ndarray)):
            if any([not callable(v) for v in var]):
                if custom_message:
                    raise TypeError(custom_message)
                else:
                    raise TypeError(f"'{get_var_name(var)}' should be given as a list of callable")
        else:
            if not callable(var):
                if custom_message:
                    raise TypeError(custom_message)
                else:
                    raise TypeError(f"'{get_var_name(var)}' should be given as callable")
        return True

    @staticmethod
    def is_dimension_valid(var, size, custom_message = None):
        if var.ndim != size:
            if custom_message:
                raise ValueError(custom_message)
            else:
                raise ValueError(f"'{get_var_name(var)}' should be given as {size}D vector")
        return True
    
    @staticmethod
    def is_sum_valid(var, sum, precision=3, custom_message = None):
        if np.round(np.sum(var), precision) != sum:
            if custom_message:
                raise ValueError(custom_message)
            else:
                raise ValueError(f"'{get_var_name(var)}' should sum up to {sum}")
        return True

    @staticmethod
    def is_shape_equal(size1, size2, custom_message = None):
        if size1 != size2:
            if custom_message:
                raise ValueError(custom_message)
            else:
                raise ValueError(f"'{get_var_name(size1)}' and '{get_var_name(size1)}' have different shape")
        return True

    @staticmethod
    def are_indexes_valid(indexes, size, custom_message = None):
        if indexes is not None:
            if isinstance(indexes, int):
                if indexes >= size or indexes < 0:
                    if custom_message:
                        raise IndexError(custom_message)
                    else:
                        raise IndexError(f"'{get_var_name(size)}' out of range. Check element ({indexes})")
            else:
                for c_idx in indexes:
                    if isinstance(c_idx, (int, np.integer)):
                        if c_idx < 0 or c_idx >= size:
                            if custom_message:
                                raise IndexError(custom_message)
                            else:
                                raise IndexError(f"'{get_var_name(size)}' out of range. Check element ({c_idx})")
                    elif isinstance(c_idx, (list, np.ndarray)):
                        if any([idx < 0 or idx >= size for idx in c_idx]):
                            if custom_message:
                                raise IndexError(custom_message)
                            else:
                                raise IndexError(f"'{get_var_name(size)}' out of range. Check element ({c_idx})")
        return True

    @staticmethod
    def is_positive_value(var, bound=0, custom_message = None):
        if var <= bound:
            if custom_message:
                raise ValueError(custom_message)
            else:
                raise ValueError(f"'{get_var_name(var)}' should be a positive value")
        return True

    @staticmethod
    def is_in_range(var, min_val, max_val, custom_message = None):
        if isinstance(var, (list, np.ndarray)):
            if any([v < min_val or v > max_val for v in var]):
                if custom_message:
                    raise ValueError(custom_message)
                else:
                    raise ValueError(f"All values from '{get_var_name(var)}' should be in range [{min_val}, {max_val}]")
        else:
            if min_val > var or var > max_val:
                if custom_message:
                    raise ValueError(custom_message)
                else:
                    raise ValueError(f"'{get_var_name(var)}' should be in range [{min_val}, {max_val}]")
        return True

    @staticmethod
    def is_in_list(var, var_list, custom_message = None):
        if isinstance(var, (list, np.ndarray)):
            if any([v not in var_list for v in var]):
                if custom_message:
                    raise ValueError(custom_message)
                else:
                    raise ValueError(f"'{get_var_name(var)}' should contain only values from {var_list}")
        else:
            if var not in var_list:
                if custom_message:
                    raise ValueError(custom_message)
                else:
                    raise ValueError(f"'{get_var_name(var)}' should be one of the values from {var_list}")
        return True

    @staticmethod
    def is_key_in_dict(keys, dict, custom_message = None):
        for key in keys:
            if key not in list(dict.keys()):
                if custom_message:
                    raise ValueError(custom_message)
                else:
                    raise ValueError(f"'{get_var_name(dict)}' should include `{key}` as one of the keys")
        return True
        
    @staticmethod
    def is_array_2D_3D(var, ref_var, custom_message = None):
        dim = 0
        try: # 2D
            if isinstance(var[0][0], (int, np.integer, float)):
                shapes = tuple(len(dv) for dv in var)
                if ref_var.shape[1] != len(shapes):
                    if custom_message:
                        raise TypeError(custom_message)
                    else:
                        raise TypeError(f"'{get_var_name(ref_var)}' and '{get_var_name(var)}' have different shapes")
                dim = 2
            else: # 3D
                dv_shape = [len(tuple(len(vals) for vals in dv)) for dv in var]
                if len(np.unique(dv_shape)) != 1 or ref_var.shape[0] != dv_shape[0] and ref_var.shape[1] != dv_shape[1]:
                    if custom_message:
                        raise TypeError(custom_message)
                    else:
                        raise TypeError(f"'{get_var_name(ref_var)}' and '{get_var_name(var)}' have different shapes")
                dim = 3
        except TypeError: 
            if custom_message:
                raise TypeError(custom_message)
            else:
                raise TypeError(f"'{get_var_name(var)}' should be given as 2D or 3D array")

        return True, dim
        
    @staticmethod
    def is_type_in_dict_valid(key, dict, type, custom_message = None):
        if not isinstance(dict[key], type):
            if custom_message:
                raise TypeError(custom_message)
            else:
                raise TypeError(f"'{key}' in '{get_var_name(dict)}' should be given as {type}")
        return True
