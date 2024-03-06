# Copyright (C) 2023 Jakub Więckowski
import functools

def memory_guard(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)

        except MemoryError:
            print("Insufficient memory to perform the operation. Please try with different parameters")

    return wrapper