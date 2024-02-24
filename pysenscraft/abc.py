# Copyright (C) 2024 Jakub Więckowski

from abc import ABC, abstractmethod

class SensitivityAnalysis(ABC):
    """
    Abstract base class for Sensitivity Analysis.

    Methods:
    --------
    - __init__(self)
        Constructor for the SensitivityAnalysis class.

    - print_results(self)
        Print formatted results from sensitivity analysis
    """

    def __init__(self):
        pass

    @abstractmethod
    def print_results(self):
        pass
