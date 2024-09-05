""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""Abstract classes used to standardize predictor models"""

from abc import ABC, abstractmethod


class PredictorModel(ABC):
    @abstractmethod
    def predict(self, Y, U=None):
        pass
