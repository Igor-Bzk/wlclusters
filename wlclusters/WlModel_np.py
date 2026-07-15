import numpy as np
from .WlModel import WlModel

class WlModel_np(WlModel):
    def dot(self, a, b):
        return np.dot(a, b)
    
    def log(self, x):
        return np.log(x)
    
    def cumsum(self, array):
        return np.cumsum(array)
    
    def concatenate(self, arrays):
        return np.concatenate(arrays)