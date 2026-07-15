import pymc.math as pm
from .WlModel import WlModel

class WlModel_pymc(WlModel):
    def dot(self, a, b):
        return pm.dot(a, b)
    
    def log(self, x):
        return pm.log(x)
    
    def cumsum(self, array):
        return pm.cumsum(array)
    
    def concatenate(self, arrays):
        return pm.concatenate(arrays)