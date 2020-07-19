import numpy as np
import cupy as cp

class l2:
    def __init__(self, strength=0.005):
        self.type = "l2"
        self.strength = strength
    
    def __repr__(self):
        return "l2(strength={})".format(self.strength)

    def forward(self, X):
        xp = cp.get_array_module(X)
        return 0.5*self.strength*xp.sum(xp.power(X,2))

    def backward(self, X):
        return self.strength*X
