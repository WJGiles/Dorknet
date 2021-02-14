import numpy as np
import cupy as cp

class SGDMomentum:
    def __init__(self, network, learning_rate, momentum):
        self.network = network
        self.learnable_layers = []
        for layer in network.layers:
            if layer.learned_params is not None:
                self.learnable_layers.append(layer)
            if hasattr(layer, "layer_list"): # For composite layers like ResidualBlock
                for l in layer.layer_list:
                    if l.learned_params is not None:
                        self.learnable_layers.append(l)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.grad_cache = {}
        for layer in self.learnable_layers:
            layer_d = {}
            for k,v in layer.grads.items():
                xp = cp.get_array_module(v)
                layer_d[k] = xp.zeros_like(v)
            self.grad_cache[layer] = layer_d
    
    def set_learning_rate(self, new_lr):
        self.learning_rate = new_lr

    def multiply_learning_rate(self, multiplier):
        self.learning_rate *= multiplier

    def update_weights(self):
        for layer in self.learnable_layers:
            for param in layer.learned_params.keys():
                dx = (
                    -self.learning_rate * layer.grads[param] + 
                     self.momentum * self.grad_cache[layer][param]
                )
                layer.learned_params[param] += dx
                self.grad_cache[layer][param] = dx