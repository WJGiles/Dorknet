import numpy as np
from multiprocessing import Pool
profile = lambda x: x

class RMSProp:
    def __init__(self, network, learning_rate, decay_rate):
        self.network = network
        self.learnable_layers = []
        for layer in network.layers:
            if layer.learned_params is not None:
                self.learnable_layers.append(layer)
            if hasattr(layer, "layer_list"): # For composite layers like ResidualBlock
                for l in layer.layer_list:
                    if layer.learned_params is not None:
                        self.learnable_layers.append(layer)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.grad_cache = {layer: {k: np.zeros_like(v) for k, v in layer.grads.items()}
                                                       for layer in self.learnable_layers}

    def set_learning_rate(self, new_lr):
        self.learning_rate = new_lr

    def multiply_learning_rate(self, multiplier):
        self.learning_rate *= multiplier

    @profile
    def update_weights(self):
        for layer in self.learnable_layers:
            for param in layer.learned_params.keys():
                self.grad_cache[layer][param] = (
                    self.decay_rate*self.grad_cache[layer][param] + 
                    (1 - self.decay_rate)*np.power(layer.grads[param],2)
                )
                dx = - self.learning_rate*layer.grads[param]/np.sqrt(self.grad_cache[layer][param] + 1e-5)
                layer.learned_params[param] += dx
