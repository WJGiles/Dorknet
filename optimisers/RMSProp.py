import numpy as np
from multiprocessing import Pool
profile = lambda x: x

class RMSProp:
    def __init__(self, network, learning_rate, decay_rate):
        self.network = network
        self.num_workers = 2
        self.learnable_layers = [l for l in network.layers if hasattr(l, "learned_params")]
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.grad_cache = {layer: {k: np.zeros_like(v) for k, v in layer.grads.items()}
                                                       for layer in self.learnable_layers}

    def set_learning_rate(self, new_lr):
        self.learning_rate = new_lr

    def multiply_learning_rate(self, multiplier):
        self.learning_rate *= multiplier

    def update_layer(self, layer):
        for param in layer.learned_params.keys():
            self.grad_cache[layer][param] = (
                self.decay_rate*self.grad_cache[layer][param] + 
                (1 - self.decay_rate)*np.power(layer.grads[param],2)
            )
            dx = - self.learning_rate*layer.grads[param]/np.sqrt(self.grad_cache[layer][param] + 1e-5)
            layer.learned_params[param] += dx

    @profile
    def update_weights(self):
        #with Pool(self.num_workers) as p:
        #    p.map(self.update_layer, self.learnable_layers)
        for layer in self.learnable_layers:
            self.update_layer(layer)
