class SGD:
    def __init__(self, network, learning_rate):
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

    def set_learning_rate(self, new_lr):
        self.learning_rate = new_lr

    def multiply_learning_rate(self, multiplier):
        self.learning_rate *= multiplier

    def update_weights(self):
        for layer in self.learnable_layers:
            for param in layer.learned_params.keys():
                dx = -self.learning_rate * layer.grads[param]
                layer.learned_params[param] += dx