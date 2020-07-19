class SGD:
    def __init__(self, network, learning_rate):
        self.network = network
        self.learnable_layers = [l for l in network.layers if hasattr(l, "learned_params")]
        self.learning_rate = learning_rate

    def set_learning_rate(self, new_lr):
        self.learning_rate = new_lr

    def multiply_learning_rate(self, multiplier):
        self.learning_rate *= multiplier

    def update_weights(self):
        for layer in self.learnable_layers:
            if hasattr(layer, "layer_list"):
                for l in layer.layer_list:
                    for param in l.learned_params.keys():
                        print(l.layer_name, param)
                        dx = -self.learning_rate * l.grads[param]
                        l.learned_params[param] += dx
            else:
                for param in layer.learned_params.keys():
                    dx = -self.learning_rate * layer.grads[param]
                    layer.learned_params[param] += dx