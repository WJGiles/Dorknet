import cupy as cp

class Layer:

    def __init__(self, layer_name, *args, **kwargs):
        self.layer_name = layer_name
        self.is_on_gpu = False
        self.learned_params = None
        self.non_learned_params = None
        self.grads = None
        self.weight_regulariser = None

    def __repr__(self):
        return "Layer of type {} didn't implement __repr__".format(
            self.__class__.__name__
        )
    
    def to_gpu(self):
        if self.is_on_gpu:
            print("Layer {} is already on GPU, ignoring request".format(self.layer_name))
        else:
            # move learned_params, non_learned_params and grads to gpu
            if self.learned_params is not None:
                for k, v in self.learned_params.items():
                    self.learned_params[k] = cp.asarray(v)
            if self.non_learned_params is not None:
                for k, v in self.non_learned_params.items():
                    if v is not None:
                        self.non_learned_params[k] = cp.asarray(v)
            if self.grads is not None:
                for k, v  in self.grads.items():
                    self.grads[k] = cp.asarray(v)
            
            self.is_on_gpu = True

    def forward(self, X, *args, test_mode=False, **kwargs):
        pass

    def backward(self, upstream_dx, *args, **kwargs):
        pass

    def regulariser_forward(self):
        out = 0
        if self.weight_regulariser:
            out += self.weight_regulariser.forward(self.learned_params["weights"])
        return out