import numpy as np
import cupy as cp
from regularisers import l2

class DenseLayer:

    def __init__(self, layer_name, incoming_chans=None, output_dim=None, with_bias=True,
                 weight_regulariser=None, weight_initialiser="normal"):

        self.is_on_gpu = False
        self.layer_name = layer_name
        self.incoming_chans = incoming_chans
        self.output_dim = output_dim
        self.with_bias = with_bias
        self.weight_regulariser = weight_regulariser
        self.downstream_X = None
        self.weight_initialiser = weight_initialiser

        if incoming_chans is not None and output_dim is not None:
            if self.weight_initialiser == "glorot_uniform":
                limit = np.sqrt(6.0 / (self.incoming_chans + self.output_dim))
                weights = np.random.uniform(low=-limit,
                                            high=limit,
                                            size=(self.incoming_chans, 
                                                self.output_dim)).astype(np.float32)
            elif self.weight_initialiser == "normal":
                weights = 0.01*np.random.randn(self.incoming_chans, self.output_dim).astype(np.float32)
            self.learned_params = {"weights": weights}
            self.grads = {"weights": np.zeros_like(weights)}
            
            if with_bias:
                bias = np.zeros(output_dim).astype(np.float32)
                self.learned_params.update({"bias": bias})
                self.grads.update({"bias": np.zeros_like(bias, dtype=np.float32)})
        else:
            self.learned_params = {}
            self.grads = {}

    def __repr__(self):
        return "DenseLayer({}, incoming_chans={}, output_dim={}, weight_regulariser={})".format(
                                       self.layer_name,
                                       self.incoming_chans,
                                       self.output_dim,
                                       repr(self.weight_regulariser)
                                       )

    def to_gpu(self):
        if self.is_on_gpu:
            print("Layer {} is already on GPU, ignoring request".format(self.layer_name))
        else:
            # move learned_params and grads to gpu
            for k, v in self.learned_params.items():
                self.learned_params[k] = cp.asarray(self.learned_params[k])
            for k, v  in self.grads.items():
                self.grads[k] = cp.asarray(self.learned_params[k])
            self.is_on_gpu = True

    def forward(self, X, test_mode=False):
        xp = cp.get_array_module(X)
        if not test_mode:
            self.downstream_X = X

        out = xp.dot(X, self.learned_params["weights"])
        if self.with_bias:
            out += self.learned_params["bias"][xp.newaxis, :]

        return out

    def backward(self, upstream_dx):
        xp = cp.get_array_module(upstream_dx)
        if self.with_bias:
            self.grads["bias"] = xp.sum(upstream_dx, axis=0)
        self.grads["weights"] = xp.dot(self.downstream_X.T, upstream_dx)
        if self.weight_regulariser:
            self.grads["weights"] += (
                self.weight_regulariser.backward(self.learned_params["weights"])
            )

        return xp.dot(upstream_dx, self.learned_params["weights"].T)

    def regulariser_forward(self):
        out = 0
        if self.weight_regulariser:
            out += self.weight_regulariser.forward(self.learned_params["weights"])
        return out
        
    def save_to_h5(self, open_f, save_grads=True):
        base_dset = open_f.create_dataset(self.layer_name + "/layer_info", dtype=np.float32)
        base_dset.attrs["type"] = self.__class__.__name__ 
        base_dset.attrs["incoming_chans"] = self.incoming_chans
        base_dset.attrs["output_dim"] = self.output_dim
        base_dset.attrs["with_bias"] = self.with_bias

        dset = open_f.create_dataset(self.layer_name + "/weights", 
                                     self.learned_params['weights'].shape,
                                     dtype=self.learned_params['weights'].dtype)
        dset[:] = cp.asnumpy(self.learned_params['weights'])
        if self.weight_regulariser is not None:
            dset.attrs["weight_regulariser_type"] = np.string_(self.weight_regulariser.type)
            dset.attrs["weight_regulariser_strength"] = np.string_(self.weight_regulariser.strength)
        if self.with_bias:
            dset_bias = open_f.create_dataset(self.layer_name + "/bias",
                                              self.learned_params['bias'].shape,
                                              dtype=self.learned_params['bias'].dtype)
            dset_bias[:] = cp.asnumpy(self.learned_params['bias'])
        
        if save_grads:
            dset_grads = open_f.create_dataset(self.layer_name + "/grads/weights",
                                               self.grads['weights'].shape,
                                               dtype=self.learned_params['weights'].dtype)
            dset_grads[:] = cp.asnumpy(self.grads["weights"])
            if self.with_bias:
                dset_grads_bias = open_f.create_dataset(self.layer_name + "/grads/bias",
                                                        self.grads['bias'].shape,
                                                        dtype=self.learned_params['bias'].dtype)
                dset_grads_bias[:] = cp.asnumpy(self.grads["bias"])

    def load_from_h5(self, open_f, load_grads=True):
        self.incoming_chans = open_f[self.layer_name + '/layer_info'].attrs['incoming_chans']
        self.output_dim = open_f[self.layer_name + '/layer_info'].attrs['output_dim']
        self.with_bias = open_f[self.layer_name + '/layer_info'].attrs['with_bias']

        weight_regulariser_type = open_f[self.layer_name + '/weights'].attrs.get("weight_regulariser_type", None)
        if weight_regulariser_type:
            weight_regulariser_strength = open_f[self.layer_name + '/weights'].attrs["weight_regulariser_strength"]
            if weight_regulariser_type == b"l2":
                self.weight_regulariser = l2.l2(strength=float(weight_regulariser_strength))

        self.learned_params['weights'] = open_f[self.layer_name + '/weights'][:]
        if self.with_bias:
            self.learned_params['bias'] = open_f[self.layer_name + '/bias'][:]
        if load_grads:
            self.grads['weights'] = open_f[self.layer_name + '/grads/weights'][:]
            if self.with_bias:
                self.grads['bias'] = open_f[self.layer_name + '/grads/bias'][:]