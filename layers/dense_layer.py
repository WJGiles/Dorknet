import numpy as np
import cupy as cp
from regularisers import l2

class DenseLayer:

    def __init__(self, layer_name, incoming_chans=None, output_dim=None, with_bias=True,
                 weight_regulariser=None, weight_initialiser="normal"):
        self.use_cp = True
        self.layer_name = layer_name
        self.incoming_chans = incoming_chans
        self.output_dim = output_dim
        self.with_bias = with_bias
        self.weight_regulariser = weight_regulariser
        self.numpy_or_cupy = np #cp if self.use_cp else np
        self.downstream_X = None
        self.weight_initialiser = weight_initialiser

        if incoming_chans is not None and output_dim is not None:
            if self.weight_initialiser == "glorot_uniform":
                limit = self.numpy_or_cupy.sqrt(6.0 / (self.incoming_chans + self.output_dim))
                weights = self.numpy_or_cupy.random.uniform(low=-limit,
                                            high=limit,
                                            size=(self.incoming_chans, 
                                                self.output_dim)).astype(self.numpy_or_cupy.float32)
            elif self.weight_initialiser == "normal":
                weights = 0.01*self.numpy_or_cupy.random.randn(self.incoming_chans, self.output_dim).astype(self.numpy_or_cupy.float32)
            self.learned_params = {"weights": weights}
            self.grads = {"weights": self.numpy_or_cupy.zeros_like(weights)}
            
            if with_bias:
                bias = self.numpy_or_cupy.zeros(output_dim).astype(self.numpy_or_cupy.float32)
                self.learned_params.update({"bias": bias})
                self.grads.update({"bias": self.numpy_or_cupy.zeros_like(bias, dtype=self.numpy_or_cupy.float32)})
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

    def forward(self, X, test_mode=False):
        xp = cp.get_array_module(X)
        if not test_mode:
            self.downstream_X = X
        if self.use_cp:
            out = xp.dot(X, cp.asarray(self.learned_params["weights"]))
            if self.with_bias:
                out += cp.asarray(self.learned_params["bias"][xp.newaxis, :])
        else: 
            out = xp.dot(X, self.learned_params["weights"])
            if self.with_bias:
                out += self.learned_params["bias"][xp.newaxis, :]
        return out

    def backward(self, upstream_dx):
        xp = cp.get_array_module(upstream_dx)
        if self.with_bias:
            self.grads["bias"] = cp.asnumpy(xp.sum(upstream_dx, axis=0))
        self.grads["weights"] = cp.asnumpy(xp.dot(self.downstream_X.T, upstream_dx))
        if self.weight_regulariser:
            self.grads["weights"] += (
                self.weight_regulariser.backward(self.learned_params["weights"])
            )
        if self.use_cp:
            return xp.dot(upstream_dx, cp.asarray(self.learned_params["weights"].T))
        else:
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
        dset[:] = self.learned_params['weights']
        if self.weight_regulariser is not None:
            dset.attrs["weight_regulariser_type"] = np.string_(self.weight_regulariser.type)
            dset.attrs["weight_regulariser_strength"] = np.string_(self.weight_regulariser.strength)
        if self.with_bias:
            dset_bias = open_f.create_dataset(self.layer_name + "/bias",
                                              self.learned_params['bias'].shape,
                                              dtype=self.learned_params['bias'].dtype)
            dset_bias[:] = self.learned_params['bias']
        
        if save_grads:
            dset_grads = open_f.create_dataset(self.layer_name + "/grads/weights",
                                               self.grads['weights'].shape,
                                               dtype=self.learned_params['weights'].dtype)
            dset_grads[:] = self.grads["weights"]
            if self.with_bias:
                dset_grads_bias = open_f.create_dataset(self.layer_name + "/grads/bias",
                                                        self.grads['bias'].shape,
                                                        dtype=self.learned_params['bias'].dtype)
                dset_grads_bias[:] = self.grads["bias"]

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