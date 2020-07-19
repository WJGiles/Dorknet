import numpy as np
import cupy as cp
from regularisers import l2

class PointwiseConvLayer:
    def __init__(self, layer_name, stride=1, filter_block_shape=None, with_bias=True,
                 weight_regulariser=None, weight_initialiser="normal"):
        """
        filter_block_shape = (num_filters, num_incoming_channels)
        """
        self.use_cp = True
        self.layer_name = layer_name
        self.stride = stride
        self.with_bias = with_bias
        self.weight_regulariser = weight_regulariser
        self.weight_initialiser = weight_initialiser
        self.numpy_or_cupy = np #cp if self.use_cp else np
        if filter_block_shape is not None:
            self.num_filters, self.num_channels = filter_block_shape
            if self.weight_initialiser == "glorot_uniform":
                limit = self.numpy_or_cupy.sqrt(6.0 / (self.num_channels + self.num_filters))
                weights = self.numpy_or_cupy.random.uniform(low=-limit,high=limit,size=filter_block_shape).astype(np.float32)
            elif self.weight_initialiser == "normal":
                weights = 0.01*self.numpy_or_cupy.random.randn(*filter_block_shape).astype(self.numpy_or_cupy.float32)
            self.learned_params = {"weights": weights}
            self.grads = {"weights": self.numpy_or_cupy.zeros_like(weights).astype(np.float32)}
            if with_bias:
                bias = self.numpy_or_cupy.zeros(self.num_filters).astype(np.float32)
                self.learned_params.update({"bias": bias})
                self.grads.update({"bias": self.numpy_or_cupy.zeros_like(bias)})
        else:
            self.num_filters = None
            self.learned_params = {}
            self.grads = {}
        
    def __repr__(self):
        out = "PointwiseConvLayer({}, ".format(self.layer_name)
        if self.num_filters is not None:
            out += "filter_block_shape=({}, {}), ".format(self.num_filters,
                                                          self.num_channels)
        out += "stride={}, with_bias={}, weight_regulariser={}, use_cp={})".format(self.stride, 
                                                                        self.with_bias,
                                                                        repr(self.weight_regulariser),
                                                                        self.use_cp)
        return out

    def forward(self, X, test_mode=False):
        xp = cp.get_array_module(X)
        if self.stride > 1:
            X = X[:,:,::self.stride,::self.stride]
        self.patches = X.transpose(0,2,3,1).reshape(-1,self.learned_params["weights"].shape[1])
        if self.use_cp:
            out = cp.dot(cp.asarray(self.patches), cp.asarray(self.learned_params["weights"].T))
            if self.with_bias:
                out += cp.asarray(self.learned_params["bias"].reshape(1,-1))
        else:
            out = xp.dot(self.patches, self.learned_params["weights"].T)
            if self.with_bias:
                out += self.learned_params["bias"].reshape(1,-1)
        return out.reshape(X.shape[0], X.shape[2], X.shape[3], self.learned_params["weights"].shape[0]).transpose(0,3,1,2)

    def backward(self, upstream_dx):
        xp = cp.get_array_module(upstream_dx)
        if self.with_bias:
            self.grads["bias"] = xp.sum(upstream_dx, axis=(0,2,3))
        upstream = upstream_dx.transpose(0,2,3,1).reshape(self.patches.shape[0], -1)
        if self.use_cp:
            self.grads["weights"] = cp.asnumpy(cp.dot(cp.asarray(upstream.T), cp.asarray(self.patches))).reshape(self.learned_params["weights"].shape)
        else:
            self.grads["weights"] = xp.dot(upstream.T, self.patches).reshape(self.learned_params["weights"].shape)
        if self.weight_regulariser:
            self.grads["weights"] += self.weight_regulariser.backward(self.learned_params["weights"])
        if self.use_cp:
            dx_rows = cp.dot(cp.asarray(upstream), cp.asarray(self.learned_params["weights"]))
        else:
            dx_rows = xp.dot(upstream, cp.asarray(self.learned_params["weights"]))
        dx = dx_rows.reshape(upstream_dx.shape[0], upstream_dx.shape[2], upstream_dx.shape[3], self.num_channels).transpose(0,3,1,2)

        if self.stride > 1:
            dx_widened = xp.zeros((dx.shape[0], dx.shape[1], dx.shape[2]*self.stride, dx.shape[3]*self.stride), dtype=np.float32)
            dx_widened[:,:,::self.stride,::self.stride] = dx

            return dx_widened
        else:

            return dx

    def regulariser_forward(self):
        out = 0
        if self.weight_regulariser:
            out += self.weight_regulariser.forward(self.learned_params["weights"])
        return out

    def save_to_h5(self, open_f, save_grads=True):
        base_dset = open_f.create_dataset(self.layer_name + "/layer_info", dtype=np.float32)
        base_dset.attrs["type"] = self.__class__.__name__ 
        base_dset.attrs["with_bias"] = self.with_bias
        base_dset.attrs["num_filters"] = self.num_filters
        base_dset.attrs["num_channels"] = self.num_channels
        base_dset.attrs["stride"] = self.stride
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
        self.num_filters = open_f[self.layer_name + '/layer_info'].attrs['num_filters']
        self.num_channels = open_f[self.layer_name + '/layer_info'].attrs['num_channels']
        stride = open_f[self.layer_name + '/layer_info'].attrs.get('stride', None)
        if stride:
            self.stride = stride
        else:
            self.stride = 1
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