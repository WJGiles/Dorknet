import numpy as np
import numexpr as ne
import cupy as cp
import batch_norm_stats_cy
ne.set_vml_accuracy_mode('low')
profile = lambda x: x

class BatchNormLayer():
    """
    https://arxiv.org/pdf/1502.03167.pdf
    """
    def __init__(self, layer_name, input_dimension=4, 
                 incoming_chans=None, run_momentum=0.95, is_on_gpu=True):
        """
        : input_dimension: should be 4 if following eg. convolution, 2 for eg. dense layer
        : incoming_chans: is number of feature maps (channels) for conv layer, 
        or features (cols) for dense layer 
        """
        self.is_on_gpu = False
        self.layer_name = layer_name
        self.eps = 1e-5 # Fuzz factor for numerical stability
        self.input_dimension = input_dimension
        self.non_learned_params = {"running_mean": None,
                                   "running_std": None}
        self.run_momentum = run_momentum
        if self.input_dimension not in {2, 4}:
            raise ValueError("BatchNorm input_dimension should have length 2 or 4...")
        if self.input_dimension == 4:
            self.av_axis = (0, 2, 3)
        elif self.input_dimension == 2:
            self.av_axis = 0
        self.incoming_chans = incoming_chans
        if incoming_chans is not None:
            gamma = np.ones(incoming_chans, dtype=np.float32) # Scale
            beta = np.zeros(incoming_chans, dtype=np.float32) # Shift
            if self.input_dimension == 4:
                gamma = gamma[np.newaxis, :, np.newaxis, np.newaxis]
                beta = beta[np.newaxis, :, np.newaxis, np.newaxis]
            
            self.learned_params = {"gamma": gamma,
                                "beta": beta}
            self.grads = {"gamma": np.zeros_like(gamma).astype(np.float32),
                        "beta": np.zeros_like(beta).astype(np.float32)}
        else:
            self.learned_params = {}
            self.grads = {}

    def __repr__(self):
        return "BatchNormLayer({}, input_dimension={}, incoming_chans={}, run_momentum={})".format(
            self.layer_name, self.input_dimension, self.incoming_chans, self.run_momentum
        )

    def to_gpu(self):
        if self.is_on_gpu:
            print("Layer {} is already on GPU, ignoring request".format(self.layer_name))
        else:
            # move learned_params and grads to gpu
            for k, v in self.learned_params.items():
                self.learned_params[k] = cp.asarray(v)
            for k, v in self.non_learned_params.items():
                if v is not None:
                    self.non_learned_params[k] = cp.asarray(v)
            for k, v  in self.grads.items():
                self.grads[k] = cp.asarray(v)
            
            self.is_on_gpu = True

    @profile
    def forward(self, X, test_mode=False, use_express=False):
        """
        X.shape = (batch_size, channel, height, width)
        Note that when following a convolution layer, we learn a (gamma, beta)
        for each of the channels (feature maps).
        """
        self.input_shape = X.shape
        xp = cp.get_array_module(X)

        if not test_mode:
            if len(X.shape) == 4 and not self.is_on_gpu:
                mean, var = batch_norm_stats_cy.channelwise_mean_and_var_4d(X)
            else:
                mean = xp.mean(X, axis=self.av_axis)
                var = xp.var(X, axis=self.av_axis)
            self.std = xp.sqrt(var + self.eps)
            if len(X.shape) == 4:
                self.std = self.std[xp.newaxis, :, xp.newaxis, xp.newaxis]
                mean = mean[xp.newaxis, :, xp.newaxis, xp.newaxis]
            self.X_demean = X - mean
            self.X_hat = self.X_demean/self.std
            
            if self.non_learned_params["running_mean"] is not None:
                self.non_learned_params["running_mean"] = (
                    self.run_momentum*self.non_learned_params["running_mean"] + 
                    (1 - self.run_momentum)*mean
                )
            else:
                self.non_learned_params["running_mean"] = mean
            if self.non_learned_params["running_std"] is not None:
                self.non_learned_params["running_std"] = (
                    self.run_momentum*self.non_learned_params["running_std"] + 
                    (1 - self.run_momentum)*self.std
                )
            else:
                self.non_learned_params["running_std"] = self.std
            if use_express:
                return (
                    ne.evaluate("gamma*X_hat + beta",
                                local_dict={**vars(self), 
                                            'gamma': self.learned_params['gamma'],
                                            'beta' : self.learned_params['beta']})
                )
            else:
                return (
                    self.learned_params['gamma']*self.X_hat + self.learned_params['beta']
                )
        else: # test_mode
            if use_express:
                X_hat = ne.evaluate("(X - running_mean)/running_std",
                                    local_dict={'running_mean': self.non_learned_params["running_mean"],
                                                'running_std': self.non_learned_params["running_std"]})
                return (
                    ne.evaluate("gamma*X_hat + beta",
                                local_dict={'gamma': self.learned_params['gamma'],
                                            'beta' : self.learned_params['beta']})
                )
            else:
                X_hat = (X - self.non_learned_params["running_mean"])/self.non_learned_params["running_std"]
                return (
                    self.learned_params['gamma']*X_hat + self.learned_params['beta']
                )

    @profile
    def backward(self, upstream_dx):
        self.grads["gamma"] = self.dgamma(upstream_dx)
        self.grads["beta"] = self.dbeta(upstream_dx)

        return self.dx(upstream_dx)

    @profile
    def dx(self, upstream_dx, use_express=False):
        xp = cp.get_array_module(upstream_dx)
        upstream_mean = xp.mean(upstream_dx, axis=self.av_axis)
        self.std_recip = 1.0/self.std

        if self.input_dimension == 4:
            upstream_mean = upstream_mean[xp.newaxis, :, xp.newaxis, xp.newaxis]
        self.effective_batch_size = float(
            self.input_shape[0]*self.input_shape[2]*self.input_shape[3] 
            if len(self.input_shape) == 4 
            else self.input_shape[0]
        )
        factor = self.learned_params["gamma"]*self.std_recip
        if use_express:
            other = ne.evaluate("(1.0/effective_batch_size)*(X_demean*(std_recip**2))",
                                local_dict=vars(self))
        else:
            other = (1.0/float(self.effective_batch_size))*(self.X_demean*(self.std_recip**2))
        
        if not self.is_on_gpu:
            dot_sum = xp.einsum("ijkl,ijkl->j", upstream_dx, self.X_demean)
        else:
            dot_sum = xp.sum(upstream_dx*self.X_demean, axis=self.av_axis)
        if len(self.input_shape) == 4:
            dot_sum = dot_sum[xp.newaxis, :, xp.newaxis, xp.newaxis]

        if use_express:
            out = ne.evaluate("factor*(upstream_dx - upstream_mean - other*dot_sum)")
        else:
            out = factor*(upstream_dx - upstream_mean - other*dot_sum)
        
        return out.astype(xp.float32)

    @profile
    def dgamma(self, upstream_dx):
        xp = cp.get_array_module(upstream_dx)
        if not self.is_on_gpu:
            dgamma = xp.einsum("ijkl,ijkl->j", upstream_dx, self.X_hat)
        else:
            dgamma = xp.sum(upstream_dx*self.X_hat, axis=self.av_axis)
        if self.input_dimension == 4:
            dgamma = dgamma[xp.newaxis, :, xp.newaxis, xp.newaxis]
        return dgamma

    def dbeta(self, upstream_dx):
        xp = cp.get_array_module(upstream_dx)
        dbeta = xp.sum(upstream_dx, axis=self.av_axis)
        if self.input_dimension == 4:
            dbeta = dbeta[xp.newaxis, :, xp.newaxis, xp.newaxis]
        return dbeta

    def save_to_h5(self, open_f, save_grads=True):
        base_dset = open_f.create_dataset(self.layer_name + "/layer_info", dtype=np.float32)
        base_dset.attrs["type"] = self.__class__.__name__
        base_dset.attrs["input_dimension"] = self.input_dimension
        base_dset.attrs["run_momentum"] = self.run_momentum
        base_dset.attrs["incoming_chans"] = self.incoming_chans
        base_dset.attrs["eps"] = self.eps
        dset = open_f.create_dataset(self.layer_name + "/gamma", 
                                     self.learned_params['gamma'].shape,
                                     dtype=self.learned_params['gamma'].dtype)
        dset[:] = cp.asnumpy(self.learned_params['gamma'])

        dset_beta = open_f.create_dataset(self.layer_name + "/beta",
                                            self.learned_params['beta'].shape,
                                            dtype=self.learned_params['beta'].dtype)
        dset_beta[:] = cp.asnumpy(self.learned_params['beta'])

        dset_running_mean = open_f.create_dataset(self.layer_name + "/running_mean",
                                            self.non_learned_params["running_mean"].shape,
                                            dtype=self.non_learned_params["running_mean"].dtype)
        dset_running_mean[:] = cp.asnumpy(self.non_learned_params["running_mean"])

        dset_running_std = open_f.create_dataset(self.layer_name + "/running_std",
                                            self.non_learned_params["running_std"].shape,
                                            dtype=self.non_learned_params["running_std"].dtype)
        dset_running_std[:] = cp.asnumpy(self.non_learned_params["running_std"])
        
        if save_grads:
            dset_grads = open_f.create_dataset(self.layer_name + "/grads/gamma",
                                               self.grads['gamma'].shape,
                                               dtype=self.learned_params['gamma'].dtype)
            dset_grads[:] = cp.asnumpy(self.grads["gamma"])
            dset_grads_beta = open_f.create_dataset(self.layer_name + "/grads/beta",
                                                    self.grads['beta'].shape,
                                                    dtype=self.learned_params['beta'].dtype)
            dset_grads_beta[:] = cp.asnumpy(self.grads["beta"])

    def load_from_h5(self, open_f, load_grads=True):
        self.eps = open_f[self.layer_name + '/layer_info'].attrs['eps']
        self.incoming_chans = open_f[self.layer_name + '/layer_info'].attrs['incoming_chans']
        self.input_dimension = open_f[self.layer_name + '/layer_info'].attrs['input_dimension']
        self.run_momentum = open_f[self.layer_name + '/layer_info'].attrs['run_momentum']

        if self.input_dimension not in {2, 4}:
            raise ValueError("BatchNorm input_dimension should have length 2 or 4...")
        if self.input_dimension == 4:
            self.av_axis = (0, 2, 3)
        elif self.input_dimension == 2:
            self.av_axis = 0

        self.learned_params['gamma'] = open_f[self.layer_name + '/gamma'][:]
        self.learned_params['beta'] = open_f[self.layer_name + '/beta'][:]
        self.non_learned_params["running_mean"] = cp.asarray(open_f[self.layer_name + '/running_mean'][:])
        self.non_learned_params["running_std"] = cp.asarray(open_f[self.layer_name + '/running_std'][:])
        if load_grads:
            self.grads['gamma'] = open_f[self.layer_name + '/grads/gamma'][:]
            self.grads['beta'] = open_f[self.layer_name + '/grads/beta'][:]
