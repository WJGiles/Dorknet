import numpy as np
import cupy as cp
import relu_cy

class ReLu:

    def __init__(self, layer_name):
        self.is_on_gpu = False
        self.layer_name = layer_name

    def __repr__(self):
        return "ReLu({})".format(self.layer_name)

    def to_gpu(self):
        self.is_on_gpu = True

    def forward(self, X, test_mode=False):
        if self.is_on_gpu:
            out = self.forward_cp(X, test_mode=test_mode)
        else:
            if X.ndim == 4:
                if not test_mode:
                    out, self.positive_locs = relu_cy.relu_4d_forward_train(X)
                else:
                    out = relu_cy.relu_4d_forward_test(X)
            elif X.ndim == 2:
                if not test_mode:
                    out, self.positive_locs = relu_cy.relu_2d_forward_train(X)
                else:
                    out = relu_cy.relu_2d_forward_test(X)

        return out

    def forward_np(self, X, test_mode=False):
        out = np.maximum(0, X)
        if not test_mode:
            self.positive_locs = (out > 0).astype(np.float32)
        return out

    def forward_cp(self, X, test_mode=False):
        xp = cp.get_array_module(X)
        out = xp.maximum(0, X)
        if not test_mode:
            self.positive_locs = (out > 0).astype(cp.float32)
        return out

    def backward(self, upstream_dx): 
        if self.is_on_gpu:
            return cp.multiply(cp.asarray(upstream_dx), cp.asarray(self.positive_locs))
        #else:
        #xp = cp.get_array_module(upstream_dx, self.positive_locs)

        return np.multiply(upstream_dx, self.positive_locs)

    def save_to_h5(self, open_f, save_grads=True):
        base_dset = open_f.create_dataset(self.layer_name + "/layer_info", dtype=np.float32)
        base_dset.attrs["type"] = self.__class__.__name__

    def load_from_h5(self, open_f, load_grads=True):
        pass