import numpy as np
import cupy as cp
import relu_cy

class ReLu:

    def __init__(self, layer_name, use_np=False):
        self.use_cp = True
        self.layer_name = layer_name
        self.use_np = use_np

    def __repr__(self):
        return "ReLu({})".format(self.layer_name)

    def forward(self, X, test_mode=False):
        if self.use_cp:
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
        if self.use_cp:
            return cp.multiply(cp.asarray(upstream_dx), cp.asarray(self.positive_locs))
        #else:
        #xp = cp.get_array_module(upstream_dx, self.positive_locs)

        return cp.asarray(np.multiply(cp.asnumpy(upstream_dx), cp.asnumpy(self.positive_locs)))

    def save_to_h5(self, open_f, save_grads=True):
        base_dset = open_f.create_dataset(self.layer_name + "/layer_info", dtype=np.float32)
        base_dset.attrs["type"] = self.__class__.__name__

    def load_from_h5(self, open_f, load_grads=True):
        pass