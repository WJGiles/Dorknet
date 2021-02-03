import numpy as np
import cupy as cp

class SoftmaxWithCrossEntropy:

    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.is_loss = True

    def __repr__(self):
        return "SoftmaxWithCrossEntropy({})".format(self.layer_name)

    def to_gpu(self):
        pass

    def forward(self, X, y_one_hot=None, test_mode=False):
        xp = cp.get_array_module(X, y_one_hot)
        e = xp.exp(X)
        X = (1.0/xp.sum(e, axis=1)).reshape(-1,1)*e
        if test_mode:
            return 0, X
        self.y_one_hot = xp.asarray(y_one_hot)
        self.downstream_x = X
        #loss = (1/float(X.shape[0]))*np.sum(-np.log(X[range(X.shape[0]), y]))
        loss = (1/float(X.shape[0]))*xp.sum(-xp.log(xp.einsum("bij,bjk->b",
                                                              X.reshape(X.shape[0], 1, X.shape[1]),
                                                              self.y_one_hot.reshape(X.shape[0],
                                                              self.y_one_hot.shape[1], 1))))
        return loss, X

    def backward(self):
        return (1/float(self.downstream_x.shape[0]))*(self.downstream_x - self.y_one_hot)

    def save_to_h5(self, open_f, save_grads=True):
        base_dset = open_f.create_dataset(self.layer_name + "/layer_info", dtype=np.float32)
        base_dset.attrs["type"] = self.__class__.__name__

    def load_from_h5(self, open_f, load_grads=True):
        pass
