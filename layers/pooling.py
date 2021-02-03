import numpy as np
import cupy as cp
from .convolution import ConvLayer
import im2col
import pooling_cy

profile = lambda x : x

class GlobalAveragePoolingLayer:
    """
    Takes the mean over spatial dimensions, reducing to one feature per channel per image
    """

    def __init__(self, layer_name):
        self.use_cp = True
        self.layer_name = layer_name

    def __repr__(self):
        return "GlobalAveragePoolingLayer({})".format(
            self.layer_name
        )

    def to_gpu(self):
        pass

    def forward(self, X, test_mode=False):
        self.spatial_shape = (X.shape[-2], X.shape[-1])
        xp = cp.get_array_module(X)
        out = xp.mean(X, axis=(2,3))
        return out

    def backward(self, upstream_dx):
        xp = cp.get_array_module(upstream_dx)
        thing = (1.0/float(np.prod(self.spatial_shape)))*upstream_dx[:, :, xp.newaxis, xp.newaxis]
        return thing*xp.ones((upstream_dx.shape[0], 
                              upstream_dx.shape[1],
                              self.spatial_shape[0],
                              self.spatial_shape[1]), 
                              dtype=xp.float32)

    def save_to_h5(self, open_f, save_grads=True):
        base_dset = open_f.create_dataset(self.layer_name + "/layer_info", dtype=np.float32)
        base_dset.attrs["type"] = self.__class__.__name__

    def load_from_h5(self, open_f, load_grads=True):
        pass

class MaxPoolLayer:

    def __init__(self, input_shape, stride=2):
        """
        Only does square pooling regions.
        """
        self.stride = stride
        self.max_locations = None

    def __repr__(self):

        return "MaxPoolLayer(stride={})".format(
                                                self.stride
                                                )

    @profile
    def forward(self, X, test_mode=False):
        """
        X.shape = (batch_size, channels, height, width)
        """
        if test_mode:
            out = pooling_cy.pool(X.astype(np.float32), self.stride)
        else:
            out, self.max_locations = pooling_cy.pool_train(X.astype(np.float32), self.stride)

        return out

    @profile
    def backward(self, upstream_dx):
        
        return pooling_cy.pool_backward(self.max_locations,
                                        upstream_dx,
                                        self.stride)
