import numpy as np
import cupy as cp
from layers.convolution import ConvLayer
from layers.depthwise_convolution import DepthwiseConvLayer
from layers.pointwise_convolution import PointwiseConvLayer
from layers.activations import ReLu
from layers.batch_norm import BatchNormLayer

from pprint import pprint

class ResidualBlock:
    """
    A block with a skip connection around the provided layer_list. The output of layer_list[-1] must have the same 
    shape as skip_projection(X) because they are joined by addition - skip_projection=None means an identity projection.
    The nonlinear activation (if not None) is applied after the join.

                                X 
                                |  \
                                |   \
                                |    \
                                |     \
                         layer_list[0] |
                         layer_list[1] |
                              ...      |
                              ...      |
                        layer_list[-1] |
                                      /
                                     /
                                    /
                                   /
                                |      
                        + skip_projection(X)
                        post_skip_activation
                                |
                               Out
    """
    def __init__(self, layer_name, layer_list=None, skip_projection=None, post_skip_activation=None):
        self.layer_name = layer_name
        self.layer_list = layer_list
        self.skip_projection = skip_projection
        self.post_skip_activation = post_skip_activation

        if layer_list is None:
            self.layer_list = []

    def __repr__(self):
        return "ResidualBlock({}, layer_list={}, skip_projection={}, post_skip_activation={})".format(
            self.layer_name, self.layer_list, self.skip_projection, self.post_skip_activation
        )
        
    def forward(self, X, test_mode=False):
        X_tmp = self.layer_list[0].forward(X, test_mode=test_mode)
        for layer in self.layer_list[1:]:
            X_tmp = layer.forward(X_tmp, test_mode=test_mode)
        
        if self.skip_projection is not None:
            skippee = self.skip_projection.forward(X, test_mode=test_mode)
        else:
            skippee = X

        joined = self.post_skip_activation.forward(X_tmp + skippee)
        return joined        

    def regulariser_forward(self):
        regularisation = 0
        for l in self.layer_list:
            if hasattr(l, "regulariser_forward"):
                        regularisation += l.regulariser_forward()

        return regularisation

    def backward(self, upstream_dx):
        joined_dx = self.post_skip_activation.backward(upstream_dx)
        dx = self.layer_list[-1].backward(joined_dx)
        for l in self.layer_list[-2::-1]:
            dx = l.backward(dx)

        if self.skip_projection is not None:
            dx_out = cp.asarray(dx) + self.skip_projection.backward(joined_dx)
        else:
            dx_out = cp.asarray(dx) + joined_dx

        return dx_out

    def save_to_h5(self, open_f, save_grads=True):
        base_dset = open_f.create_dataset(self.layer_name + "/layer_info", dtype=np.float32)
        base_dset.attrs["type"] = self.__class__.__name__
        base_dset.attrs["layer_type_list"] = [l.__class__.__name__ for l in self.layer_list]
        base_dset.attrs["layer_name_list"] = [l.layer_name for l in self.layer_list]
        base_dset.attrs["post_skip_activation_type"] = self.post_skip_activation.__class__.__name__
        base_dset.attrs["post_skip_activation_name"] = self.post_skip_activation.layer_name
        if self.skip_projection is not None:
            base_dset.attrs["skip_projection_type"] = self.skip_projection.__class__.__name__
            base_dset.attrs["skip_projection_name"] = self.skip_projection.layer_name

        for l in self.layer_list:
            l.save_to_h5(open_f, save_grads=save_grads)
        if self.skip_projection is not None:
            self.skip_projection.save_to_h5(open_f, save_grads=save_grads)
        self.post_skip_activation.save_to_h5(open_f, save_grads=save_grads)
        
    def load_from_h5(self, open_f, load_grads=True):
        layer_type_list = open_f[self.layer_name + '/layer_info'].attrs['layer_type_list']
        layer_name_list = open_f[self.layer_name + '/layer_info'].attrs['layer_name_list']
        for l_type, layer_name in zip(layer_type_list, layer_name_list):
                if l_type == "ConvLayer":
                    l = ConvLayer(layer_name)
                elif l_type == "BatchNormLayer":
                    l = BatchNormLayer(layer_name)
                elif l_type == "ReLu":
                    l = ReLu(layer_name)
                elif l_type == "DepthwiseConvLayer":
                    l = DepthwiseConvLayer(layer_name)
                elif l_type == "PointwiseConvLayer":
                    l = PointwiseConvLayer(layer_name)
                elif l_type == "ResidualBlock":
                    l = ResidualBlock(layer_name)
                self.layer_list.append(l)

        for l in self.layer_list:
            l.load_from_h5(open_f, load_grads=load_grads)
        if open_f[self.layer_name + '/layer_info'].attrs.get("skip_projection_type", None):
            skip_projection_type = open_f[self.layer_name + '/layer_info'].attrs["skip_projection_type"]
            skip_projection_name = open_f[self.layer_name + '/layer_info'].attrs["skip_projection_name"]
            if skip_projection_type == "PointwiseConvLayer":
                self.skip_projection = PointwiseConvLayer(skip_projection_name)
                self.skip_projection.load_from_h5(open_f, load_grads=load_grads)
            else:
                print("ResidualBlock: Unrecognised skip_projection type {}".format(skip_projection_type))

        post_skip_activation_type = open_f[self.layer_name + '/layer_info'].attrs["post_skip_activation_type"]
        post_skip_activation_name = open_f[self.layer_name + '/layer_info'].attrs["post_skip_activation_name"]
        if post_skip_activation_type == "ReLu":
            self.post_skip_activation = ReLu(post_skip_activation_name)
            self.post_skip_activation.load_from_h5(open_f, load_grads=load_grads)
        else:
            print("ResidualBlock: Unrecognised post_skip_activation type {}".format(post_skip_activation_type))
