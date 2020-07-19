import numpy as np

class ReshapeLayer:
    def __init__(self, input_shape, output_shape):
        self.type = "reshape"
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __repr__(self):
        s = "ReshapeLayer(input_shape={}, output_shape={})".format(self.input_shape,
                                                                   self.output_shape)
        return s                                                                  

    def forward(self, X, test_mode=False):

        return X.reshape(self.output_shape)

    def backward(self, upstream_dx):

        return upstream_dx.reshape(self.input_shape)