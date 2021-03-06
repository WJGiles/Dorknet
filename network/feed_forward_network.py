import numpy as np
import cupy as cp
import h5py, json
from tqdm import tqdm
from layers.dense_layer import DenseLayer
from layers.convolution import ConvLayer
from layers.depthwise_convolution import DepthwiseConvLayer
from layers.pointwise_convolution import PointwiseConvLayer
from layers.residual_block import ResidualBlock
from layers.activations import ReLu
from layers.pooling import GlobalAveragePoolingLayer
from layers.batch_norm import BatchNormLayer
from layers.losses import SoftmaxWithCrossEntropy

class FeedForwardNetwork:
    def __init__(self, name):
        self.name = name
        self.is_on_gpu = False
        self.layers = []
        self.loss_layer = None

    def __repr__(self):
        out = "{}: \n".format(self.name)
        for l in self.layers:
            out += "\t" + l.__repr__() + "\n"

        return out

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss_layer(self, loss_layer):
        self.loss_layer = loss_layer

    def to_gpu(self):
        if self.is_on_gpu:
            print("Model already on GPU, ignoring request")
        else:
            try:
                for layer in self.layers:
                    layer.to_gpu()
                self.is_on_gpu = True
            except Exception as e:
                print("Error putting layer {} on GPU, error was: {}".format(layer, e))
                raise e # TODO: You probably want to fall back on CPU, should implement this...

    def forward(self, X, y_one_hot, test_mode=False, terminal_layer_name=None):
        loss = 0
        regularisation_terms = []
        for layer in self.layers:
            X = layer.forward(X, test_mode=test_mode)
            if layer.layer_name == terminal_layer_name:
                return loss, X
            if not test_mode:
                if hasattr(layer, "regulariser_forward"):
                    regularisation_terms.append(layer.regulariser_forward())
        if self.loss_layer is not None:
            this_loss, X = self.loss_layer.forward(X, y_one_hot, test_mode=test_mode)
            loss += this_loss
            loss += sum(regularisation_terms)
    
        return loss, X # NB if test_mode=True, you get softmax scores ("logits")

    def backward(self):
        if self.loss_layer is not None:
            upstream_dx = self.loss_layer.backward()
        else:
            raise(ValueError("Network doesn't have a loss, can't run backward pass."))
        for layer in self.layers[::-1]:
            upstream_dx = layer.backward(upstream_dx)

    def test(self, data_loader, batch_size, test_set_size):
        test_correct_total = 0
        for X_test_batch, y_test_batch, _ in tqdm(data_loader,
                                            total=test_set_size/batch_size):
            if self.is_on_gpu:
                X_test_batch = cp.asarray(X_test_batch)
            _, batch_scores = self.forward(X_test_batch,
                                           y_one_hot=None,
                                           test_mode=True)
            test_correct_total += np.sum(
                y_test_batch == np.argmax(cp.asnumpy(batch_scores),
                                          axis=1)                         
                )

        test_acc = float(test_correct_total) / test_set_size

        return test_acc

    def save_weights_to_h5(self, fname):
        with h5py.File(fname, "w")  as f:
            for layer in self.layers:
                layer.save_to_h5(f)
            if self.loss_layer is not None:
                self.loss_layer.save_to_h5(f)

    def save_layer_structure_to_json(self, fname):
        structure_dict = {"name": self.name}
        for layer in self.layers:
            structure_dict[layer.layer_name] = repr(layer)
        if self.loss_layer is not None:
            structure_dict[self.loss_layer.layer_name] = repr(self.loss_layer)
        with open(fname, "w")  as f:
            json.dump(structure_dict, f, indent=4)

    def load_network_from_json_and_h5(self, json_fname, h5_fname):
        with open(json_fname, "r") as f:
            json_structure = json.load(f)
        with h5py.File(h5_fname, "r")  as f:
            self.name = json_structure["name"]
            del json_structure['name']

            for layer_name in json_structure.keys():
                l_type = f[layer_name + "/layer_info"].attrs["type"]

                if l_type == "SoftmaxWithCrossEntropy":
                    l = SoftmaxWithCrossEntropy(layer_name)
                    l.load_from_h5(f)
                    self.loss_layer = l
                    continue
                elif l_type == "ConvLayer":
                    l = ConvLayer(layer_name)
                elif l_type == "BatchNormLayer":
                    l = BatchNormLayer(layer_name)
                elif l_type == "ReLu":
                    l = ReLu(layer_name)
                elif l_type == "DepthwiseConvLayer":
                    l = DepthwiseConvLayer(layer_name)
                elif l_type == "PointwiseConvLayer":
                    l = PointwiseConvLayer(layer_name)
                elif l_type == "GlobalAveragePoolingLayer":
                    l = GlobalAveragePoolingLayer(layer_name)
                elif l_type == "DenseLayer":
                    l = DenseLayer(layer_name)
                elif l_type == "ResidualBlock":
                    l = ResidualBlock(layer_name)

                l.load_from_h5(f)
                self.layers.append(l)
