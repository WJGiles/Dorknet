import numpy as np
import cupy as cp
from tqdm import tqdm
import os, json, logging, h5py
from network import Network
from layers.dense_layer import DenseLayer
from layers.convolution import ConvLayer
from layers.depthwise_convolution import DepthwiseConvLayer
from layers.pointwise_convolution import PointwiseConvLayer
from layers.basic_residual_block import BasicResidualBlock
from layers.residual_block import ResidualBlock
from layers.activations import ReLu
from layers.pooling import GlobalAveragePoolingLayer
from layers.batch_norm import BatchNormLayer
from layers.losses import SoftmaxWithCrossEntropy
from regularisers.l2 import l2
from data_loading.image_data_loader import ImageDataLoader
from data_loading.image_augmentation import ImageAugmenter
from optimisers.RMSProp import RMSProp
from optimisers.SGDMomentum import SGDMomentum

#import line_profiler

#profile = lambda x: x

BATCH_SIZE = 60
DOCKER = False
if DOCKER:
    data_folder = "/Datasets"
else:
    data_folder = "/home/will/Datasets"

class ResNet18(Network):

    def depthwise_sep_layer(self, layer_name, incoming_chans, filter_block_shape,
                            stride=1, padding=1,
                            with_bias=False, batch_norm_depthwise=True, relu_depthwise=False,
                            batch_norm_pointwise=True, depthwise_weight_regulariser=None,
                            pointwise_weight_regulariser=None, final_relu=True, add_layers=False):
        """
        filter_block_shape: (outgoing_chans, incoming_chans, f_rows, f_cols)
        """
        depthwise_filter_shape = (incoming_chans, filter_block_shape[-2], filter_block_shape[-1])
        pointwise_filter_shape = (filter_block_shape[0], incoming_chans)
        layer_list = []
        layer_list.append(DepthwiseConvLayer(layer_name + "_dw",
                                          filter_block_shape=depthwise_filter_shape,
                                          stride=stride, padding=padding, with_bias=with_bias,
                                          weight_regulariser=depthwise_weight_regulariser))
        if batch_norm_depthwise:
            layer_list.append(BatchNormLayer(layer_name + "_dw_bn",
                                          input_dimension=4,
                                          incoming_chans=incoming_chans
            ))
        if relu_depthwise:
            layer_list.append(ReLu(layer_name + "dw_relu"))
        layer_list.append(PointwiseConvLayer(layer_name + "_pw",
                                          filter_block_shape=pointwise_filter_shape, with_bias=with_bias,
                                          weight_regulariser=pointwise_weight_regulariser))
        if batch_norm_pointwise:
            layer_list.append(BatchNormLayer(layer_name + "_pw_bn",
                                          input_dimension=4,
                                          incoming_chans=filter_block_shape[0]
            ))
        if final_relu:
            layer_list.append(ReLu(layer_name + "pw_relu"))
        if add_layers:
            for layer in layer_list:
                self.add_layer(layer)
        else:
            return layer_list

    def add_res_block(self, layer_name, first_filter_block_shape, 
                      downsample=False, weight_regulariser_strength=0.0001, depthwise_sep=False):
        num_filters, incoming_chans, f_rows, f_cols = first_filter_block_shape
        layer_list = []
        if depthwise_sep:
            layer_list += self.depthwise_sep_layer(layer_name + "_dw1", incoming_chans,
                                                   first_filter_block_shape,
                                                   stride=2 if downsample else 1, padding=1,
                                                   depthwise_weight_regulariser=None,
                                                   pointwise_weight_regulariser=l2(strength=weight_regulariser_strength),
                                                   final_relu=True, add_layers=False)
        else:
            layer_list.append(ConvLayer(layer_name + "_conv1", filter_block_shape=first_filter_block_shape,
                            stride=2 if downsample else 1, padding=1, with_bias=False,
                            weight_regulariser=l2(strength=weight_regulariser_strength)))
            layer_list.append(BatchNormLayer(layer_name + "_bn1", input_dimension=4, incoming_chans=num_filters))
            layer_list.append(ReLu(layer_name + "_relu1"))
        if depthwise_sep:
            layer_list += self.depthwise_sep_layer(layer_name + "_dw2", num_filters,
                                                   (num_filters,num_filters,f_rows,f_cols),
                                                   stride=1, padding=1,
                                                   depthwise_weight_regulariser=None,
                                                   pointwise_weight_regulariser=l2(strength=weight_regulariser_strength),
                                                   final_relu=False, add_layers=False)
        else:
            layer_list.append(ConvLayer(layer_name + "_conv2", filter_block_shape=(num_filters,num_filters,f_rows,f_cols),
                              stride=1, padding=1, with_bias=False, weight_regulariser=l2(strength=weight_regulariser_strength)))
            layer_list.append(BatchNormLayer(layer_name + "_bn2", input_dimension=4, incoming_chans=num_filters))
        if downsample:
            skip_proj = PointwiseConvLayer(layer_name + "_pw_skip", filter_block_shape=(num_filters,incoming_chans),
                                           stride=2, with_bias=False, weight_regulariser=l2(strength=weight_regulariser_strength))
        else:
            skip_proj = None
        relu2 = ReLu(layer_name + "_relu2")
        self.add_layer(ResidualBlock(layer_name, layer_list=layer_list, 
                                     skip_projection=skip_proj, post_skip_activation=relu2))

    def __init__(self, name, load_layers=True):
        super().__init__(name)
        if load_layers:
            # 0 Spatial (225, 225) --> (112, 112)
            self.add_layer(ConvLayer("conv0",
                                     filter_block_shape=(64,3,5,5),
                                     with_bias=False, stride=2, padding=1,
                                     weight_regulariser=l2(0.0001)))
            self.add_layer(BatchNormLayer("conv0_bn", input_dimension=4, incoming_chans=64))
            self.add_layer(ReLu("conv0_relu"))

            # 0 Spatial (112, 112) --> (56, 56)
            self.add_layer(PointwiseConvLayer("pw0",
                                     filter_block_shape=(64,64),
                                     with_bias=False, stride=2,
                                     weight_regulariser=l2(0.0001)))
            self.add_layer(BatchNormLayer("pw0_bn", input_dimension=4, incoming_chans=64))
            self.add_layer(ReLu("pw0_relu"))

            # 1 Spatial (56,56) --> (56,56)
            self.add_res_block("res1", (64, 64, 3, 3), depthwise_sep=True)

            # 2 Spatial (56,56) --> (56,56)
            self.add_res_block("res2", (64, 64, 3, 3), depthwise_sep=True)

            # 3 Spatial (56,56) --> (28,28)
            self.add_res_block("res3", (128,64,3,3), downsample=True, depthwise_sep=True)

            # 4
            self.add_res_block("res4", (128, 128, 3, 3), depthwise_sep=True)

            # 6 Spatial (28,28) --> (14,14)
            self.add_res_block("res5", (256,128,3,3), downsample=True, depthwise_sep=True)

            # 7
            self.add_res_block("res6", (256,256,3,3), depthwise_sep=True)

            # 9 Spatial (14,14) --> (7,7)
            self.add_res_block("res7", (512,256,3,3), downsample=True, depthwise_sep=True)

            # 10
            self.add_res_block("res8", (512,512,3,3), depthwise_sep=True)

            # 11 Spatial (7,7) --> (1,)
            self.add_layer(GlobalAveragePoolingLayer("global_pool1"))

            # 12
            self.add_layer(DenseLayer("dense1",
                                    incoming_chans=512,
                                    output_dim=120,
                                    weight_regulariser=l2(0.0001)))
            self.add_layer(SoftmaxWithCrossEntropy("softmax1"))

if __name__ == "__main__":
    augmenter = ImageAugmenter(hsv_pert_tuples=[(0.9,1.1), (0.5,2.0), (0.5,2.0)],
                               rotation_tuple=(-15,15),
                               horizontal_flip_prob=0.5)
    train_data_loader = ImageDataLoader(os.path.join(data_folder, "ImageNet2012/ILSVRC2012_dogs/train_img"),
                                        BATCH_SIZE,
                                        image_size=(225,225),
                                        class_balance=False,
                                        image_augmenter=augmenter,
                                        mixup_range_tuple=(0, 0.3),
                                        crop_mode="random")
    val_data_loader = ImageDataLoader(os.path.join(data_folder, "ImageNet2012/ILSVRC2012_dogs/val_img"),
                                      BATCH_SIZE,
                                      image_size=(225,225),
                                      crop_mode="center")

    experiment_name = "DogsImageNet225ResNet18DepSepREFACTORTEST"
    logging.basicConfig(filename="logging/" + experiment_name + '.log',level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    network = ResNet18(experiment_name, load_layers=True)
    if not os.path.isdir(experiment_name):
        os.mkdir(experiment_name)
    network.save_layer_structure_to_json(os.path.join(experiment_name, experiment_name + ".json"))
    #network = ResNet18("", load_layers=False)
    #network.load_network_from_json_and_h5(os.path.join(original_experiment_name, original_experiment_name + ".json"),
    #                                      os.path.join(old_experiment_name, "epoch_15_testacc_0.4935.h5"))
    print(network)
    sgd = SGDMomentum(network, 0.05*(BATCH_SIZE/200.0), 0.9)
    logging.info(network)

    try:
        for e in range(1, 40, 1):
            running_loss_average = None
            logging.info("Epoch {}:".format(e))
            logging.info("Shuffling data: ")
            train_data_loader.shuffle_indices()
            correct_total = 0
            if e == 16 or e == 20 or e == 25:
                logging.info("Multiplying learning rate by 0.5")
                sgd.multiply_learning_rate(0.5)
            for i, (X_batch, y_batch, y_one_hot) in enumerate(tqdm(train_data_loader.pull_batch(int(150473/BATCH_SIZE)), 
                                                        total=int(150473/BATCH_SIZE))):
                loss, batch_scores = network.forward(X_batch, y_one_hot)
                if running_loss_average is None:
                    running_loss_average = loss
                else:
                    running_loss_average = 0.9*running_loss_average + 0.1*loss
                correct_total += np.sum(y_batch == np.argmax(cp.asnumpy(batch_scores), axis=1))
                network.backward()
                sgd.update_weights()
                if (i%10 == 0): logging.info("Running loss average: {}".format(running_loss_average))
                if (i%100 == 0) and (i > 0):
                    logging.info("Running Ave Loss: {}, Loss: {}, Accuracy over current epoch so far: {} ".format(
                                                                running_loss_average,
                                                                loss,
                                                                correct_total/(i*BATCH_SIZE)))
            logging.info("Testing...")
            test_acc = network.test(network, val_data_loader.pull_batch(int(120*50/BATCH_SIZE)), BATCH_SIZE, 120*50)
            logging.info("Test acc: {}".format(test_acc))
            network.save_weights_to_h5(os.path.join(experiment_name, "epoch_{}_testacc_{}.h5".format(e, test_acc)))
    except KeyboardInterrupt:
        train_data_loader.stop_thread()
        val_data_loader.stop_thread()
        raise
