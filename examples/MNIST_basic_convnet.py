from tqdm import tqdm
from network.network import Network
from layers.convolution import ConvLayer
from layers.batch_norm import BatchNormLayer
from layers.activations import ReLu
from layers.pooling import GlobalAveragePoolingLayer
from layers.dense_layer import DenseLayer
from layers.losses import SoftmaxWithCrossEntropy
from regularisers.l2 import l2
from optimisers.SGDMomentum import SGDMomentum
from data_loading.MNIST_data_loading import get_MNIST_data, epoch_image_generator

BATCH_SIZE = 200

class MNISTNet(Network):

    def __init__(self, name, load_layers=True):
        super().__init__(name)
        if load_layers:
            # 0 Spatial (28, 28) --> (28, 28)
            self.add_layer(ConvLayer("conv_1",
                                     filter_block_shape=(32,1,3,3),
                                     with_bias=False,
                                     weight_regulariser=l2(0.0001)))
            self.add_layer(BatchNormLayer("bn_1",
                                          incoming_chans=32))
            self.add_layer(ReLu("relu_1"))
            # 1 Spatial (28, 28) --> (28, 28)
            self.add_layer(ConvLayer("conv_2",
                                     filter_block_shape=(32,32,3,3),
                                     with_bias=False,
                                     weight_regulariser=l2(0.0001)))
            self.add_layer(BatchNormLayer("bn_2",
                                          incoming_chans=32))
            self.add_layer(ReLu("relu_2"))
            # 2 Spatial (28, 28) --> (14, 14)
            self.add_layer(ConvLayer("conv_3",
                                     filter_block_shape=(64,32,4,4),
                                     with_bias=False,
                                     stride=2,
                                     weight_regulariser=l2(0.0001)))
            self.add_layer(BatchNormLayer("bn_3",
                                          incoming_chans=64))
            self.add_layer(ReLu("relu_3"))
            # 3 Spatial (14, 14) --> (14, 14)
            self.add_layer(ConvLayer("conv_4",
                                     filter_block_shape=(64,64,3,3),
                                     with_bias=False,
                                     weight_regulariser=l2(0.0001)))
            self.add_layer(BatchNormLayer("bn_4",
                                          incoming_chans=64))
            self.add_layer(ReLu("relu_4"))
            # 4 Spatial (14, 14) --> (7, 7)
            self.add_layer(ConvLayer("conv_5",
                                     filter_block_shape=(128,64,4,4),
                                     with_bias=False,
                                     stride=2,
                                     weight_regulariser=l2(0.0001)))
            self.add_layer(BatchNormLayer("bn_5",
                                          incoming_chans=128))
            self.add_layer(ReLu("relu_4"))
            # Spatial (7, 7) --> (1,)
            self.add_layer(GlobalAveragePoolingLayer("global_pool"))

            self.add_layer(DenseLayer("dense_1",
                                      incoming_chans=128,
                                      output_dim=10,
                                      weight_regulariser=l2(0.0005)))
            self.add_layer(SoftmaxWithCrossEntropy("softmax"))

X_train, y_train, X_val, y_val, X_test, y_test = get_MNIST_data(num_training=50000, 
                                                                num_validation=10000, 
                                                                num_test=10000)
network = MNISTNet("MNISTDemo")
sgd = SGDMomentum(network, 0.01, 0.9)
print(network)

for e in range(1, 15, 1):
    print("Epoch {}:".format(e))
    if e%5 == 0:
        sgd.multiply_learning_rate(0.1)
    for i, (X_batch, y_batch, y_one_hot) in enumerate(tqdm(epoch_image_generator(
                                                                X_train,
                                                                y_train,
                                                                BATCH_SIZE,
                                                                num_classes=10
                                                            ), 
                                                            total=50000/BATCH_SIZE)):
        loss, batch_scores = network.forward(X_batch, y_one_hot)
        #print(loss)
        network.backward()
        sgd.update_weights()
    print("Testing...")
    test_acc = network.test(epoch_image_generator(X_test,
                                                  y_test,
                                                  BATCH_SIZE,
                                                  num_classes=10),
                            BATCH_SIZE,
                            10000)
    print("Test acc: {}".format(test_acc))
