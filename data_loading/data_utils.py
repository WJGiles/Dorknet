import numpy as np
import os

"""
Old code for loading MNIST
"""

def epoch_generator(X_full, y_full, batch_size=100):
    shuffle_indices = np.random.permutation(X_full.shape[0])
    X_full = X_full[shuffle_indices, :]
    y_full = y_full[shuffle_indices]
    index = 0
    for i in range(int(X_full.shape[0]/batch_size)):
        X_batch = X_full[index:index+batch_size, :]
        y_batch = y_full[index:index+batch_size]
        index += batch_size

        yield X_batch, y_batch

def epoch_image_generator(X_full, y_full, batch_size=1, num_classes=10):
    shuffle_indices = np.random.permutation(X_full.shape[0])
    X_full = X_full[shuffle_indices, :]
    y_full = y_full[shuffle_indices]
    index = 0
    for i in range(int(X_full.shape[0]/batch_size)):
        X_batch = X_full[index:index+batch_size, :].reshape((batch_size, 1, 28, 28))
        y_batch = y_full[index:index+batch_size]
        one_hot_y = np.array([np.eye(num_classes, dtype=np.float32)[i, :] for i in y_batch])
        index += batch_size

        yield X_batch.astype(np.float32), y_batch, one_hot_y

def get_MNIST_data(num_training=59000, num_validation=1000, num_test=10000):

    mnist_dir = 'data/MNIST' 
    X_train = np.load(os.path.join(mnist_dir, 'MNISTTrainImages.npy')).astype(np.float32) 
    y_train = np.load(os.path.join(mnist_dir, 'MNISTTrainLabels.npy')).astype(np.int32) 
    X_test = np.load(os.path.join(mnist_dir, 'MNISTTestImages.npy')).astype(np.float32) 
    y_test = np.load(os.path.join(mnist_dir, 'MNISTTestLabels.npy')).astype(np.int32) 

    # Train/Val split
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0

    return X_train, y_train, X_val, y_val, X_test, y_test
