import numpy as np

arr = np.fromfile('train-images-idx3-ubyte', 'B')
train = arr[16:].reshape((60000,784))

arrLabs = np.fromfile('train-labels-idx1-ubyte', 'B')
trainLabels = arrLabs[8:]

testArr = np.fromfile('t10k-images-idx3-ubyte', 'B') 
testIms = testArr[16:].reshape((10000,784))

testLabs = np.fromfile('t10k-labels-idx1-ubyte', 'B') 
testLabs = testLabs[8:]

np.save('MNIST_data/MNISTTrainImages.npy', train)
np.save('MNIST_data/MNISTTrainLabels.npy', trainLabels)
np.save('MNIST_data/MNISTTestImages.npy', testIms)
np.save('MNIST_data/MNISTTestLabels.npy', testLabs)
