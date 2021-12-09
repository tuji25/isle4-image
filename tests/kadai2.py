import numpy as np
import mnist
from numpy.random import default_rng
from ireco import neural, utils

def main():
    # dimension of input vector
    D = 784
    # the number of hidden units
    M = 10
    # the number of classes
    C = 10
    # batch size
    B = 100

    raw_X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
    raw_Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
    # convert mnist data to D-d vector
    X = np.reshape(raw_X, (raw_X.shape[0], 784))
    
    # conver labels to one-hot vector
    Y = utils.to_categorical(raw_Y, 10)

    # make mini-batch
    XB, YB = utils.minibatch(X, Y, B, seed=0)

    # generate random weight
    rng = default_rng(seed=1)
    W1 = rng.normal(0, np.sqrt(1/D), (M, D+1))
    W2 = rng.normal(0, np.sqrt(1/M), (C, M+1))

    # forward propagation
    Y2 = neural.forward_propagation(XB, W1, W2)

    loss = neural.cross_entropy(YB, Y2)
    print("cross entropy loss :", loss)


if __name__ == "__main__":
    main()