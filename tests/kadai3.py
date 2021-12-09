import numpy as np
import mnist
from ireco import neural, utils

def main():
    raw_X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
    raw_Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
    # convert mnist data to D-d vector
    X = np.reshape(raw_X, (raw_X.shape[0], 784))
    
    # conver labels to one-hot vector
    Y = utils.to_categorical(raw_Y, 10)

    neural.learning(X, Y, 10, 1000, 10, 0.01)

if __name__ == '__main__':
    main()