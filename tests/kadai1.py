import mnist
import numpy as np
from numpy.random import default_rng, seed
from ireco import neural

def main():
    # dimension of input vector
    D = 784
    # the number of hidden units
    M = 10
    # the number of classes
    C = 10

    raw_X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
    raw_Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
    s = input("Enter 0~9999: ")
    i = int(s)

    # convert mnist data to D-d vector
    x = np.reshape(raw_X[i], D)

    # generate random weight
    rng = default_rng(seed=1)
    W1 = rng.normal(0, np.sqrt(1/D), (M, D+1))
    W2 = rng.normal(0, np.sqrt(1/M), (C, M+1))

    # forward propagation
    Y = neural.predict(x, W1, W2)

    answer = np.argmax(Y)
    print("answer:", raw_Y[i])
    print("prediction:", answer)

if __name__ == '__main__':
    main()