from fileinput import filename
import numpy as np
import mnist
from ireco import neural, utils

def main():
    raw_X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
    raw_Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")

    # convert mnist data to D-d vector
    X = np.reshape(raw_X, (raw_X.shape[0], 784))

    filename = input("filename:")
    activation = input("activation:")
    prediction = neural.fpredict(X, filename, activation=activation)

    y_pred = np.argmax(prediction, axis=1)

    print("accuracy:", utils.accuracy(raw_Y, y_pred))

if __name__ == "__main__":
    main()