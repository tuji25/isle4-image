import numpy as np
import mnist
from ireco import neural, utils
from ireco.optimizer import adagrad

def main():
    raw_X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
    raw_Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
    # convert mnist data to D-d vector
    X = np.reshape(raw_X, (raw_X.shape[0], 784))
    
    # conver labels to one-hot vector
    Y = utils.to_categorical(raw_Y, 10)

    M = 512
    batch_size = 100
    epochs = 25

    opt = adagrad.AdaGrad()

    neural.fit(X, Y, num_hidden_units=M, batch_size=batch_size, epochs=epochs, optimizer=opt, activation="relu", filename="weight_data/weight_relu_adagrad.npz")

if __name__ == '__main__':
    main()