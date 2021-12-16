import mnist
import numpy as np
from ireco import neural

def main():
    raw_X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
    raw_Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")

    s = input("Enter 0~9999: ")
    i = int(s)

    # convert mnist data to D-d vector
    x = np.reshape(raw_X[i], 784)

    prediction = neural.fpredict(x, filename="weight_data/weight.npz")

    print("answer:", raw_Y[i])
    print("prediction:", np.argmax(prediction))

if __name__ == "__main__":
    main()