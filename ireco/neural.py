from cmath import nan
import numpy as np
from numpy.random import default_rng
from ireco import utils

# linear sum
def lsum(x, W):
    """
    This function return W * t_(1, x)
    (t_p means transpose of p)

    Parameters
    ----------
    x : D-d vecotor
    W : M * D+1 matrix
    
    Returns
    -------
    M-d vector
    """
    x1 = np.insert(x, 0, 1)
    return np.dot(W, x1)
 
# sigmoid function
def sigmoid(x):
    size = x.shape[0]
    ones = np.ones(shape=size)
    return ones / (ones + np.exp(-x))

# ReLU function
def relu(x):
    return np.maximum(x, 0)

# softmax function
def softmax(x):
    max = np.amax(x)
    maxes = np.full_like(x, max)
    t = np.exp(x - maxes)
    return t / np.sum(t)

def cross_entropy(Y1, Y2):
    """
    Calculate cross entropy loss

    Parameters
    ----------
    Y1 : B * C matrix
         B one-hot vectors(each vector is a C-d vector)
    Y2 : B * C matrix
         B outputs(each output is a C-d vector)

    Return
    ------
    float64
    """
    # batch size
    B = Y1.shape[0]
    log = np.log(Y2)
    # rewrite -inf
    log = np.where(np.isneginf(log), -1.0e+300, log)
    losses = np.sum(- Y1 * log, axis=0)
    return np.sum(losses) / B

def sigmoid_differential(H):
    """
    Parameters
    ----------
    H : B * M matrix
        B sets of sigmoid outputs(each output is h1, ... , hM)

    Return
    ------
    dH : M * B matrix
    """
    dH = ((np.ones_like(H) - H)*H).T
    return dH

def relu_differential(H):
    """
    Parameters
    ----------
    A : B * M matrix
        B sets of sigmoid outputs(each output is h1, ... , hM)

    Return
    ------
    M * B matrix
    """
    return (H > 0).T

def differential_by_output_unit_activation(Y1, Y2, batchsize):
    dEn_da2 = ((Y2 - Y1) / np.full_like(Y1, batchsize)).T
    return dEn_da2

def differential_by_activation(previous_differential, W, activation_func_differential):
    """
    parameters
    ----------
    previous_differential : C * B matrix
    W : C * (M+1) matrix
    activation_func_differential : M * B matrix

    Return
    ------
    M * B matrix
    """
    W_delete_1column = np.delete(W, 0, axis=1)
    dEn_da = activation_func_differential * np.dot(W_delete_1column.T, previous_differential)
    return dEn_da

def fit(X, Y, num_hidden_units, batch_size, epochs=1, learning_rate=0.01, activation="sigmoid", filename=None):
    """
    Trains the neural network

    Parameters
    ----------
    X : Input data
    Y : Target data
    num_hidden_units : Integer
    batch_size : Integer
    epochs : Integer
    learning_rate : float
    activation : string
    filename : string
        Default value is 'None'. When 'filename' is 'None', weight is not saved to file.

    Return
    ------
    array of weight
    """
    # the number of data
    N = X.shape[0]
    # dimension of input vector
    D = X.shape[1]
    # the number of classes
    C = Y.shape[1]
    # initialize weight
    rng = default_rng()
    W1 = rng.normal(0, np.sqrt(1/D), (num_hidden_units, D+1))
    W2 = rng.normal(0, np.sqrt(1/num_hidden_units), (C, num_hidden_units+1))

    if activation == "sigmoid":
        activation_func = sigmoid
        activation_differential = sigmoid_differential
    elif activation == "relu":
        activation_func = relu
        activation_differential = relu_differential
    else :
        raise ValueError("This activation function is not implemented!")

    each_epoch  = N // batch_size
    sum_loss = 0
    for i in range(each_epoch * epochs):
        # make mini-batch
        XB, YB = utils.minibatch(X, Y, batch_size)

        # forward propagation
        Z1 = np.array([activation_func(lsum(x, W1)) for x in XB])
        Y2 = np.array([softmax(lsum(x, W2)) for x in Z1])

        # cross entropy loss
        sum_loss += cross_entropy(YB, Y2)

        # error backpropagation
        dEn_da2 = differential_by_output_unit_activation(YB, Y2, batch_size)
        Z1_add_ones = np.insert(Z1, 0, 1, axis=1)
        dE_dW2 = np.dot(dEn_da2, Z1_add_ones)
        dh_da1 = activation_differential(Z1)
        dEn_da1 = differential_by_activation(dEn_da2, W2, dh_da1)
        XB_add_ones = np.insert(XB, 0, 1, axis=1)
        dE_dW1 = np.dot(dEn_da1, XB_add_ones)

        # renew weight
        W1 = W1 - learning_rate * dE_dW1
        W2 = W2 - learning_rate * dE_dW2
        
        if i != 0 and i % each_epoch == each_epoch - 1:
            print("epoch", i // each_epoch, "cross entropy:",sum_loss/each_epoch)
            sum_loss = 0
    
    if filename is not None:
        np.savez(filename, W1, W2)

    return W1, W2

def predict(X, W1, W2, activation="sigmoid"):
    """
    Generates output predictions for the input samples

    Parameters
    ----------
    X : Input data
    W1 : weight of first layer
    W2 : weight of second layer
    activation : string. activation function

    Return
    ------
    Numpy array(s) of predictions
    """
    if activation == "sigmoid": 
        activation_func = sigmoid
    elif activation == "relu":
        activation_func = relu
    else:
        raise ValueError("This activation function is not implemented!")

    if X.ndim == 1 :
        Y2 = softmax(lsum(activation_func(lsum(X, W1)),W2))
    else :
        Y2 = np.array([softmax(lsum(activation_func(lsum(x, W1)), W2)) for x in X])
        
    return Y2

def fpredict(X, filename, activation="sigmoid"):
    """
    predict using weight saved in file

    Parameters
    ----------
    X : Input data
    filename : String
    activation : string. activation function

    Return
    ------
    Numpy array(s) of predictions
    """
    weights = np.load(filename)
    W1 = weights[weights.files[0]]
    W2 = weights[weights.files[1]]
    return predict(X, W1, W2, activation=activation)