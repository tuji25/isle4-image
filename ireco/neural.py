import numpy as np
from numpy.random import default_rng
from ireco import utils
from ireco.optimizer import sgd, adagrad

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
    x = np.where(x > 600, 600, x)
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
    dH : B * M matrix
    """
    dH = (np.ones_like(H) - H)*H
    return dH

def relu_differential(H):
    """
    Parameters
    ----------
    A : B * M matrix
        B sets of sigmoid outputs(each output is h1, ... , hM)

    Return
    ------
    B * M matrix
    """
    return H > 0

def fit(X, Y, num_hidden_units, batch_size, epochs=1, optimizer="sgd", activation="sigmoid", filename=None):
    """
    Trains the neural network

    Parameters
    ----------
    X : Input data
    Y : Target data
    num_hidden_units : Integer
    batch_size : Integer
    epochs : Integer
    optimizer : string or instance of optimizer class
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

    if optimizer == "sgd":
        opt = sgd.SGD()
    elif optimizer == "adagrad":
        opt = adagrad.AdaGrad()
    elif type(optimizer) == str:
        raise ValueError("This optimizer is not implemented!")
    else :
        opt = optimizer

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
        Y1 = np.array([activation_func(lsum(x, W1)) for x in XB])
        Y2 = np.array([softmax(lsum(x, W2)) for x in Y1])

        # cross entropy loss
        sum_loss += cross_entropy(YB, Y2)

        # error backpropagation
        dEb_da2 = Y2 - YB
        Y1_add_ones = np.insert(Y1, 0, 1, axis=1)
        dE_dW2 = np.dot((dEb_da2).T, Y1_add_ones) / batch_size
        dh_da1 = activation_differential(Y1)
        W2_delete_1column = np.delete(W2, 0, axis=1)
        dEb_da1 = dh_da1 * np.dot(dEb_da2,W2_delete_1column)
        XB_add_ones = np.insert(XB, 0, 1, axis=1)
        dE_dW1 = np.dot((dEb_da1).T, XB_add_ones) / batch_size

        # update weight
        W1, W2 = opt.update(W1, W2, dE_dW1, dE_dW2)
        
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