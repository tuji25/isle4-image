import numpy as np
from numpy.random import default_rng
import time


def minibatch(X, Y, B, seed=None):
    """
    Make mini batch
    
    Parameters
    ----------
    X : ndarray (N * D) 
        All train data. N input vectors(each vector is a D-d vector)
    Y : ndarray (N * C) 
        labels of all train data. N one-hot vectors(each label is a C-d vector)
    B : integer
        Batch size
    seed : integer
        seed
    
    Returns
    ------
    XB : B * D matrix
    YB : B * C matrix
    """
    D = X.ndim
    C = Y.ndim
    if seed is None:
        ut = int(time.time())
        rng1 = default_rng(seed=ut)
        rng2 = default_rng(seed=ut)
    else:
        rng1 = default_rng(seed=seed)
        rng2 = default_rng(seed=seed)
    XB = rng1.choice(X, size=B, replace=False, shuffle=False)
    YB = rng2.choice(Y, size=B, replace=False, shuffle=False)
    return XB, YB

def to_categorical(y, num_classes):
    """
    Create one-hot vector

    Parameters
    ----------
    y : Array with class values
        each element is 0 ~ (num_classes - 1)
    num_classes : number of classes

    Return
    ------
    Y : n * num_classes matrix
    """
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

# calculate accuracy rate
def accuracy(y_true, y_pred):
    """
    Calculate accuracy rate

    Parameter
    ---------
    y_true : Numpy array(s) of correct label(s)
    y_pred : Numpy array(s) of predicted label(s)

    Return
    ------
    accuracy rate
    """
    data_size = y_true.size
    return np.count_nonzero(y_true == y_pred) / data_size