import numpy as np

class SGD():
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.delta1 = None
        self.delta2 = None
    
    def update(self, W1, W2, grad_W1, grad_W2):
        # initialize delta1 and delta2
        if self.delta1 is None:
            self.delta1 = np.zeros_like(W1, dtype=float)
        
        if self.delta2 is None:
            self.delta2 = np.zeros_like(W2, dtype=float)

        # update delata and weight
        self.delta1 = self.momentum * self.delta1 - self.learning_rate * grad_W1
        self.delta2 = self.momentum * self.delta2 - self.learning_rate * grad_W2

        new_W1 = W1 + self.delta1
        new_W2 = W2 + self.delta2

        return new_W1, new_W2