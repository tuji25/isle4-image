import numpy as np

class AdaGrad():
    def __init__(self, learning_rate=0.001, epsilon=1.0e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.h1 = None
        self.h2 = None

    def update(self, W1, W2, grad_W1, grad_W2):
        # initialize h1 and h2
        if self.h1 is None:
            self.h1 = np.full_like(W1, self.epsilon, dtype=float)
        
        if self.h2 is None:
            self.h2 = np.full_like(W2, self.epsilon, dtype=float)

        # update h and weight
        self.h1 = self.h1 + grad_W1 * grad_W1
        self.h2 = self.h2 + grad_W2 * grad_W2

        new_W1 = W1 - self.learning_rate * (grad_W1 / np.sqrt(self.h1))
        new_W2 = W2 - self.learning_rate * (grad_W2 / np.sqrt(self.h2))

        return new_W1, new_W2