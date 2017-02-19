from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.activation import Activation


class Tanh(Activation): # Computes hyperbolic tangent of x element-wise
    def map_func(self, X):
        return np.tanh(X)

    def get_x_grad(self, X):
        return np.diag(1 - (np.tanh(X) ** 2))


if __name__ == "__main__":
    X = np.random.randn(10)
    model = Tanh(10)
    model.check_gradient(X)