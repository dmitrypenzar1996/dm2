from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.activation import Activation


class Relu(Activation): # Computes hyperbolic tangent of x element-wise
    def map_func(self, X):
        Y = X.copy()
        Y[Y < 0] = 0
        return Y

    def get_x_grad(self, X):
        Y = X.copy()
        Y[Y < 0] = 0
        Y[Y > 0] = 1
        return np.diag(Y)


if __name__ == "__main__":
    X = np.random.randn(10)
    model = Relu(10)
    model.check_gradient(X)