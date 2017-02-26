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

    def get_x_grad(self, X, Y):
        Z = Y.copy()
        Z[X < 0] = 0
        return Z

if __name__ == "__main__":
    X = np.random.randn(3, 10)# to avoid bad part of function
    model = Relu(10)
    answer = np.random.randn(3, 10)
    model.check_gradient(X, answer)