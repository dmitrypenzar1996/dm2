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
        if len(X.shape) != 1:
            grad = np.zeros(X.shape + (X.shape[1],))
            for i in xrange(X.shape[0]):
                grad[i] = np.diag(Y[i])
            return grad
        else:
            return np.diag(Y)


if __name__ == "__main__":
    X = np.random.randn(100, 10)
    model = Relu(10)
    model.check_gradient(X)