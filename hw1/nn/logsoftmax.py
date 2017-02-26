from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.activation import Activation


class LogSoftMax(Activation):  # Computes hyperbolic tangent of x element-wise
    def softmax(self, X):
        Y = np.exp(X)
        return Y / Y.sum(axis = -1, keepdims=True)

    def map_func(self, X):
        return np.log(self.softmax(X))

    def get_x_grad(self, X, Y):
        fx = self.softmax(X)
        return (Y - np.einsum('...i,...j->...i', fx, Y))


if __name__ == "__main__":
    model = LogSoftMax(10)
    X = np.random.randn(3,10)
    answer = np.random.randn(3, 10)
    model.check_gradient(X, answer)
