from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from nn.activation import Activation


class Sigmoid(Activation):
    def map_func(self, X):
        return 1 / (1 + np.exp(-X))

    def get_x_grad(self, X):
        fx = self.map_func(X)
        if len(X.shape) != 1:
            grad = np.zeros(X.shape)
        else:
            return np.diag(fx * (1 - fx))


if __name__ == "__main__":
    X = np.random.randn(10)
    model = Sigmoid(10)
    model.check_gradient(X)