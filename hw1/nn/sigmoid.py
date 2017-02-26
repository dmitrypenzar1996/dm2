from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from nn.activation import Activation


class Sigmoid(Activation):
    def map_func(self, X):
        return 1 / (1 + np.exp(-X))

    def get_x_grad(self, X, Y):
        fx = self.map_func(X)
        return  Y * (1 - fx) * fx

if __name__ == "__main__":
    X = np.random.randn(3, 10)
    model = Sigmoid(10)
    answer = np.random.randn(3, 10)
    model.check_gradient(X, answer)