from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.activation import Activation


class SoftMax(Activation): # Computes hyperbolic tangent of x element-wise
    def map_func(self, X):
        Y = np.exp(X)
        return Y / Y.sum(axis = -1, keepdims=True)

    def get_x_grad(self, X, Y):
        fx = self.map_func(X)
        return fx * (Y - (Y * fx).sum(axis=-1, keepdims=True))




if __name__ == "__main__":
    model = SoftMax(10)
    X = np.random.randn(3, 10)
    answer = np.random.randn(3, 10)
    model.check_gradient(X, answer)