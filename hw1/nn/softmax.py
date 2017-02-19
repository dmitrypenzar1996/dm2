from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.activation import Activation


class SoftMax(Activation): # Computes hyperbolic tangent of x element-wise
    def map_func(self, X):
        Y = np.exp(X)
        return Y / Y.sum()

    def get_x_grad(self, X):
        fx = self.map_func(X)
        return np.diag(fx) - np.dot(fx.reshape(*(fx.shape + (1,))),
                                    fx.reshape(1, *fx.shape))

    def check_gradient(self, X, epsilon=1e-12, rtol=1e-4, atol=1e-4):
        super(SoftMax, self).check_gradient(X, epsilon = epsilon, rtol = rtol, atol = atol)


if __name__ == "__main__":
    X = np.random.randn(10)
    model = SoftMax(10)
    model.check_gradient(X)