from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.activation import Activation


class SoftMax(Activation): # Computes hyperbolic tangent of x element-wise
    def map_func(self, X):
        Y = np.exp(X)
        if len(X.shape) > 1:
            return Y / Y.sum(axis=0, keepdims=True)
        return Y / Y.sum()

    def get_x_grad(self, X):
        fx = self.map_func(X)
        if len(X.shape) > 1:
            grad = np.zeros(X.shape + (X.shape[1],))
            for i in xrange(X.shape[0]):
                grad[i] = np.diag(fx[i]) - np.dot(np.expand_dims(fx[i], axis=-1),
                                    np.expand_dims(fx[i], axis = 0))
            return grad
        else:
            return np.diag(fx) - np.dot(np.expand_dims(fx, axis=-1),
                                    np.expand_dims(fx, axis = 0))

    def check_gradient(self, X, epsilon=1e-12, rtol=1e-4, atol=1e-4):
        super(SoftMax, self).check_gradient(X, epsilon = epsilon, rtol = rtol, atol = atol)


if __name__ == "__main__":
    X = np.random.randn(10)
    model = SoftMax(10)
    model.check_gradient(X)