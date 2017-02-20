from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.activation import Activation


class LogSoftMax(Activation):  # Computes hyperbolic tangent of x element-wise
    def map_func(self, X):
        if len(X.shape) > 1:
            add = np.log(np.exp(X).sum(axis=1, keepdims=True))
        else:
            add = np.log(np.exp(X).sum())
        return X - add

    def get_x_grad(self, X):

        fx = np.exp(X)

        if len(X.shape) > 1:
            fx = fx / fx.sum(axis=1, keepdims=True)
            grad = np.zeros(X.shape + (X.shape[1],))
            for i in xrange(X.shape[0]):
                grad[i] = np.diag(np.ones(X.shape[1])) - np.expand_dims(fx[i], axis=-1)
            return grad
        else:
            fx = fx / fx.sum()
            return np.diag(np.ones(X.shape)) - np.expand_dims(fx, axis=len(fx.shape))

    def check_gradient(self, X, epsilon=1e-12, rtol=1e-4, atol=1e-3):
        super(LogSoftMax, self).check_gradient(X, epsilon = epsilon, rtol = rtol, atol = atol)

if __name__ == "__main__":
    X = np.random.randn(3, 10)
    model = LogSoftMax(10)
    model.check_gradient(X)