from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.output_layer import OutputLayer


class CrossEntropy(OutputLayer):
    def softmax(self, X):
        X = np.exp(X)
        if len(X.shape) > 1:
            return X / X.sum(axis=1, keepdims=True)
        else:
            return X / X.sum()

    def map_func(self, X):
        return -(self.answer * np.log(self.softmax(X))).sum()

    def get_x_grad(self, X):
        return self.softmax(X) - self.answer

    def check_gradient(self, X, epsilon=1e-4, rtol=1e-4, atol=1e-1):
        super(CrossEntropy, self).check_gradient(X, epsilon = epsilon, rtol = rtol, atol = atol)

if __name__ == "__main__":
    X = np.abs(0.25 * np.random.randn(3, 10))
    X[X>=1] = 0.5
    X[X<=0] = 0.25
    Y = np.zeros((3, 10))
    Y[0][0] = 1
    Y[1][1] = 1
    Y[2][2] = 1
    model = CrossEntropy(10)
    model.set_answer(Y)
    model.check_gradient(X)