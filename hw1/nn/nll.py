from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.output_layer import OutputLayer


class NegativeLogLikelihood(OutputLayer):
    def map_func(self, X, answer):
        return - (answer * np.log(X)).sum()

    def get_x_grad(self, X, answer):
        return - answer / X


if __name__ == "__main__":
    X = np.abs(0.25 * np.random.randn(3, 10))
    X[X>=1] = 0.5
    X[X<=0] = 0.25
    Y = np.zeros((3, 10))
    Y[0][0] = 1
    Y[1][1] = 1
    Y[2][2] = 1
    model = NegativeLogLikelihood(10)
    model.set_answer(Y)
    model.check_gradient(X, Y)