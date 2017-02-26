from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.output_layer import OutputLayer
from nn.module import mse_func, mse_prime


class MSE(OutputLayer):
    def map_func(self, X, answer):
        return mse_func(X, answer)

    def get_x_grad(self, X, answer):
        return mse_prime(X, answer)


if __name__ == "__main__":
    answer = np.random.randn(10)
    X = np.random.randn(10)
    model = MSE(10)
    model.set_answer(answer)
    model.check_gradient(X, answer)
