from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.output_layer import OutputLayer


class MSE(OutputLayer):
    def map_func(self, X):
        return np.mean((X - self.answer) ** 2)

    def get_x_grad(self, X):
        return 2 * (X - self.answer) / X.size

if __name__ == "__main__":
    answer = np.random.randn(5,10)
    X = np.random.randn(5,10)
    model = MSE(10)
    model.set_answer(answer)
    model.check_gradient(X)
