from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.module import Module

class Linear(Module):
    def __init__(self, n, m):
        super(Linear, self).__init__(n, m)
        self.weights = self.param_init(m, n)
        self.bias = self.param_init(m)
        self.grad_input = None
        self.grad_next = None
        self.in_data = None

    def map_func(self, X, weights, bias):
        return np.dot(weights, X.T).T + bias

    def forward(self, in_data):
        self.in_data = in_data
        self.output = self.map_func(self.in_data, self.weights, self.bias)
        return self.output

    def update_grad_input(self, grad_next):
        self.grad_next = grad_next
        self.grad_input = self.get_x_grad(self.grad_next)
        return self.grad_input

    def update_parameters(self, alpha=1e-5):
        self.weights -= alpha * self.get_weights_grad(self.in_data, self.grad_next)
        self.bias -= alpha * self.get_bias_grad(self.grad_next)

    def get_weights_grad(self, X, Y):
        if len(X.shape) > 1:
            return np.dot(Y.T, X) / X.shape[0]
        else:
            return np.expand_dims(Y, axis = -1) * X

    def get_bias_grad(self, Y):
        if len(Y.shape) > 1:
            Y = Y.mean(axis=0)
        return Y

    def get_x_grad(self, Y):
        return np.dot(Y, self.weights)

    def get_params(self):
        return {'weights': self.weights, 'bias': self.bias}

    def get_analytic_gradient(self, X, answer):
        Y = self.loss_function_prime(self.forward(X), answer)
        return {'weights': self.get_weights_grad(X, Y),
                'bias': self.get_bias_grad(Y),
                    'X': self.get_x_grad(Y)}


if __name__ == "__main__":
    X = np.random.randn(3, 10)
    answer = np.random.randn(3, 15)
    lin = Linear(10, 15)
    lin.check_gradient(X, answer)
