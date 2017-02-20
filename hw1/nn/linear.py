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
        if len(self.in_data.shape) != 1:
            self.grad_input = np.matmul(self.get_x_grad(self.in_data), self.grad_next).sum(axis=-1)
        else:
            self.grad_input = np.matmul(self.get_x_grad(self.in_data), self.grad_next)
        return self.grad_input

    def update_parameters(self, alpha=1e-5):
        if len(self.in_data.shape) != 1:
            self.bias += alpha * (np.matmul(self.get_bias_grad(), self.grad_next.T)).mean(axis=-1)
            self.weights += alpha * (np.matmul(self.get_weights_grad(self.in_data), self.grad_next.T).mean(axis=-1))
        else:
            self.bias += alpha * (np.matmul(self.get_bias_grad(), self.grad_next.T))
            self.weights += alpha * (np.matmul(self.get_weights_grad(self.in_data), self.grad_next.T))

    def get_weights_grad(self, X):
        weights_grad = np.zeros(shape =\
                self.out_shape + self.in_shape + self.out_shape)
        for i in range(self.out_shape[0]):
            weights_grad[i, :, i] = X.mean(axis = 0)
        return weights_grad

    def get_bias_grad(self):
        return np.diag(np.ones(self.bias.shape))

    def get_x_grad(self, X):
        if len(X.shape) == 1:
            return self.weights.T
        else:
            return np.repeat(np.expand_dims(self.weights.T, axis = 0), X.shape[0], axis = 0)


    def get_params(self):
        return {'weights': self.weights, 'bias': self.bias}

    def get_analytic_gradient(self, X):
        return {'weights': self.get_weights_grad(X),
                'bias': self.get_bias_grad(),
                    'X': self.get_x_grad(X)}


if __name__ == "__main__":
    X = np.random.randn(3, 10)
    a = Linear(10, 10)
    a.check_gradient(X)
