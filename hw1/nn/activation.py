from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from nn.module import Module


class Activation(Module):  # Computes hyperbolic tangent of x element-wise
    def __init__(self, n):
        super(Activation, self).__init__(n,n)
        self.grad_input = None
        self.in_data = None
        self.output = None
        self.grad_next = None

    def forward(self, in_data):
        self.in_data = in_data
        self.output = self.map_func(in_data)
        return self.output

    def update_grad_input(self, grad_next):
        self.grad_next = grad_next
        self.grad_input = self.get_x_grad(self.in_data, self.grad_next)
        return self.grad_input

    def get_analytic_gradient(self, X, answer):
        Y = self.loss_function_prime(self.forward(X), answer)
        return {"X": self.get_x_grad(X, Y)}

    def get_params(self):
        return {}
