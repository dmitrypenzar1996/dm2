from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from nn.module import Module

class OutputLayer(Module):
    def __init__(self, m):
        super(OutputLayer, self).__init__(m, 1)
        self.grad_input = None
        self.in_data = None
        self.output = None
        self.answer = None

    def forward(self, in_data):
        self.in_data = in_data
        self.output = self.map_func(in_data, self.answer)
        return self.output

    def update_grad_input(self, answer):
        self.answer = answer
        self.grad_input = self.get_x_grad(self.in_data, self.answer)  # we want to minimize
        return self.grad_input

    def get_analytic_gradient(self, X, answer):
        return {"X" : self.get_x_grad(X, answer)}

    @staticmethod
    def loss_function(Y, answer):
        return Y

    def get_params(self):
        return {}

    def set_answer(self, answer):
        self.answer = answer