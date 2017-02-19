from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np

class Sequential(Module):
    def __init__(self):
        super(Sequential, self).__init__()
        self.layer = []

    def add(self, module):
        if not self.layer:
            self.layer.append(module)
            return
        if self.layer[-1].out_shape != module.in_shape:
            raise ValueError("Out shape of last added module is not equal to {}".format(module.in_shape))
        self.layer.append(module)

    def remove(self, module):

        ind = self.layer.index(module)

        if 0 < ind < len(self.layer) - 1 and\
                self.layer[ind - 1].out_shape != self.layer[ind + 1].in_shape:
            raise ValueError("After removing this layer there is no consistency in shapes of layers")

        self.layer.remove(module)

    def forward(self, in_data, answer):
        self.layer[-1].set_answer(answer)
        for layer in self.layer:
            in_data = layer.forward(in_data)
        self.output = in_data
        return self.output

    def update_grad_input(self, answer):
        init_input = answer
        for layer in self.layer[::-1]:
            init_input = layer.update_grad_input(init_input)

    def update_parameters(self, *args, **kwargs):
        for layer in self.layer:
            layer.update_parameters()
        # that's fine not to implement this method
        # module may have not parameters (for example - MSE criterion)
