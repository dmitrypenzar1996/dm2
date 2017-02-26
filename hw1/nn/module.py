from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
from itertools import product


def mse_func(X, answer):
    return np.mean((X - answer) ** 2)


def mse_prime(X, answer):
    return 2 * (X - answer) / X.size


class Module(object):
    init_sigma = 0.000001
    init_mu = 0

    def __init__(self, n=0, m=0, batch_size = 1):
        self.output = None
        self.grad_input = None
        self.in_shape = (n, ) if type(n) is int else n
        self.out_shape = (m, ) if type(m) is int else m

    def map_func(self, *args, **kwargs):
        raise NotImplementedError('implement map func of layer!')

    @staticmethod
    def param_init(*dim):
        return Module.init_sigma * np.random.randn(*dim) + Module.init_mu


    def get_analytic_gradient(self, *args, **kwargs):
        raise NotImplementedError('implement get_analytic_gradient!')

    def get_params(self):
        raise NotImplementedError('get_params')


    @staticmethod
    def loss_function(Y, answer):
        return mse_func(Y, answer)

    @staticmethod
    def loss_function_prime(Y, answer):
        return mse_prime(Y, answer)

    def check_gradient(self, X, answer, epsilon = 1e-10, rtol = 1e-4, atol = None):
        atol = atol or rtol * 100

        sys.stdout.write("Start checking\n")

        num_grad = self.get_numeric_gradient(X, answer, epsilon)

        an_grad = self.get_analytic_gradient(X, answer)

        for key in num_grad.iterkeys():
            sys.stdout.write("Checking correctness of gradient for {} ... ".format(key))
            if np.allclose(num_grad[key], an_grad[key], rtol, atol):
                sys.stdout.write("Correct\n")
            else:
                print ("Numerical: ")
                print (num_grad[key])
                print ("Analytical: ")
                print(an_grad[key])
                sys.stdout.write("Fail\n")
        sys.stdout.write("Done\n")

    def get_numeric_gradient(self, X, answer, epsilon):
        params = self.get_params()
        params["X"] = X
        all_num_grad = {}

        for p_name, p_value in params.iteritems():
            num_grad = np.zeros(p_value.shape)
            for i in product(* map(range, p_value.shape)):
                p_value[i] += epsilon

                loss1 = self.loss_function(self.forward(X), answer)

                p_value[i] -= 2 * epsilon
                loss2 = self.loss_function(self.forward(X), answer)

                num_grad[i] = (loss1 - loss2) / (epsilon * 2)
                p_value += epsilon
            if len(X.shape) > 1 and p_name != "X":
                num_grad = num_grad / X.shape[0]
            all_num_grad[p_name] = num_grad

        return all_num_grad

    def forward(self, *args, **kwargs):
        raise NotImplementedError('implement forward pass!')

    def backward(self, *args, **kwargs):
        self.update_grad_input(*args, **kwargs)
        self.update_parameters(*args, **kwargs)

    def update_grad_input(self, *args, **kwargs):
        raise NotImplementedError('implement computation of gradient w.r.t. input! df(x)/dx!')

    def update_parameters(self, *args, **kwargs):
        # that's fine not to implement this method
        # module may have not parameters (for example - MSE criterion)
        pass
