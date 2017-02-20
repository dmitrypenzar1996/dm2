from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
from itertools import product


class Module(object):
    init_sigma = 0.001
    init_mu = 0.01

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

    def get_numeric_gradient(self, epsilon, **kwargs):
        all_num_grad = {}

        is_batch = len(kwargs['X'].shape) > 1
        for p_name, p_value in kwargs.iteritems():
            if self.out_shape != (1,):
                num_grad = np.zeros(shape = p_value.shape + self.out_shape)
            else:
                num_grad = np.zeros(p_value.shape)

            for i in product(* map(range, p_value.shape)):
                p_value[i] += epsilon
                cur_value = self.map_func(**kwargs)
                p_value[i] -= 2 * epsilon
                cur_value -= self.map_func(**kwargs)
                if is_batch:
                    if p_name != 'X':
                        num_grad[i] = cur_value.mean(axis=0) / (epsilon * 2)
                    else:
                        if self.out_shape != (1,):
                            num_grad[i] = cur_value[i[0]] / (epsilon * 2)
                        else:
                            num_grad[i] = cur_value / (epsilon * 2)
                else:
                    num_grad[i] = cur_value / (epsilon * 2)
                p_value += epsilon
            all_num_grad[p_name] = num_grad
        return all_num_grad

    def get_analytic_gradient(self, X):
        raise NotImplementedError('implement get_analytic_gradient!')

    def get_params(self):
        raise NotImplementedError('get_params')

    def check_gradient(self, X, epsilon = 1e-10, rtol = 1e-4, atol = None):
        atol = atol or rtol * 100

        sys.stdout.write("Start checking\n")

        params = self.get_params()
        params['X'] = X
        num_grad = self.get_numeric_gradient(epsilon, **params)

        an_grad = self.get_analytic_gradient(X)
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
