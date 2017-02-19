from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import nn


def main():

    #trivial test



    X = np.random.randn(5)
    A = np.random.randn(10, 5)
    bias = np.random.randn(10)

    Y = np.dot(A, X) + bias

    model = nn.Sequential()
    model.add(nn.Linear(5, 10))
    model.add(nn.Tanh(10))
    model.add(nn.Linear(10,10))
    model.add(nn.MSE(10, Y))

    for i in xrange(1000000):
        print ("Loss", model.forward(X))
        model.backward()


if __name__ == '__main__':
    main()