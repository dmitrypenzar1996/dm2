from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import nn


def main():

    #trivial test

    X = np.random.randn(100, 5)
    A = np.random.randn(10, 5)
    bias = np.random.randn(10)

    Y = np.dot(A, X.T).T + bias
    print (np.dot(A, X.T).T.shape)

    model = nn.Sequential()
    model.add(nn.Linear(5, 10))
    model.add(nn.Tanh(10))
    model.add(nn.Linear(10, 10))
    model.add(nn.MSE(10))

    print("Batch mode")
    for i in xrange(10000):
        print ("Loss", model.forward(X, Y))
        model.backward(alpha = 0.1)

    print("Single mode")
    for i in xrange(10000):
        for j in xrange(X.shape[0]):
            print ("Loss", model.forward(X[j], Y[j]))
            model.backward(alpha = 0.1)



if __name__ == '__main__':
    main()