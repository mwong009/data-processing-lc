#!/usr/bin/python3
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

import numpy as np

from layers.generator import gen_param


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):

        if W is None:
            W = gen_param(name='W', shape=(n_in, n_out))

        if b is None:
            b = gen_param(name='b', shape=(n_out,))

        self.W = W
        self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.input = input

    def negative_loglikelihood(self, y):
        return -T.mean(
            T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        )

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class UnimodalLogistic(LogisticRegression):
    def __init__(self, input, n_in, n_out, W=None, b=None):
        LogisticRegression.__init__(self, input, n_in, n_out, W=None, b=None)

        lin_out = T.dot(input, self.W) + self.b
        n = n_out - 1
        k = np.arange(n_out)
        binom = (
            T.log(T.gamma(n + 1)) -
            T.log(T.gamma(k + 1)) -
            T.log(T.gamma(n - k + 1))
        )
        pre_softmax = -(n-k) * lin_out - n * T.nnet.softplus(-lin_out)

        self.p_y_given_x = T.nnet.softmax(binom + pre_softmax)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
