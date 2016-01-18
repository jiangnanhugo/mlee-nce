
__docformat__ = 'restructedtext en'

import cPickle as pickle
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T




class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX),
            name='W',borrow=True)
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX),
            name='b',borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def mean_square_error(self,y):
	return T.mean((self.p_y_given_x-y)**2)
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


    def pp_errors(self,y,prob,ioi):
    	"""Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        ioi: the index that you are interested in.
        prob: the prob, which is p_y_given_x
        """
        #prob = 0.5
        #ioi = 1
        # check if y has same dimension of y_pred
        if y.ndim!=self.y_pred.ndim:
        	raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
        	inprob=self.p_y_given_x[:,ioi]
        	pt1=T.gt(inprob,prob)
        	pt2=T.eq(self.y_pred,ioi)
        	pt3=T.eq(y,ioi)
        	ppn=T.sum(pt1&pt2&pt3)
        	predn=T.sum(pt1&pt2)
        	return (ppn,predn)
        else:
        	raise NotImplementedError() 


