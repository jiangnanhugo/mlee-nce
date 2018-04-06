import numpy as np
import theano
import theano.tensor as T


class sigmoid_layer(object):
    def __init__(self, input, n_input, n_output, prefix='layer_'):
        init_W = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_input + n_output)),
                                              high=np.sqrt(6. / (n_input + n_output)),
                                              size=(n_input, n_output)),
                            dtype=theano.config.floatX)
        W = theano.shared(value=init_W, name=prefix + '_basic_W', borrow=True)
        init_b = np.zeros((n_output,), dtype=theano.config.floatX)
        b = theano.shared(value=init_b, name=prefix + '_basic_b', borrow=True)

        self.activation = T.nnet.sigmoid(T.dot(input, W) + b)

        self.params = [W, b]


class relu_layer(object):
    def __init__(self, input, n_input, n_output, prefix='layer_'):
        init_W = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_input + n_output)),
                                              high=np.sqrt(6. / (n_input + n_output)),
                                              size=(n_input, n_output)),
                            dtype=theano.config.floatX)
        W = theano.shared(value=init_W, name=prefix + '_relu_W', borrow=True)
        init_b = np.zeros((n_output,), dtype=theano.config.floatX)
        b = theano.shared(value=init_b, name=prefix + '_relu_b', borrow=True)

        self.activation = T.nnet.relu(T.dot(input, W) + b)
        self.params = [W, b]


class selu_layer(object):
    def __init__(self, input, n_input, n_output, prefix='layer_'):
        init_W = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_input + n_output)),
                                              high=np.sqrt(6. / (n_input + n_output)),
                                              size=(n_input, n_output)),
                            dtype=theano.config.floatX)
        W = theano.shared(value=init_W, name=prefix + '_selu_W', borrow=True)
        init_b = np.zeros((n_output,), dtype=theano.config.floatX)
        b = theano.shared(value=init_b, name=prefix + '_selu_b', borrow=True)

        self.activation = T.nnet.selu(T.dot(input, W) + b)
        self.params = [W, b]
