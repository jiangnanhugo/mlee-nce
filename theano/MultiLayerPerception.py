import theano

if theano.config.device == 'cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from dense import *
from updates import sgd, adam, rmsprop
from softmax import softmax


class MLP(object):
    def __init__(self, type='relu', nlayers=[300, 50, 10, 2], optimizer='sgd', p=0.5):
        self.x = T.matrix('batched_x')
        self.y = T.matrix('batched_y')
        self.nlayers = nlayers
        self.type = type
        self.optimizer = optimizer
        self.p = p
        self.is_train = T.iscalar('is_train')

        self.rng = RandomStreams(1234)
        self.build()

    def build(self):
        print 'building rnn cell...'
        hidden_layer = None
        input = self.x
        self.params = []
        print range(1, len(self.nlayers) - 1)
        for i in range(1, len(self.nlayers) - 1):
            if self.type == 'sigmoid':
                hidden_layer = sigmoid_layer(input, self.nlayers[i - 1], self.nlayers[i], prefix='hid_' + str(i))
            elif self.type == 'relu':
                hidden_layer = relu_layer(input, self.nlayers[i - 1], self.nlayers[i], prefix='hid_' + str(i))
            elif self.type == 'selu':
                hidden_layer = selu_layer(input, self.nlayers[i - 1], self.nlayers[i], prefix='hid_' + str(i))
            # Dropout
            if self.p > 0:
                drop_mask = self.rng.binomial(n=1, p=1 - self.p, size=hidden_layer.activation.shape, dtype=theano.config.floatX)
                input = T.switch(self.is_train, hidden_layer.activation * drop_mask, hidden_layer.activation * (1 - self.p))
            else:
                input = T.switch(self.is_train, hidden_layer.activation, hidden_layer.activation)
            self.params += hidden_layer.params

        print 'building softmax output layer...'
        output_layer = softmax(input, self.nlayers[-2], self.nlayers[-1])
        self.params += output_layer.params
        cost = T.sum(T.nnet.categorical_crossentropy(output_layer.activation, self.y))
        acc = T.sum(T.eq(output_layer.predict, T.max(self.y,axis=-1)))

        lr = T.scalar("lr")
        gparams = [T.clip(T.grad(cost, p), -3, 3) for p in self.params]
        updates = None
        if self.optimizer == 'sgd':
            updates = sgd(self.params, gparams, lr)
        elif self.optimizer == 'adam':
            updates = adam(self.params, gparams, lr)
        elif self.optimizer == 'rmsprop':
            updates = rmsprop(params=self.params, grads=gparams, learning_rate=lr)

        self.train = theano.function(inputs=[self.x, self.y, lr],
                                     outputs=[cost, acc],
                                     updates=updates,
                                     givens={self.is_train: np.cast['int32'](1)})

        self.test = theano.function(inputs=[self.x],
                                    outputs=output_layer.predict,
                                    givens={self.is_train: np.cast['int32'](0)})
