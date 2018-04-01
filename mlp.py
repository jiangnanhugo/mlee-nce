"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'

import os
import sys
import time
import numpy
import theano
import theano.tensor as T
import cPickle as pickle
from logistic_sgd import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# type 19 is the NONE_type,0-18 is the trigger type
type_list = ['Cell_proliferation', 'Development', 'Blood_vessel_development', 'Growth', 'Death', 'Breakdown', 'Remodeling',
             'Synthesis', 'Gene_expression', 'Transcription', 'Catabolism', 'Phosphorylation', 'Dephosphorylation', 'Localization',
             'Binding', 'Regulation', 'Positive_regulation', 'Negative_regulation', 'Planned_process']


class HiddenLayer(object):
    def __init__(self, rng, is_train, input, n_in, n_out, W=None, b=None, p=1.0,
                 activation=T.nnet.sigmoid):  # tanh):
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        # self define begin
        output = (lin_output if activation is None
                  else activation(lin_output))
        train_output = output * srng.binomial(size=(n_out,), p=p)
        self.output = T.switch(T.neq(is_train, 0), train_output, p * output)
        # self define end
        self.params = [self.W, self.b]

    def drop(self, input_activation):
        """
        :type inpiut:numpy.aarray 
        :param input: layer or weight matrix on which dropout resp,dropconnect is applied

        :type p:float or double between 0. and 1.
        :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
        """
        mask = self.srng.binomial(n=1, p=self.p, size=input_activation.shape, dtype=theano.config.floatX)
        return input_activation * mask


class MLP(object):

    def __init__(self, rng, is_train, input, n_in, n_hidden, n_out, drop_p=0.5):
        self.hiddenLayer = HiddenLayer(rng=rng, is_train=is_train, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh, p=drop_p)

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
                abs(self.hiddenLayer.W).sum()
                + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
                (self.hiddenLayer.W ** 2).sum()
                + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (self.logRegressionLayer.negative_log_likelihood)
        self.errors = self.logRegressionLayer.errors
        # self define
        self.pp_errors = self.logRegressionLayer.pp_errors
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.max_prob = self.p_y_given_x[T.arange(input.shape[0]), self.y_pred]
        # self define end
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.input = input


def load_data(chosen_type):
    with open('data/train_set_x', 'r') as f: train_set_x = pickle.load(f)
    with open('data/train_set_y', 'r') as f: train_set_y = pickle.load(f)
    with open('data/valid_set_x', 'r') as f: valid_set_x = pickle.load(f)
    with open('data/valid_set_y', 'r') as f: valid_set_y = pickle.load(f)
    with open('data/test_set_x', 'r')  as f: test_set_x = pickle.load(f)
    with open('data/test_set_y', 'r')  as f: test_set_y = pickle.load(f)

    train_set_y = one_versus_rest_preprocess(train_set_y, chosen_type)
    valid_set_y = one_versus_rest_preprocess(valid_set_y, chosen_type)
    test_set_y = one_versus_rest_preprocess(test_set_y, chosen_type)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


def one_versus_rest_preprocess(label_set, chosen_type):
    for i in range(len(label_set)):
        if label_set[i] == chosen_type:
            label_set[i] = 1
        else:
            label_set[i] = 0
    return label_set


def tp_fn_fp_tn(y, y_pred):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    if len(y_pred) != len(y):
        print "y_pred:", len(y_pred), ' y:', len(y)
        raise TypeError('y should have the same shape as self.y_pred')
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y[i] == 0 and y_pred[i] == 1:
            fp += 1
        else:
            tn += 1
    precision = tp * 1.0 / (tp + fp + 0.01)
    recall = tp * 1.0 / (tp + fn + 0.01)
    f1_score = 2 * precision * recall / (precision + recall + 0.01)
    return tp, fn, fp, tn, precision, recall, f1_score


def test_mlp(chosen_type, learning_rate_start=1e-2, learning_rate_end=1e-4, L1_reg=0.000, L2_reg=0.001, n_epochs=10000,
             batch_size=11, n_hidden=50, drop_p=1.0, model_err_thresh=0.001, prob_thresh=0.5):
    datasets = load_data(chosen_type)

    train_set = datasets[0]
    valid_set = datasets[1]
    test_set = datasets[2]

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    test_y_numpy = test_set_y.eval()
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_length = test_set_x.get_value(borrow=True).shape[0]

    '''
    print "train set:",train_set_x.get_value(borrow=True).shape[0],"train label:",len(train_set_y.eval())
    print "valid set:",valid_set_x.get_value(borrow=True).shape[0],"valid label:",len(valid_set_y.eval())
    print "test set:", test_set_x.get_value(borrow=True).shape[0], "test label:", len(test_y_numpy)
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    '''

    # allocate symbolic variables for the data
    lrlist = numpy.arange(learning_rate_start, learning_rate_end, (learning_rate_end - learning_rate_start) / n_epochs)
    learning_rate = T.scalar('lr')  # learning rate to use
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    is_train = T.iscalar('is_train')  # pseudo boolean for switching between training and prediction
    rng = numpy.random.RandomState(1234)

    input_dimension = train_set_x.get_value(borrow=True).shape[1]
    # print 'input_dimension:',input_dimension

    # construct the MLP class
    classifier = MLP(rng=rng, is_train=is_train, input=x, n_in=input_dimension, n_hidden=n_hidden, n_out=2, drop_p=drop_p)

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    model_error = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)})

    pp_error = theano.function(
        inputs=[index],
        outputs=classifier.pp_errors(y, prob_thresh, 1),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)})

    test_prob = theano.function(
        inputs=[index],
        outputs=classifier.logRegressionLayer.p_y_given_x,
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)})

    predict_model = theano.function(
        inputs=[index],
        outputs=classifier.logRegressionLayer.y_pred,
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)})

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size],
            is_train: numpy.cast['int32'](0)})

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    updates = []
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    train_model = theano.function(
        inputs=[index, learning_rate],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.cast['int32'](1)})
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    # print '... training'

    # early-stopping parameters
    patience = 20  # look as this many examples regardless
    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    too_fit = False
    tn_hist = []

    while (epoch < n_epochs) and (True or not done_looping) and (not too_fit):

        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index, lrlist[epoch])

        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)

        model_losses = [model_error(i) for i in xrange(n_train_batches)]
        this_model_loss = numpy.mean(model_losses)
        tn_hist.append({'train_loss': this_model_loss, 'valid_loss': this_validation_loss})

        '''
        pp_losses=[pp_error(i) for i in xrange(n_valid_batches)]
        prednlist=[te[1] for te in pp_losses]
        ppnlist=[te[0] for te in pp_losses]
        predn=numpy.sum(prednlist)
        ppn=numpy.sum(ppnlist)
        if predn>0:
            pp=float(ppn)/predn
        else:
            pp=0.0
        '''
        # print 'e %i, verr %2.1f %% , merr %2.1f %%  \r' % (epoch, this_validation_loss * 100., this_model_loss*100.)
        epoch = epoch + 1
        if this_model_loss < model_err_thresh:
            too_fit = True

    end_time = time.clock()
    y_pred = []
    for i in xrange(n_test_batches):
        y_pred = y_pred + list(predict_model(i))

    tp, fn, fp, tn, precision, recall, f1_score = tp_fn_fp_tn(test_y_numpy, y_pred)
    return "tp: %d \t fn: %d \t fp: %d \t tn: %d \t Recall: %.2f %%\t Precision: %.2f %%\t F1 score: %.2f %%, type: %s \n" % (
        tp, fn, fp, tn, recall * 100, precision * 100, f1_score * 100, type_list[chosen_type])

    # print >> sys.stderr, ('The code for file '+os.path.split(__file__)[1]+' ran for %.2fm'%((end_time - start_time) / 60.))
    # visualize the learning process.


if __name__ == '__main__':
    result_file = open('result_file', 'w')
    for i in range(0, 19):
        result_string = test_mlp(i)
        print result_string
        result_file.write(result_string)

    result_file.flush()
    result_file.close()
