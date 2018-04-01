import time
import os
from MultiLayerPerception import *
import cPickle as pickle
from utils import save_model, load_config_with_defaults
import logging
from logging.config import fileConfig

logger = logging.getLogger()

from argparse import ArgumentParser

argument = ArgumentParser(usage='it is usage tip', description='no')
argument.add_argument('--cfgfile', default='./config/model.json', type=str, help='model config')
argument.add_argument('--basic_cfgfile', default='./config/basic.json', type=str, help='basic model config')

arguments = argument.parse_args()
args = load_config_with_defaults(arguments.cfgfile, arguments.basic_cfgfile)
print args
train_file = args['train_file']
valid_file = args['valid_file']
test_file = args['test_file']
checkpoint = args['checkpoint']
n_batch = args['batch_size']
optimizer = args['optimizer']
clip_freq = args['clip_freq']
NEPOCH = args['epochs']
dropout = args['dropout']
lr = args['learning_rate']
hidden_layers = args['hidden_layers']
disp_freq = 50
save_freq = 1000
valid_freq = 1000
test_freq = 1000


def tfpn(y_pred, y):
    tp = fn = fp = tn = 0
    if len(y_pred) != len(y):
        print("y_pred:", len(y_pred), ' y:', len(y))
        raise TypeError('y should have the same shape as self.y_pred')
    for i in range(len(y)):
        ly = y[i][0] > y[i][1]
        lp = y_pred[i][0] > y_pred[i][1]
        if ly == 0 and lp == 0:
            tp += 1
        elif ly == 0 and lp == 1:
            fn += 1
        elif ly == 1 and lp == 0:
            fp += 1
        elif ly == 1 and lp == 1:
            tn += 1
    prec = tp * 1.0 / (tp + fp + 0.0001)
    rec = tp * 1.0 / (tp + fn + 0.0001)
    f1 = 2 * prec * rec / (prec + rec + 0.0001)
    return tp, fn, fp, tn, prec, rec, f1


def train():
    print 'loading dataset...'
    logger.info('loading dataset...')
    with open(train_file[0], 'r') as f:
        x_train = pickle.load(f)
    with open(train_file[1], 'r') as f:
        y_train = pickle.load(f)
    with open(valid_file[0], 'r') as f:
        x_valid = pickle.load(f)
    with open(valid_file[1], 'r') as f:
        y_valid = pickle.load(f)
    print '=' * 40
    print x_train.shape
    print '-' * 40
    print y_train.shape
    print '=' * 40
    print 'building model...'
    model = MLP(nlayers=hidden_layers, optimizer=optimizer, p=dropout)
    print 'training start...'
    start = time.time()

    for epoch in xrange(NEPOCH):
        error = []
        accs = []
        maxiter = x_train.shape[0] / n_batch
        for i in range(maxiter):
            x = x_train[i * n_batch:(i + 1) * n_batch]
            y = y_train[i * n_batch:(i + 1) * n_batch]
            cost, acc = model.train(x, y, lr)
            error.append(cost)
            accs.append(acc)
            # print cost
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1

        error = np.asarray(error).flatten()
        accs = np.asarray(accs).flatten()
        print 'epoch: %d, cost: %f, acc: %f' % (epoch, np.average(error), np.average(accs))
        filename = './model/param_{:.2f}.pkl'.format((time.time() - start))
        logger.info('dumping...' + filename)
        save_model(filename, model)
        for input_feat, label in zip(x_valid, y_valid):
            input_feat = input_feat.reshape([1, -1])
            predicted = model.test(input_feat)
            # print predicted, label

        # logger.info('validation cost: %f perplexity: %f' % (valid_cost, np.exp(valid_cost)))
        #
        # test_cost = evaluate_ppl(test_data, model)
        # logger.info('test cost: %f perplexity: %f' % (test_cost, np.exp(test_cost)))

    print "Finished. Time = " + str(time.time() - start)


def test():
    test_data = TextIterator(test_datafile, n_batch=n_batch)
    valid_data = TextIterator(valid_datafile, n_batch=n_batch)
    model = RNNLM(n_input, n_hidden, vocabulary_size, rnn_cell, optimizer, p)
    if os.path.isfile(args.model_dir):
        print 'loading pretrained model:', args.model_dir
        model = load_model(args.model_dir, model)
    else:
        print args.model_dir, 'not found'
    mean_cost = evaluate_ppl(valid_data, model)
    print 'valid perplexity:', np.exp(mean_cost),
    mean_wer = evaluate_wer(valid_data, model)
    print 'valid WER:', mean_wer

    mean_cost = evaluate_ppl(test_data, model)
    print 'test perplexity:', np.exp(mean_cost),
    mean_wer = evaluate_wer(test_data, model)
    print 'test WER:', mean_wer


if __name__ == '__main__':
    if args['mode'] == 'train':
        train()
    elif args['mode'] == 'test':
        test()
