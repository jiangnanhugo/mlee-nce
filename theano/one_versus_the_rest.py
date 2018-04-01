# translate the multi-label classification into binary classification
from aliasmethod import *
import cPickle as pickle
import os
from collections import defaultdict

alpha = 0.75
from gensim.models import word2vec

base_dir = '../dumped'

stemming = False
if stemming:
    word2vec_file = os.path.join(base_dir, 'stemed.300features_40minwords_10context.model')
else:
    word2vec_file = os.path.join(base_dir, '300features_40minwords_10context.model')


def get_word_freq_dist(words):
    word_freq = defaultdict(int)
    for w in words:
        word_freq[w] += 1

    sorted_words = sorted(word_freq.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    return sorted_words


def convert_nce(data_file, label_file, unigram_file, max_categories=19, k=20):
    with open(data_file, 'r')as f:
        data = pickle.load(f)
    with open(label_file, 'r')as f:
        labels = pickle.load(f)
    with open(unigram_file, 'r')as f:
        words = pickle.load(f)
        vocab_freq = get_word_freq_dist(words)
    vocab_p = Q_w(vocab_freq, alpha)
    J, q = alias_setup(vocab_p)
    E = word2vec.Word2Vec.load(word2vec_file)

    for type in range(max_categories):
        type_data = []
        type_label = []

        for i in range(len(labels)):
            if labels[i] == type:  ### the true categories ones
                # print data[i]
                type_data.append(E[data[i]])
                type_label.append(np.asarray([0., 1.], dtype=np.float32))  ## the positive ones
                negs = negative_sample(k, J, q, words)
                for it in negs:  # negative samples
                    # print it, it in E
                    type_data.append(E[it])
                    type_label.append(np.asarray([1., 0.], dtype=np.float32))
        print len(type_data)
        with open(os.path.join(base_dir, "category_" + str(type) + "_k_" + str(k) + '_' + data_file.split('/')[-1]), 'w')as f:
            pickle.dump(np.asarray(type_data), f)
        print len(type_label)
        with open(os.path.join(base_dir, "category_" + str(type) + "_k_" + str(k) + '_' + label_file.split('/')[-1]), 'w')as f:
            pickle.dump(np.asarray(type_label), f)


def convert_multilabel_into_binary(data_file, label_file, max_categories=19):
    with open(data_file, 'r')as f:
        data = pickle.load(f)
    with open(label_file, 'r')as f:
        labels = pickle.load(f)

    E = word2vec.Word2Vec.load(word2vec_file)
    for type in range(max_categories):
        type_data = []
        type_label = []
        for i in range(len(labels)):
            type_data.append(E[data[i]])
            if labels[i] == type:
                type_label.append(np.asarray([0., 1.], dtype=np.float32))
            else:
                type_label.append(np.asarray([1., 0.], dtype=np.float32))

        print len(type_data)
        with open(os.path.join(base_dir, "category_" + str(type) + '_' + data_file.split('/')[-1]), 'w')as f:
            pickle.dump(np.asarray(type_data), f)
        print len(type_label)
        with open(os.path.join(base_dir, "category_" + str(type) + '_' + label_file.split('/')[-1]), 'w')as f:
            pickle.dump(np.asarray(type_label), f)


data_file = '../data/X_train_word.pkl'
label_file = '../data/y_train.pkl'
unigram_file = '../data/X_train_word.pkl'
convert_nce(data_file, label_file, unigram_file, k=5)

# data_file = '../data/X_valid_word.pkl'
# label_file = '../data/y_valid.pkl'
# convert_multilabel_into_binary(data_file, label_file)
# #
# data_file = '../data/X_test_word.pkl'
# label_file = '../data/y_test.pkl'
# convert_multilabel_into_binary(data_file, label_file)
