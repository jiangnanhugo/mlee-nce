from nltk.stem import PorterStemmer
import os

pt = PorterStemmer()
import numpy as np
from gensim.models import word2vec
import cPickle as pickle
from util import get_sentences, get_sentences2
import warnings

warnings.filterwarnings('ignore')
type_list = ['Cell_proliferation', 'Development', 'Blood_vessel_development', 'Growth', 'Death', 'Breakdown', 'Remodeling',
             'Synthesis', 'Gene_expression', 'Transcription', 'Catabolism', 'Phosphorylation', 'Dephosphorylation', 'Localization',
             'Binding', 'Regulation', 'Positive_regulation', 'Negative_regulation', 'Planned_process']

input_size = 300
punc_list = ['$', '.', '"', ',', ';', ':', ')', ']', '}', '!', '?']
end_list = ['.', ':', ';', '?', '!']

sw_list = open('stop_word.txt').read().split('\n')

stemming = False
if stemming:
    word2vec_file = 'stemed.300features_40minwords_10context.model'
else:
    word2vec_file = '300features_40minwords_10context.model'


def filter_func(string):
    if len(string) <= 1:    return ""
    string = filter(str.isalpha, string)
    return string.lower()


class trigger(object):
    def __init__(self, name, label, st, ed, filepath):
        self.trigger_name = name
        self.trigger_sentence_index = -1
        self.trigger_label = label
        self.trigger_st = st
        self.trigger_ed = ed
        self.trigger_filepath = filepath

    def set_sentence_index(self, index):
        self.trigger_sentence_index = index

    def string(self):
        return "data_path: %s; start: %s;end %s; name: %s; label: %s; sentence: %s;" % (
            self.trigger_filepath, self.trigger_st, self.trigger_ed, self.trigger_name, self.trigger_label, self.trigger_sentence_index)


def is_trigger(s, filepath):
    index = s.find('$')
    if index < 0:     return False
    for i in range(index, len(s) - 1):
        if not s[i] == '$':     return False

    if not s[-1] in punc_list:
        return False

    return True


def save_trigger(trigger_list, filepath='trigger.txt'):
    fw = open(filepath, 'a')
    for trigger in trigger_list:
        fw.write(trigger.string() + '\n')
    fw.close()


def load_trigger(filepath):
    dir_list = os.listdir(filepath)
    trigger_list = []
    sentence_index = 0
    for dirs in dir_list:
        data_path = os.path.join(filepath, dirs)
        if os.path.isfile(data_path) and dirs.endswith('txt'):
            label_path = os.path.join(filepath, dirs.split('.')[0] + '.a2')
            data = open(data_path).read()
            labels = open(label_path).read().split('\n')
            trig_list = []
            for label in labels:
                if len(label) > 1 and label[0] == 'T':  # trigger
                    cand = label.strip().split('\t')[-1]
                    words = label.strip().replace('\t', ' ')
                    item = words.split(' ')
                    if len(item) > 5:
                        # print item
                        continue
                    name = filter_func(item[4])
                    if stemming:
                        name = pt.stem(name)
                    label = item[1]
                    st = int(item[2])
                    ed = int(item[3])
                    if label in type_list:
                        trig = trigger(name, label, st, ed, data_path)
                        trig_list.append(trig)
                        rep = '$' * (ed - st)
                        data = data[:st] + rep + data[ed:]

            trig_list.sort(lambda x, y: cmp(x.trigger_st, y.trigger_st))
            word_index = 0
            # Generate data
            splited_text = data.split('\n')
            for para in splited_text:
                words = para.split(' ')
                for word in words:
                    if len(word) >= 1 and word[-1] in end_list:
                        sentence_index += 1
                    if len(word) >= 1 and is_trigger(word, data_path):
                        trig_list[word_index].set_sentence_index(sentence_index)
                        if word_index + 1 < len(trig_list) and trig_list[word_index + 1].trigger_st == trig_list[word_index].trigger_st:
                            trig_list[word_index + 1].set_sentence_index(sentence_index)
                            word_index += 2
                        else:
                            word_index += 1
            trigger_list += trig_list
    save_trigger(trigger_list)
    return trigger_list


def load_mlee(filepath):
    lists = os.listdir(filepath)
    data_sum = ''
    for item in lists:
        data_path = os.path.join(filepath, item)
        if os.path.isfile(data_path) and item.endswith('txt'):
            label_path = os.path.join(filepath, item.split('.')[0] + '.a2')
            data = open(data_path).read()
            all_label = open(label_path).read().split('\n')
            for line in all_label:
                if len(line) > 1 and line[0] == 'T':  # trigger
                    line = line.strip().replace('\t', ' ')
                    words = line.split(' ')
                    if len(words) > 5:
                        # print words
                        continue

                    if words[1] in type_list:
                        st = int(words[2])
                        ed = int(words[3])
                        rep = ' ' * (ed - st)
                        data = data[:st] + rep + data[ed:]
            data_sum += data
    all_data = ''
    splited_text = data_sum.split('\n')
    for text in splited_text:
        lines = text.split(' ')
        for word in lines:
            if len(word) >= 1 and word[-1] in end_list:
                item = filter_func(word[:-1])
                if stemming:
                    item = pt.stem(item)
                # item=word[:-1]
                if len(item) >= 1 and (not item in sw_list):
                    all_data += item + '\n'
            elif len(word) >= 1:
                item = filter_func(word)
                if stemming:
                    item = pt.stem(item)
                # item=word
                if len(item) >= 1 and (not item in sw_list):
                    all_data += item + ' '
    return all_data


# load the dataset
def load_dataset(filepath, is_train=1):
    _x = []
    _xx = []
    _y = []
    trigger_list = load_trigger(filepath)
    embd = word2vec.Word2Vec.load(word2vec_file)
    word_dict = pickle.load(open('wordcount.pkl', 'rb'))
    if is_train:
        ext = 50
    else:
        ext = 1
    print ext,
    print 'loading trigger...'
    for i in range(ext):
        for item in trigger_list:
            word = str(item.trigger_name)
            if word in embd:
                _x.append(embd[word])
                _xx.append(word)
                _y.append(type_list.index(item.trigger_label))
            else:
                print word, type_list.index(item.trigger_label)
    print len(_y), ' ',
    all_text = load_mlee(filepath)
    lines = all_text.split('\n')
    print 'loading non-trigger....'
    for line in lines:
        words = line.split(' ')
        for word in words:
            # print word
            if word in embd:
                if word_dict[word] <= 1000 and is_train == 1:
                    _x.append(embd['unknown'])
                    _xx.append('unknown')
                else:
                    _x.append(embd[word])
                    _xx.append(word)
                _y.append(19)
    print len(_y)
    return np.asarray(_x), _xx, np.asarray(_y)


if __name__ == "__main__":
    train_set_x, train_word_x, train_set_y = load_dataset('./mlee/train', is_train=1)
    with open('data/X_train.pkl', 'w') as f:
        pickle.dump(train_set_x, f)
        f.flush()
        f.close()
    with open('data/X_train_word.pkl', 'w') as f:
        pickle.dump(train_word_x, f)
        f.flush()
        f.close()
    with open('data/y_train.pkl', 'w') as f:
        pickle.dump(train_set_y, f)
        f.flush()
        f.close()

    valid_set_x, valid_word_x, valid_set_y = load_dataset('./mlee/valid', is_train=1)
    with open('data/X_valid.pkl', 'w') as f:
        pickle.dump(valid_set_x, f)
        f.flush()
        f.close()
    with open('data/X_valid_word.pkl', 'w') as f:
        pickle.dump(valid_word_x, f)
        f.flush()
        f.close()
    with open('data/y_valid.pkl', 'w') as f:
        pickle.dump(valid_set_y, f)
        f.flush()
        f.close()

    test_set_x, test_word_x, test_set_y = load_dataset('./mlee/test', is_train=0)
    with open('data/X_test.pkl', 'w') as f:
        pickle.dump(test_set_x, f)
        f.flush()
        f.close()
    with open('data/X_test_word.pkl', 'w') as f:
        pickle.dump(test_word_x, f)
        f.flush()
        f.close()
    with open('data/y_test.pkl', 'w') as f:
        pickle.dump(test_set_y, f)
        f.flush()
        f.close()
