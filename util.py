import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from nltk.stem import PorterStemmer

pt = PorterStemmer()
import sys
import numpy as np
import os
import cPickle as pickle

MAX_LENGTH = 99
'''
stopword_set = set(
    ['is', 'was', 'am', 'i', 'me', 'you', 'of', 'to', 'were', 'it', 'so', 'that', 'this', 'these', 'our', 'he', 'her',
     'his','their', 'who', 'whom', 'those', 'us', 'do', 'does', 'did', 'doing', 'have', 'had', 'has', 'having', 'would',
     'should','could', 'may', 'might', 'must', 'will', 'shall', 'can', 'might'])
'''
punc_list = ['.', ';', ':', '!', '?']
end_list = ['.', ':', '?', '!', ';']


def filter_func(string):
    if len(string) <= 0:
        return ''
    string = filter(str.isalpha, string)
    return pt.stem(string.lower())


'''
	corpus init
    input: abstracts.txt
    output: porcess_word.txt
            one sentence per line; filter non-alphabet charracter
'''


def corpus_init(filepath='abstracts.txt'):
    all_text = open(filepath).read()
    fw = open('process_word.txt', 'w')
    splited_text = all_text.split('\n')
    lens = len(splited_text)
    count = 0
    for line in splited_text:
        words = line.split(' ')
        count += 1
        perc = count * 100.0 / lens
        print 'Percentage:%f %%\r' % perc,
        sys.stdout.flush()
        for word in words:
            if len(word) >= 1 and word[-1] in punc_list:
                fw.write(filter_func(word[:-1]) + '\n')
            else:
                fw.write(filter_func(word) + ' ')
    fw.close()


def corpus_stem(filepath='process_word.txt'):
    all_text = open(filepath).read().split('\n')
    fw = open('corpus_stem.txt', 'w')
    for line in all_text:
        words = line.split(' ')
        if len(words) > 100:
            continue
        fw.write(line + '\n')
    fw.close()


def get_dict(filepath):
    WORD_DICTIONARY = 'word_dictionary.pkl'
    if os.path.isfile(WORD_DICTIONARY):
        word_dictionary = pickle.load(open(WORD_DICTIONARY, 'rb'))
        return word_dictionary
    else:
        all_text = open(filepath).read()
        sentences = all_text.split('\n')
        word_dict = {}
        for sentence in sentences:
            words = sentence.split(' ')
            for word in words:
                if not word in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
    f = file(WORD_DICTIONARY, 'w')
    pickle.dump(word_dict, f)
    f.close()
    return word_dict


'''
def get_dict(filepath):
    WORD_DICT_PKL = 'word_dict.pkl'
    INDEX_PKL = 'index.pkl'
    if os.path.isfile(WORD_DICT_PKL) and os.path.isfile(INDEX_PKL):
        word_dict = pickle.load(open(WORD_DICT_PKL, 'rb'))
        index = pickle.load(open(INDEX_PKL, 'rb'))
        print 'load dict pickle file'
        return word_dict, index
    else:
        all_text = open(filepath).read()
        sentences = all_text.split("\n")
        word_set = set()
        for sentence in sentences:
            words = sentence.split(" ")
            for word in words:
                if not word in word_set:
                    word_set.add(word)
        word_dict = {}
        index = 0
        for word in word_set:
            word_dict[word] = index
            index += 1
        f = file(WORD_DICT_PKL, 'w')
        pickle.dump(word_dict, f)
        f.close()
        
        print 'create dict pickle file'
        return word_dict,index
'''


def get_max_length(splited_text):
    MAX_LENGTH_PKL = 'max_length.pkl'
    if os.path.isfile(MAX_LENGTH_PKL):
        max_length = pickle.load(open(MAX_LENGTH_PKL, 'rb'))
        print 'max_length loaded'
        return max_length

    max_length = -1

    for line in splited_text:
        words = line.split(" ")
        max_length = max(max_length, len(words))
    f = file(MAX_LENGTH_PKL, 'w')
    pickle.dump(max_length, f)
    f.close()
    return max_length


def get_sentences(filepath="corpus_stem.txt"):
    word_dict, lens = get_dict()
    splited_text = open(filepath).read().split('\n')
    data = []
    max_length = get_max_length(splited_text)
    print('max_length: ', max_length)
    limits = 10000
    for line in splited_text:
        if limits == 0:
            break;
        else:
            limits -= 1
        words = line.split(" ")
        sequence = np.zeros(max_length)
        index = 0
        for word in words:
            if word in word_dict:
                sequence[index] = word_dict[word] * 2.0 / lens
                index += 1
        data.append(sequence)
    return np.asarray(data), max_length


def get_sentences2(filepath):
    word_dict, lens = get_dict()
    splited_text = open(filepath).read().split('\n')
    data = []
    for line in splited_text:
        words = line.split(" ")
        sequence = np.zeros(MAX_LENGTH)
        index = 0
        for word in words:
            if word in word_dict and index < MAX_LENGTH:
                sequence[index] = word_dict[word] * 2.0 / lens
                index += 1
        data.append(sequence)
    return np.asarray(data)


def summarize_file(filepath):
    fl = os.listdir(filepath)
    all_text = ''
    for item in fl:
        data_path = os.path.join(filepath, item)
        if os.path.isfile(data_path) and item.endswith('txt'):
            text = open(data_path).read()
            all_text += text
            fw = open(os.path.join(filepath, 'all_text'), 'w')
            splited_text = all_text.split('\n')
            for line in splited_text:
                words = line.split(' ')
                for word in words:
                    if len(word) >= 1 and word[-1] in end_list:
                        fw.write(filter_func(word[:-1]) + '\n')
                    else:
                        fw.write(filter_func(word) + ' ')
    fw.close()


class trigger:
    def __init__(self, name, freq):
        self.name = name
        self.freq = freq


if __name__ == '__main__':
    print '1 corpus_init; 2 get_sentence; 3 get_max_length; 4 lstm AE; 5 all_text; 6 get sentences2; 7 corpus_stem;8 get word_dict'
    inputs = input("choose: ")
    if inputs == 1:
        corpus_init()
        corpus_stem()
    elif inputs == 2:
        get_sentences()
    elif inputs == 3:
        filepath = 'corpus_stem.txt'
        splited_text = open(filepath).read().split('\n')
        print get_max_length(splited_text)
    elif inputs == 4:
        data, length = get_sentences()
    elif inputs == 5:
        summarize_file('./mlee/train/')
        summarize_file('./mlee/valid/')
        summarize_file('./mlee/test/')
    elif inputs == 6:
        get_sentences2('./mlee/train/all_text')
        print 'terminate.'
    elif inputs == 7:
        corpus_stem()
    elif inputs == 8:
        # word_dict=get_dict('corpus_stem.txt')
        word_dict = get_dict('abstract.EP.txt')
        word_list = []
        for key in word_dict:
            word_list.append(trigger(key, word_dict[key]))
        word_list.sort(lambda x, y: cmp(x.freq, y.freq))
        for item in word_list:
            print item.name, item.freq
