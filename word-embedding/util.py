import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from nltk.stem import PorterStemmer
pt = PorterStemmer()
import sys
import numpy as np
import os
#from lstm_autoencoder import *
import cPickle as pickle
MAX_LENGTH=256
stopword_set = set(
    ['is', 'was', 'am', 'i', 'me', 'you', 'of', 'to', 'were', 'it', 'so', 'that', 'this', 'these', 'our', 'he', 'her',
     'his','their', 'who', 'whom', 'those', 'us', 'do', 'does', 'did', 'doing', 'have', 'had', 'has', 'having', 'would',
     'should','could', 'may', 'might', 'must', 'will', 'shall', 'can', 'might'])

def filter_func(string):
    if len(string) <= 0:
        return ''
    string = filter(str.isalpha, string)
    return pt.stem(string.lower())

#corpus init
def corpus_init(filepath='abstracts.txt'):
    all_text = open(filepath).read()
    fw = open('process_word.txt', 'w')
    splited_text = all_text.split('\n')
    lens=len(splited_text)
    count=0
    for line in splited_text:
        words=line.split(' ')
        count+=1
        perc=count*100.0/lens
        print 'Percentage:%f %%\r'% perc,
        sys.stdout.flush()
        for word in words:
            if len(word)>=1 and word[-1]=='.':
                fw.write(filter_func(word[:-1])+'\n')
            else:
                fw.write(filter_func(word)+' ')
    fw.close()

'''
def corpus_stem(filepath="process_word.txt"):
    all_text = open(filepath).read()
    fw = open('stemed_word.txt', 'w')
    splited_text = all_text.split("\n")
    for para in splited_text:
        sentences = para.split(".")
        for sentence in sentences:
            words = sentence.split(" ")
            max_length=max(max_length,len(words))
            print max_length
            for word in words:
                word = pt.stem(filter_func(word))
                if not word in stopword_set:
                    fw.write("%s " % word)
            fw.write("\n")
    fw.close()
'''

def get_dict(filepath="stemed_word.txt"):
    WORD_DICT_PKL = 'word_dict.pkl'
    INDEX_PKL = 'index.pkl'
    if os.path.isfile(WORD_DICT_PKL) and os.path.isfile(INDEX_PKL):
        word_dict = pickle.load(open(WORD_DICT_PKL, 'rb'))
        index = pickle.load(open(INDEX_PKL, 'rb'))
        print 'already exist pickle file'
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
        f = file('word_dict.pkl', 'w')
        pickle.dump(word_dict, f)
        f.close()
        f = file('index.pkl', 'w')
        pickle.dump(index, f)
        f.close()
        print 'create pickle file'
        return word_dict, index


def get_max_length(splited_text):
    max_length=-1
    for line in splited_text:
        words = line.split(" ")
        max_length=max(max_length,len(words))

    return max_length


def get_sentences(filepath="stemed_word.txt",data=None):
    word_dict, _ = get_dict()
    if data is None:
        all_text = open(filepath).read()
    else:
        all_text=data
    splited_text = all_text.split('\n')
    data = []
    max_length=get_max_length(splited_text)
    print('max_length:',max_length)
    limits=10000
    for line in splited_text:
        if not limits==0:
            limits-=1
        else:
            break
        words = line.split(" ")
        sequence = np.zeros(max_length)
        index=0
        for i in range(len(words)):
            word=words[i].strip()
            if word in word_dict:
                sequence[index]=word_dict[word]
                index+=1
        data.append(sequence)
    return np.asarray(data),max_length

def get_sentences2(filepath):
	word_dict,_=get_dict()
	all_text=open(filepath).read()
	splited_text=all_text.split('\n')
	data=[]
	for line in splited_text:
		words=line.split(" ")
		sequence=np.zeros(MAX_LENGTH)
		index=0
		for i in range(len(words)):
			word=words[i].strip()
			if word in word_dict:
				sequence[index]=word_dict[word]
				index+=1
		data.append(sequence)
	return np.asarray(data)



def summarize_file(filepath):
    lists=os.listdir(filepath)
    all_text=''

    for item in lists:
        data_path = os.path.join(filepath, item)
        if os.path.isfile(data_path) and item.endswith('txt'):
            text=open(data_path).read()
            all_text+=text
	fw=open(os.path.join(filepath,'all_text'),'w')
	splited_text=all_text.split('\n')
	for line in splited_text:
		words=line.split(' ')
		for word in words:
			if len(word)>=1 and word[-1]=='.':
				fw.write( word[:-1]+'. ')
			else:
				fw.write("%s "% word)
		fw.write("\n")
    fw.close()

def get_dir_file(filepath):
	all_texts=open(os.path.join(filepath,'all_text'),'r').read()
	splited_text=all_texts.split('\n')
	fw=open(os.path.join(filepath,'cleaned_text'),'w')
	for para in splited_text:
		lines=para.split('.')
		for line in lines:
			words=line.split(' ')
			for word in words:
				word = pt.stem(filter_func(word))
				fw.write(word+' ')
			fw.write('\n')
	fw.close()
#corpus_init('abstract_lines.txt')
#corpus_stem()
#get_dict()
'''
get_dir_file('mlee/train/')
get_dir_file('mlee/valid/')
get_dir_file('mlee/test/')

data,length=get_sentences()
ae=AE(data,length)
ae.build()
feature=ae.get_feature(data)
print feature
'''
if __name__=='__main__':
	corpus_init()
