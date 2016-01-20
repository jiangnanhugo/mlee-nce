from nltk.stem import PorterStemmer
import os
pt = PorterStemmer()
import numpy as np
from gensim.models import word2vec
import cPickle as pickle
from util import get_sentences,get_sentences2
from lstm_autoencoder import *
import warnings

warnings.filterwarnings('ignore')
type_list = ['Cell_proliferation', 'Development', 'Blood_vessel_development', 'Growth', 'Death', 'Breakdown','Remodeling',
             'Synthesis', 'Gene_expression', 'Transcription', 'Catabolism', 'Phosphorylation', 'Dephosphorylation','Localization',
             'Binding', 'Regulation', 'Positive_regulation', 'Negative_regulation', 'Planned_process']

word2vec_file = '300features_40minwords_10context.model'

input_size = 300

punc_list=['$','.','"',',',';',':',')',']','}','!','?']
end_list=['.',':',';','?','!']


def filter_func(string):
    if len(string) <= 1:
        return ""
    string = filter(str.isalpha, string)
    return string.lower()

class trigger:
    def __init__(self,name,label,st,ed,filepath):
        self.trigger_name=name
        self.trigger_sentence_index=-1
        self.trigger_label=label
        self.trigger_st=st
        self.trigger_ed=ed
        self.trigger_filepath=filepath

    def set_sentence_index(self,index):
        self.trigger_sentence_index=index
	
    def string(self):
        return "data_path: %s; start: %s;end %s; name: %s; label: %s; sentence: %s;" %(self.trigger_filepath,self.trigger_st,self.trigger_ed,self.trigger_name,self.trigger_label,self.trigger_sentence_index)

def is_trigger(s,filepath):
	index=s.find('$')
	if index < 0:
		return False

	for i in range(index,len(s)-1):
		if not s[i]=='$':
			print 'error1:',s,filepath
			return False

	if not s[-1] in punc_list:
		print 'error2:',s,filepath
		return False
	else:
		#print s,filepath
		return True


def save_trigger(trigger_list,filepath='trigger.txt'):
    fw=open(filepath,'w')
    for trigger in trigger_list:
        fw.write(trigger.string()+'\n')
    fw.close()

def load_trigger(filepath):
    dir_list=os.listdir(filepath)
    trigger_list=[]
    sentence_index=0
    for dir in dir_list:
        data_path=os.path.join(filepath,dir)
        if os.path.isfile(data_path) and dir.endswith('txt'):
            label_path=os.path.join(filepath,dir.split('.')[0]+'.a2')
            data=open(data_path).read()
            labels=open(label_path).read().split('\n')
            trig_list=[]
            for label in labels:
                if len(label)>1 and label[0]=='T': # trigger
                    words=label.strip().replace('\t',' ')
                    item=words.split(' ')
                    name=pt.stem(filter_func(item[4]))
                    label=item[1]
                    st=int(item[2])
                    ed=int(item[3])
                    if label in type_list:
                        trig=trigger(name,label,st,ed,data_path)
                        trig_list.append(trig)
                        rep='$'*(ed-st)
                        data=data[:st]+rep+data[ed:]
			
			trig_list.sort(lambda x,y:cmp(x.trigger_st,y.trigger_st))
			word_index=0
			# Generate data
            splited_text=data.split('\n')
            for para in splited_text:
                words=para.split(' ')
                for word in words:
                    if len(word)>=1 and word[-1] in end_list:
                        sentence_index+=1					
                    if len(word)>=1 and is_trigger(word,data_path):
                        trig_list[word_index].set_sentence_index(sentence_index)
                        if word_index+1<len(trig_list) and trig_list[word_index+1].trigger_st==trig_list[word_index].trigger_st:
                                trig_list[word_index+1].set_sentence_index(sentence_index)
                                word_index+=2
                        else:
                                word_index+=1
            trigger_list+=trig_list
    save_trigger(trigger_list)
    return trigger_list


def load_mlee(filepath):
    lists = os.listdir(filepath)
    all_data =''    
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
					if words[1] in type_list:
						st = int(words[2])
						ed = int(words[3])
						rep=' '*(ed-st)
						data=data[:st]+rep+data[ed:]
            all_data+=data


    all_data=''	
    splited_text=all_data.split('\n')
    for text in splited_text:
        lines=text.split(' ')
        for word in lines:
            if len(word)>=1 and word[-1] in end_list:
                all_data+=pt.stem(filter_func(word[:-1]))+'\n'
            else:
                all_data+=pt.stem(filter_func(word))+' '
    return all_data

def get_embd():
	return word2vec.Word2Vec.load(word2vec_file)

#load the dataset
def load_dataset(filepath):#,ae):
    train_set_x = []
    train_set_y = []
    #ae_sf = ae.get_feature(
    #sentences,max_length=get_sentences(os.path.join(filepath,'all_text'))
    #print 'max_length: ',max_length
    #ae_sf=ae.get_feature(sentences)
    all_text= load_mlee(filepath)
    trigger_list=load_trigger(filepath)
    embd=get_embd()
    for item in trigger_list:
        temp_data = np.zeros(input_size)
        word=str(item.trigger_name)
        si=item.trigger_sentence_index
        if word in embd:
            #temp_data = np.hstack((embd[word],ae_sf[si][0]))
            temp_data+=embd[word]
            train_set_x.append(temp_data)
            train_set_y.append(type_list.index(item.trigger_label))

    lines = all_text.split('\n')
    si=0
    for line in lines:
        words=line.split(' ')
        for word in words:
            if word in embd:
                #train_set_x.append(np.hstack((embd[word],ae_sf[si][0])))
                train_set_x.append(embd[word])
                train_set_y.append(19)
    return train_set_x, train_set_y



if __name__=="__main__":
    data,filepath=get_sentences()
    #ae=AE(data,filepath)
    #ae.build()
    #ae.load('LSTM_AE')
    train_set_x, train_set_y = load_dataset('./mlee/train')#,ae)
    with open('data/train_set_x', 'w') as f:
        pickle.dump(train_set_x, f)
        f.flush()
        f.close()

    with open('data/train_set_y', 'w') as f:
        pickle.dump(train_set_y, f)
        f.flush()
        f.close()
    
    print '**********loading valid set*****************'
    valid_set_x, valid_set_y = load_dataset('./mlee/valid')#, ae)
    with open('data/valid_set_x', 'w') as f:
		pickle.dump(valid_set_x, f)
		f.flush()
		f.close()

    with open('data/valid_set_y', 'w') as f:
		pickle.dump(valid_set_y, f)
		f.flush()
		f.close()

    print '**********loading testing set***************'
    test_set_x, test_set_y = load_dataset('./mlee/test')#, ae)
    with open('data/test_set_x', 'w') as f:
		pickle.dump(test_set_x, f)
		f.flush()
		f.close()

    with open('data/test_set_y', 'w') as f:
		pickle.dump(test_set_y, f)
		f.flush()
		f.close()
