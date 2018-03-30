# -*- coding:utf-8 -*- 
import os
import codecs
from xml.etree import cElementTree as ET
import xml.dom.minidom
from nltk.stem import PorterStemmer
pt=PorterStemmer()

def extract_content():
    fw=codecs.open('abstracts.txt','w','utf-8')
    for subdir, dirs, files in os.walk('data'):
        for f in files:
            filename=os.path.join(subdir,f)
            #codecs.open(os.path.join(subdir,f),'r','utf-8').read())
            try:
                e=ET.parse(filename).getroot()
            except:
                print filename
            #abss= root.getElementsByTagName("AbstractText")
            #for atype in e.findall('AbstractText'):
            for ne in e.iter('AbstractText'):
                if ne.text:
                    fw.write(ne.text+"\n")
        fw.flush()
import re
import sys
def end_processing():
    fr=codecs.open('./abstracts.txt','r','utf-8')
    #fr=open('abstracts.txt','r')
    index=0
    fw=codecs.open('./abstracts.EP.txt','w','utf-8')
    al=30762915
    for line in fr:
    #while True:
        #lines=fr.readlines(1000)
        #lines=fr.readlines(10)
        
        #if not lines: 
        #    break
        #for line in lines:
        index+=1
        if index% 1000==0:
            sys.stdout.write("\r {}%".format(index*100./al))
            sys.stdout.flush()
        line=re.sub(r"([^a-zA-Z0-9 \-])",r" \1 ",line)
        line=re.sub(r'\s+',' ',line)
        fw.write(line.strip()+'\n')
    fw.flush()
    fw.close()

def ascii_lower():
    fr=codecs.open('./abstracts.EP.txt','r','utf-8')
    #fr=open('abstracts.txt','r')
    index=0
    fw=codecs.open('./abstracts.EP.ascii.lower.txt','w','ascii')
    al=30762915
    for line in fr:
        index+=1
        if index% 1000==0:
            sys.stdout.write("\r {}%".format(index*100./al))
            sys.stdout.flush()
        #line=re.sub(r"([^a-zA-Z0-9 \-])",r" \1 ",line)
        #line=re.sub(r'\s+',' ',line)
        words=line.strip().lower().split(' ')
        for w in words:
            try:
                fw.write(w+' ')
            except:
                pass
        fw.write('\n')
    fw.flush()
    fw.close()

from collections import defaultdict
import cPickle as pickle
def counts(filepath='abstract.EP.txt'):
    all_text=open(filepath)
    word_dict=defaultdict(int)
    index=0
    al=30762915
    for sentence in all_text:
        index+=1
        if index% 100000==0:
            sys.stdout.write("\r {}%".format(index*100./al))
            sys.stdout.flush()
        words=sentence.strip().split(' ')
        for word in words:
            word_dict[word]+=1
    f=file('wordcount.pkl','w')
    pickle.dump(word_dict,f)
    f.close()
    return word_dict

def stemming():
    wordcounts=pickle.load(open('wordcount.pkl','r'))
    stemed_words=dict()
    for k in wordcounts:
        w=pt.stem(k)
        stemed_words[k]=w
    with open('stemed_word.pkl','w')as f:
        pickle.dump(stemed_words,f)

def rewrite_stemed_sentence(filepath='abstracts.EP.ascii.lower.tcbb.txt'):
    all_text=open(filepath)
    with open('stemed_word.pkl')as f:
        word_dict=pickle.load(f)
    index=0
    al=21424031
    fw=open('stemed'+filepath,'w')
    for sentence in all_text:
        index+=1
        if index% 100000==0:
            sys.stdout.write("\r {}%".format(index*100./al))
            sys.stdout.flush()
        words=sentence.strip().split(' ')
        for word in words:
            fw.write(word_dict[word]+' ')
        fw.write('\n')
    fw.flush()
    fw.close()
    #f=file('wordcount.pkl','w')
    #pickle.dump(word_dict,f)
    #f.close()


from collections import defaultdict
def compare(wordfreq_filename,trigger_filename,nontrigger_filename):
    if wordfreq_filename=='wordcount.pkl':
        #wordfreq=word_stem()
        with open('wordcount.pkl','r')as f:
            wordfreq=pickle.load(f)
    else:
        with open('wordfreq.pkl','r')as f:
            wordfreq=pickle.load(f)
    def word_stem():
        with open('wordcount.pkl','r')as f:
            wordcounts=pickle.load(f)
        print 'init....'
        wordfreq=defaultdict(int)
        for k in wordcounts:
            name=pt.stem(k)
            wordfreq[name]+=wordcounts[k]
        with open('wordfreq.pkl','r')as f:
            pickle.dump(wordfreq,f)
        return wordfreq

    fr=open(trigger_filename,'r')
    triggername=defaultdict(int)
    for word in fr:
        word=word.strip().lower()
        triggername[word]+=1
    fr=open(nontrigger_filename,'r')
    nontriggername=defaultdict(int)
    for w in fr:
        word=w.strip().lower()
        nontriggername[word]+=1
    for k in triggername:
        print "name:\t",k,"\tex-counts:\t", 
        try:
            print wordfreq[k],
        except:
            print '0',
        print "\tin-trigger-counts:\t",triggername[k],
        print '\tin-nonitrigger-counts:\t',        
        if k in nontriggername:
            print nontriggername[k]
        else:
            print 0
    print 'metastasis',triggername['metastasis'],nontriggername['metastasis'],wordfreq['metastasis']



if __name__=="__main__":

    #extract_content()
    #end_processing()
    #ascii_lower()
    #counts('abstracts.EP.ascii.lower.tcbb.txt')
    #stemming()
    rewrite_stemed_sentence()
    #compare('wordcount.pkl','trigger.noStem.txt')
    #compare('wordcount.pkl','trgger.stemed.lowered.test.txt')
    #compare('wordcount.pkl','trgger.stemed.lowered.test.txt','nontrigger.stem.lower.txt')
    #compare('wordcount.pkl','trgger.stemed.lowered.test.txt','nontrigger.noStem.noLower.txt')
 
