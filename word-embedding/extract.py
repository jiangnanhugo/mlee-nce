# -*- coding:utf-8 -*- 
import os
import codecs
from xml.etree import cElementTree as ET
import xml.dom.minidom

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
    fr=codecs.open('./abstract.txt','r','utf-8')
    index=0
    patt=r"([^a-zA-Z0-9 ])"
    fw=codecs.open('./abstract.EP.txt','w','utf-8')
    al=1114645
    while True:
        lines=fr.readlines(100000)
        if not lines: break
        for line in lines:
            index+=1
            if index% 10000==0:
                sys.stdout.write("\r {}%".format(index*100./al))
                sys.stdout.flush()
            line=re.sub(r"([^a-zA-Z0-9 \-])",r" \1 ",line)
            line=re.sub(r'\s+',' ',line)
            fw.write(line.strip()+'\n')

        fw.flush()
    fw.close()
import cPickle as pickle
def counts(filepath='abstract.EP.txt'):
    all_text=open(filepath).read()
    sentences=all_text.split('\n')   
    word_dict={}
    for sentence in sentences:
        words=sentence.split(' ')
        for word in words:
            if not word in word_dict:
                word_dict[word]=1
            else:
                word_dict[word]+=1
    f=file('wordcount.pkl','w')
    pickle.dump(word_dict,f)
    f.close()
    return word_dict
from collections import defaultdict
def compare():
    with open('wordcount.pkl','r')as f:
        wordcounts=pickle.load(f)
    fr=open('triggername.txt','r')
    triggername=defaultdict(int)
    for word in fr:
        word=word.strip()
        triggername[word]+=1
    for k in triggername:
        print "name:\t",k,"\tex-counts:\t", 
        try:
            print wordcounts[k],
        except:
            print '0',
        print "\tin-counts:\t",triggername[k]


if __name__=="__main__":
    extract_content()
    #end_processing()
    #counts()
    #compare()


