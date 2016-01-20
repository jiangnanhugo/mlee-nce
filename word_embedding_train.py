# -*- coding: utf-8 -*-
# import the built-in logging module and configure it so that Word2Vec
# create nice output messages

import logging 
import re
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
import sys
reload(sys)
sys.setdefaultencoding('utf8')

#set values for various parameters
num_features=300  #Word vector dimensionality
min_word_count=40 #minimum word count
num_workers=10    #Number of threads to run in parallel
context=20        #context window size
downsampling=1e-3 # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
sentences=word2vec.LineSentence('corpus_stem.txt')
model=word2vec.Word2Vec(sentences,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-dfficient

model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and save
# the model for later use. You can load it later using  Word2Vec.load()
model_name="300features_40minwords_10context"
model.save(model_name+'.model')

# Store the learned weights, in a format the original C tool understands
# or, import word weights created by the (faster) C word2vec
# this way, you can switch between the C/Python toolkits easily
model.save_word2vec_format(model_name+'.bin',binary=True)
