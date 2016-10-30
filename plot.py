
import sys, logging
import matplotlib.pyplot as plt
import cPickle as pickle
import plotly.plotly as py
import random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('bio_tsne')



for i in range(10,11):
    token_name='dataset'+str(i)+'.pkl'

    with open(token_name,'r')as f:
        (tokens,labels)=pickle.load(f)
    vector2d_name='new_values'+str(i)+'.txt'
    with open(vector2d_name,'r')as f:
        vectors2d=pickle.load(f)


    print 'loaded....'
    plt.figure(figsize=(15, 15))
    for j,label in enumerate(labels):
        x, y = vectors2d[j,:]
        
        if label==1:
        	plt.scatter(y, x,c='r',marker=(5,2))
        else:
        	plt.scatter(y, x,c='black',marker="+")
            #plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.ylim(-20,20)
    plt.xlim(-20,20)
    plt.savefig('tsne_15_'+str(i)+'.pdf',format="pdf")
    print "saved"

