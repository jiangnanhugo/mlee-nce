
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import cPickle as pickle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

def load_data(label):
  with open('data/X_train','r') as f: X_train = pickle.load(f)
  with open('data/y_train','r') as f: y_train = pickle.load(f)
  with open('data/X_valid','r') as f: X_valid = pickle.load(f)
  with open('data/y_valid','r') as f: y_valid = pickle.load(f)
  with open('data/X_test','r')  as f: X_test  = pickle.load(f)
  with open('data/y_test','r')  as f: y_test  = pickle.load(f)

  y_train=ovrp(y_train,label)
  y_valid=ovrp(y_valid,label)
  y_test =ovrp(y_test,label)

  return (X_train,y_train), (X_valid, y_valid),(X_test, y_test)
  

def ovrp(label_set,label):
  n=p=0
  for i in range(len(label_set)):
    if label_set[i] == label:
      label_set[i]=1
      p+=1
    else:
      label_set[i]=0
      n+=1
  print('positive: ',p,' negative: ',n)
  return label_set

def tfpn(y_pred,y):
  tp=fn=fp=tn=0
  if len(y_pred)!=len(y):
    print("y_pred:",len(y_pred),' y:',len(y))
    raise TypeError( 'y should have the same shape as self.y_pred')
  for i in range(len(y)):
    ly=y[i][0]>y[i][1]
    lp=y_pred[i][0]>y_pred[i][1]
    if ly==0 and lp==0:   tp+=1
    elif ly==0 and lp==1:   fn+=1
    elif ly==1 and lp==0:   fp+=1
    elif ly==1 and lp==1:   tn+=1
  prec=tp*1.0/(tp+fp+0.01)
  rec=tp*1.0/(tp+fn+0.01)
  f1=2*prec*rec/(prec+rec+0.01)
  return tp,fn,fp,tn,prec,rec,f1

def mlp(label,input_size=300,batch_size = 3,nb_classes = 2,nb_epoch = 50):
  # the data, shuffled and split between tran and test sets
  (X_train, y_train), (X_valid,y_valid),(X_test, y_test) = load_data(label)

  '''
  print(X_train.shape[0], 'train samples')
  print(X_valid.shape[0], 'valid samples')
  print(X_test.shape[0], 'test samples')
  '''
  # convert class vectors to binary class matrices
  Y_train = np_utils.to_categorical(y_train, nb_classes)
  Y_valid = np_utils.to_categorical(y_valid, nb_classes)
  Y_test  = np_utils.to_categorical(y_test , nb_classes)

  model = Sequential()
  model.add(Dense(50, input_shape=(input_size,)))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2))
  model.add(Activation('softmax'))

  rms = RMSprop(lr=0.0001)
  model.compile(loss='categorical_crossentropy', optimizer=rms)

  model.fit(X_train, Y_train,
      batch_size=batch_size, 
      epochs=nb_epoch,
      verbose=2,
      validation_data=(X_valid, Y_valid))
  #score = model.evaluate(X_test, Y_test, verbose=0)


  Y_pred=model.predict(X_test,verbose=0)
  print(tfpn(Y_pred,Y_test))
if __name__=='__main__':
  #for i in range(19):
  #for epoch in range(1,20):
    mlp(label=16,nb_epoch=5,batch_size=3)
