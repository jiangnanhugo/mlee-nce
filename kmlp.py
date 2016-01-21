
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import cPickle as pickle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

def load_data(chosen_type):
    with open('data/train_set_x','r') as f: X_train = pickle.load(f)
    with open('data/train_set_y','r') as f: y_train = pickle.load(f)
    with open('data/valid_set_x','r') as f: X_valid = pickle.load(f)
    with open('data/valid_set_y','r') as f: y_valid = pickle.load(f)
    with open('data/test_set_x','r')  as f: X_test  = pickle.load(f)
    with open('data/test_set_y','r')  as f: y_test  = pickle.load(f)

    y_train=ovrp(y_train,chosen_type)
    y_valid=ovrp(y_valid,chosen_type)
    y_test =ovrp(y_test,chosen_type)

    return (X_train,y_train), (X_valid, y_valid),(X_test, y_test)
    

def ovrp(label_set,chosen_type):
    for i in range(len(label_set)):
        if label_set[i] == chosen_type:
            label_set[i]=1
        else:
            label_set[i]=0
    return label_set

input_size=300
batch_size = 5
nb_classes = 2
nb_epoch = 20

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_valid,y_valid),(X_test, y_test) = load_data(1)


print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)
Y_test  = np_utils.to_categorical(y_test , nb_classes)

model = Sequential()
model.add(Dense(50, input_shape=(input_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=2,
          validation_data=(X_valid, Y_valid))
score = model.evaluate(X_test, Y_test,
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
