#coding: utf-8
'''
author:chalresyy
'''
import pandas as pd
import numpy as np
import jieba
import xlrd
import matplotlib.pyplot as plt

from keras.preprocessing import sequence 
from keras.optimizers import SGD,RMSprop,Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU #GRU(Gated Recurrent Unit)


# train Corpus Labeling
neg = pd.read_excel('neg.xls', header=None, index=None)
neg['mark'] = 0
pos = pd.read_excel('pos.xls', header=None, index=None)
pos['mark'] = 1

# The maximum sequence length value set from the histogram and the average number of words per file
numWords = []
line = neg.readline()
counter = len(line.split())
numWords.append(counter)
import matplotlib.pyplot as plt
plt.hist(numWords, 50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axis([0, 1200, 0, 8000])
plt.show()


#Positive and negative corpus for consolidation
pn = pd.concat([pos, neg], ignore_index=True)
neglen = len(neg)
poslen = len(pos)
print('negative length:', neglen)
print('postive length:', poslen)

# segmentation
cw = lambda x:list(jieba.cut(x))
# def cw(x):
#     return list(jieba.cut(x))
pn['words'] = pn[0].apply(cw)

#Read in the comments and segmentation
comment = pd.read_excel('sum.xls')
comment = comment[comment['rateContent'].notnull()]#Read only non-empty comments
comment['words'] = comment['rateContent'].apply(cw)#

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True)

# Integrate all corpora
w = []  
for i in d2v_train:
    w.extend(i)
dict = pd.DataFrame(pd.Series(w).value_counts()) #Statistics occurrences
del w, d2v_train #Deleted variables, not data
dict['id'] = list(range(1, len(dict) + 1))
get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent)

maxlen = 50 
print("Pad sequence (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
print(pn)


x_train = np.array(list(pn['sent']))[::2] 
y_train = np.array(list(pn['mark']))[::2]
x_test = np.array(list(pn['sent']))[1::2] 
y_test = np.array(list(pn['mark']))[1::2]
x_all = np.array(list(pn['sent'])) 
y_all = np.array(list(pn['mark']))
print('Build model...')

#Initialize a neural network
model = Sequential() 
#Embedding
model.add(Embedding(len(dict)+1, 256, input_length=maxlen))

#LSTM
model.add(LSTM(activation="sigmoid", units=128, recurrent_activation="hard_sigmoid"))
#Dropout
model.add(Dropout(0.3))
#Dense
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])#SGD,RMSprop,Adagrad,Adadelta,Adam,
#model.fit(x_train,y_train,batch_size=16,epochs = 10,verbose=1,validation_data = (x_test,y_test))
score = model.evaluate(x_test, y_test, verbose=1)


#print('Test score:',score[0])
#classes = model.predict_classes(x_test)
#acc = np_utils.accuracy(classes,y_test)
#print('Test accuracy:',acc)     
#print('Test accuracy:',score[1])


# plot the result
result = model.fit(x_train,y_train,batch_size=64,epochs = 20,verbose=1,validation_data = (x_test,y_test))#epochs = 20迭代20次
plt.figure
plt.plot(result.epoch,result.history['acc'],label="acc")
plt.plot(result.epoch,result.history['val_acc'],label="val_acc")
plt.scatter(result.epoch,result.history['acc'],marker='*')
plt.scatter(result.epoch,result.history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()

plt.figure
plt.plot(result.epoch,result.history['loss'],label="loss")
plt.plot(result.epoch,result.history['val_loss'],label="val_loss")
plt.scatter(result.epoch,result.history['loss'],marker='*')
plt.scatter(result.epoch,result.history['val_loss'],marker='*')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.show()
