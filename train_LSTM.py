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
from keras.layers.recurrent import LSTM,GRU #GRU(Gated Recurrent Unit,相比LSTM模型参数减少，性能不相上下)

# from __future__ import absolute_import #导入3.x 的特征函数
# from __future__ import print_function
#给训练语聊贴上标签
neg = pd.read_excel('neg.xls', header=None, index=None)
neg['mark'] = 0
pos = pd.read_excel('pos.xls', header=None, index=None)#读取训练预料完毕
pos['mark'] = 1

# #从直方图以及每个文件的平均字数,来设置的最大序列长度值
# numWords = []
# line = neg.readline()
# counter = len(line.split())
# numWords.append(counter)
# import matplotlib.pyplot as plt
# plt.hist(numWords, 50)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.axis([0, 1200, 0, 8000])
# plt.show()


#正负语聊进行合并
pn = pd.concat([pos, neg], ignore_index=True) #pn = neg.append(pos,index = None)
neglen = len(neg)
poslen = len(pos)#计算语料数目
print('negative length:', neglen)
print('postive length:', poslen)

cw = lambda x:list(jieba.cut(x))#定义分词函数
# def cw(x):
#     return list(jieba.cut(x))
# lambda作为一个表达式，定义了一个匿名函数，代码中x为入口参数，list(jieba.cut(x))为函数体
pn['words'] = pn[0].apply(cw)

#读入评论内容
comment = pd.read_excel('sum.xls')
comment = comment[comment['rateContent'].notnull()]#仅读取非空评论
comment['words'] = comment['rateContent'].apply(cw)# 评论分词

d2v_train = pd.concat([pn['words'],comment['words']],ignore_index = True)

w = []  #将所有词语整合在一起
for i in d2v_train:
    w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
del w, d2v_train #del删除的是变量，而不是数据
dict['id'] = list(range(1, len(dict) + 1))

get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent)

maxlen = 50 #截断字数
print("Pad sequence (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))#短于该长度的序列都会在后部填充0以达到该长度。长于nb_timesteps的序列将会被截断，以使其匹配目标长度

print(pn)


x_train = np.array(list(pn['sent']))[::2] #训练集
y_train = np.array(list(pn['mark']))[::2]
x_test = np.array(list(pn['sent']))[1::2] #测试集
y_test = np.array(list(pn['mark']))[1::2]
x_all = np.array(list(pn['sent'])) #全集
y_all = np.array(list(pn['mark']))
print('Build model...')

model = Sequential() #初始化一个神经网络

model.add(Embedding(len(dict)+1, 256, input_length=maxlen))#词向量

#LSTM模型
model.add(LSTM(activation="sigmoid", units=128, recurrent_activation="hard_sigmoid"))#activation="sigmoid"
model.add(Dropout(0.3))#Dropout：防止过拟合
model.add(Dense(1))#Dense:全联接层定义它有1个输出的 feature。同样的，此处不需要再定义输入的维度，因为它接收的是上一层的输出。
model.add(Activation('softmax'))#激活函数有：sigmoid，tanh，ReLUs，Softplus,softmax ...
#目标函数，或称损失函数,binary_crossentropy:对二分类问题,计算在所有预测值上的平均正确率
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])#SGD,RMSprop,Adagrad,Adadelta,Adam,
#model.fit(x_train,y_train,batch_size=16,epochs = 10,verbose=1,validation_data = (x_test,y_test))
score = model.evaluate(x_test, y_test, verbose=1)#通过验证集的数据显示model的性能.

#batch_size：每一次迭代样本数目
#show_accuracy：每个epoch是否显示分类正确率
#verbose: 0 表示不更新日志, 1 更新日志
#validation_data: tuple (X, y) 数据作为验证集. 将加载validation_split.


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
plt.legend(loc='lower right')#loc表示位置的
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
