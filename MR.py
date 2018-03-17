# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 20:52:01 2018

@author: lenovo-
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
from tqdm import tqdm
import tensorflow as tf
import edward as ed
from edward.models import Normal, Categorical
import matplotlib.pyplot as plt

data = pd.read_csv('mushrooms.csv')
#data.head(n=5)
#data.shape
#data.isnull().sum()#filter missing data.
#data.dtypes #view data types

data2 = pd.get_dummies(data)
#data2.head(n=5)
#data2.shape

data2['class_e'].sum() / data.shape[0] # class rate

data_x = data2.loc[:, 'cap-shape_b':].as_matrix().astype(np.float32)  #column from cap-shape_b to the end
data_y = data2.loc[:, :'class_p'].as_matrix().astype(np.float32)   #two first columns

N = 7000
train_x, test_x = data_x[:N], data_x[N:]
train_y, test_y = data_y[:N], data_y[N:]

D = train_x.shape[1]  # nombre of features
K = train_y.shape[1]  #nombre of class


EPOCH_NUM = 500   
batch = 100  #batch

# for bayesian neural network
train_y2 = np.argmax(train_y, axis=1)
test_y2 = np.argmax(test_y, axis=1)  #index of max 

x_ = tf.placeholder(tf.float32, shape=(None, D))
y_ = tf.placeholder(tf.int32, shape=(batch))
# Normal(0,1) priors for the variables. 
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros([K]), scale=tf.ones([K]))
y_pre = Categorical(tf.matmul(x_, w) + b)

qw = Normal(loc=tf.Variable(tf.random_normal([D, K])), scale=tf.Variable(tf.random_normal([D, K])))
qb = Normal(loc=tf.Variable(tf.random_normal([K])), scale=tf.Variable(tf.random_normal([K])))

y = Categorical(tf.matmul(x_, qw) + qb)

inference = ed.KLqp({w: qw, b: qb}, data={y_pre: y_})
#inference.initialize()
inference.initialize(n_iter=500, n_print=100, scale={y: float(N) / batch})


sess = tf.Session()
sess.run(tf.global_variables_initializer())

samples_num = 100
"""loss = []
acc = []

with sess:
    perm = np.random.permutation(N) #disorganize the data
    for i in range(inference.n_iter): # from 0 to N interval is Batch size
        batch_x = train_x[perm[i:i+batch]]
        batch_y = train_y2[perm[i:i+batch]]
        info_dict = inference.update(feed_dict={x_: batch_x, y_: batch_y})
        inference.print_progress(info_dict)
        if i % 100 == 0:
            y_samples = y.sample(samples_num).eval(feed_dict={x_: train_x})
            acc = (np.round(y_samples.sum(axis=0) / samples_num) == train_y2).mean()
            acc.append(acc)
        
    plt.figure()
    plt.plot(loss)
    plt.xlabel('interation')
    plt.ylabel('loss value')
    
    plt.figure()
    plt.plot(acc)
    plt.xlabel('interation')
    plt.ylabel('acc')
    plt.show() """

with sess:
    for epoch in tqdm(range(EPOCH_NUM), file=sys.stdout): #print progress
        perm = np.random.permutation(N) #disorganize the data
        for i in range(0, N, batch): #from 0 to N interval is Batch size
            batch_x = train_x[perm[i:i+batch]]
            batch_y = train_y2[perm[i:i+batch]]
            inference.update(feed_dict={x_: batch_x, y_: batch_y})
           # inference.print_progress(info_dict)
        y_samples = y.sample(samples_num).eval(feed_dict={x_: train_x})
        acc = (np.round(y_samples.sum(axis=0) / samples_num) == train_y2).mean()
        y_samples = y.sample(samples_num).eval(feed_dict={x_: test_x})
        test_acc = (np.round(y_samples.sum(axis=0) / samples_num) == test_y2).mean()
        if (epoch+1) % 50 == 0:
            tqdm.write('epoch:\t{}\taccuracy:\t{}\tvaridation accuracy:\t{}'.format(epoch+1, acc, test_acc))

