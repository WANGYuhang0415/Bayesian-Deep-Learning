# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 20:52:01 2018

@author: lenovo-yuhang
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import sys
from tqdm import tqdm
import tensorflow as tf
import edward as ed
from edward.models import Normal, Categorical
import matplotlib.pyplot as plt

data = pd.read_csv('E:\ICM\semestre III\PRcode\mushroom\Bayesian-Deep-Learning\mushrooms.csv')
#data.head(n=5)

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


EPOCH_NUM = 100  
batch = 100  #batch

# for bayesian neural network
train_y2 = np.argmax(train_y, axis=1)
test_y2 = np.argmax(test_y, axis=1)  #index of max 



x_ = tf.placeholder(tf.float32, [None, D])
y_ = tf.placeholder(tf.int32, shape=(batch))
keep_prob = tf.placeholder(tf.float32) 
#n_hidden = 10
H = 10


W_0 = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
W_1 = Normal(loc=tf.zeros([H, 10]), scale=tf.ones([H, 10]))
b_0 = Normal(loc=tf.zeros(H), scale=tf.ones(H))
b_1 = Normal(loc=tf.zeros(10), scale=tf.ones(10))

y_pre = Categorical(logits=tf.matmul(tf.nn.tanh(tf.matmul(x_, W_0) + b_0), W_1) + b_1)

with tf.variable_scope("posterior"):
    tf.get_variable_scope().reuse_variables()
    with tf.variable_scope("qW_0"):
      loc = tf.get_variable("loc", [D, H])
      scale = tf.nn.softplus(tf.get_variable("scale", [D, H]))
      qW_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_1"):
      loc = tf.get_variable("loc", [H, 10])
      scale = tf.nn.softplus(tf.get_variable("scale", [H, 10]))
      qW_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_0"):
      loc = tf.get_variable("loc", [H])
      scale = tf.nn.softplus(tf.get_variable("scale", [H]))
      qb_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_1"):
      loc = tf.get_variable("loc", [10])
      scale = tf.nn.softplus(tf.get_variable("scale", [10]))
      qb_1 = Normal(loc=loc, scale=scale)


inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1}, data={y_pre: y_})
    
inference.initialize(n_iter=1000)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

with sess: 
    samples_num = 100
    
    for epoch in tqdm(range(EPOCH_NUM)):
        # Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
        perm = np.random.permutation(N) #disorganize the data
        for i in range(inference.n_iter): # from 0 to N interval is Batch size
            batch_x = train_x[perm[i:i+batch]]
            batch_y = train_y2[perm[i:i+batch]]    
        #info_dict = inference.update(feed_dict={x_: batch_x, y_: batch_y, keep_prob: 0.5})
        info_dict = inference.update(feed_dict={x_: batch_x, y_: batch_y})
        inference.print_progress(info_dict)

    prob_lst = []
    samples = []
    W0_samples = []
    W1_samples = []
    b0_samples = []
    b1_samples = []
    for _ in range(samples_num):
        W0_samp = qW_0.sample()
        W1_samp = qW_1.sample()
        b0_samp = qb_0.sample()
        b1_samp = qb_1.sample()
        W0_samples.append(W0_samp)
        W1_samples.append(W1_samp)
        b0_samples.append(b0_samp)
        b1_samples.append(b1_samp)
    
        # Also compue the probabiliy of each class for each (w,b) sample.
        prob = tf.nn.softmax(tf.matmul(tf.nn.tanh(tf.matmul(test_x, W0_samp) + b0_samp), W1_samp) + b1_samp)
        prob_lst.append(prob.eval())
    
    accy_test = []
    for prob in prob_lst:
        #print(prob_lst[prob].astype(np.float32))
        y_test_prd = np.argmax(prob,axis=1).astype(np.float32)
        #print(y_trn_prd)
        acc = (y_test_prd == test_y2).mean()*100
        accy_test.append(acc)

    plt.hist(accy_test)
    plt.title("Histogram of prediction accuracies in the test data")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")


   
    


