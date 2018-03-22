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
data.head(n=5)
#data.shape
#data.isnull().sum()#filter missing data.
#data.dtypes #view data types

data2 = pd.get_dummies(data)
data2.head(n=5)
#data2.shape

data2['class_e'].sum() / data.shape[0] # class rate

data_x = data2.loc[:, 'cap-shape_b':].as_matrix().astype(np.float32)  #column from cap-shape_b to the end
data_y = data2.loc[:, :'class_p'].as_matrix().astype(np.float32)   #two first columns

N = 7000
train_x, test_x = data_x[:N], data_x[N:]
train_y, test_y = data_y[:N], data_y[N:]
#train_x
D = train_x.shape[1]  # nombre of features
K = train_y.shape[1]  #nombre of class

EPOCH_NUM = 100  
batch = 100  #batch

# for bayesian neural network
train_y2 = np.argmax(train_y, axis=1)
test_y2 = np.argmax(test_y, axis=1)  #index of max 

x_ = tf.placeholder(tf.float32, shape=(None, D))
y_ = tf.placeholder(tf.int32, shape=(batch))
keep_prob = tf.placeholder(tf.float32) 
 
# Normal(0,1) priors for the variables. 
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros([K]), scale=tf.ones([K]))
Wx_plus_b = tf.matmul(x_, w) + b  
#to dropout
Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)  
y_pre = Categorical(Wx_plus_b)

qw = Normal(loc=tf.Variable(tf.random_normal([D, K])), scale=tf.Variable(tf.random_normal([D, K])))
qb = Normal(loc=tf.Variable(tf.random_normal([K])), scale=tf.Variable(tf.random_normal([K])))

y = Categorical(tf.matmul(x_, qw) + qb)

inference = ed.KLqp({w: qw, b: qb}, data={y_pre: y_})
#inference.initialize()
inference.initialize(n_iter=1000, n_print=100, scale={y: float(N) / batch})


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# summary writer goes in here  
#train_writer = tf.summary.FileWriter("logs/train",sess.graph)  
#test_writer = tf.summary.FileWriter("logs/test",sess.graph)  

with sess:   
    samples_num = 400 
    accy_train = []
    accy_test = []
    
    for epoch in tqdm(range(EPOCH_NUM)):
        # Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
        perm = np.random.permutation(N) #disorganize the data
        for i in range(inference.n_iter): # from 0 to N interval is Batch size
            batch_x = train_x[perm[i:i+batch]]
            batch_y = train_y2[perm[i:i+batch]]    
        info_dict = inference.update(feed_dict={x_: batch_x, y_: batch_y, keep_prob: 0.5})
        inference.print_progress(info_dict)


    y_samples1 = y.sample(samples_num).eval(feed_dict={x_: train_x, keep_prob: 1})
    y_samples = y.sample(samples_num).eval(feed_dict={x_: test_x, keep_prob: 1})    
  
   
    for i in range(samples_num):
        acc = (y_samples[i] == test_y2).mean()*100
        temp = (y_samples1[i] == train_y2).mean()*100
        accy_test.append(acc)
        accy_train.append(temp)  
    
            
    plt.hist(accy_test)
    plt.title("Histogram of prediction accuracies in the test data")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    
    test_mushroom = test_x[0:1]
    test_label = test_y2[0]
    print('0:edible, 1:poisonous')
    print('truth = ',test_label)
    
    
    w_samples = []
    b_samples = []
    for _ in range(samples_num):
        w_samp = qw.sample()
        b_samp = qb.sample()
        w_samples.append(w_samp)
        b_samples.append(b_samp)
    
    # Now the check what the model perdicts for each (w,b) sample from the posterior. 
    sing_mushroom_probs = []
    for w_samp,b_samp in zip(w_samples,b_samples):
        prob = tf.nn.softmax(tf.matmul( test_x[0:1],w_samp ) + b_samp)
        sing_mushroom_probs.append(prob.eval())
    
    # Create a histogram of these predictions.
    plt.hist(np.argmax(sing_mushroom_probs,axis=2),bins=range(3))
    plt.xticks(np.arange(0,3))
    plt.xlim(0,3)
    plt.xlabel("Accuracy of the prediction of the test digit")
    plt.ylabel("Frequency")

"""with sess:
    for epoch in tqdm(range(EPOCH_NUM)):
        perm = np.random.permutation(N) #disorganize the data
        for i in range(0, N, batch): #from 0 to N interval is Batch size
            batch_x = train_x[perm[i:i+batch]]
            batch_y = train_y2[perm[i:i+batch]]
            inference.update(feed_dict={x_: batch_x, y_: batch_y, keep_prob: 0.5})
           # inference.print_progress(info_dict)
        y_samples = y.sample(samples_num).eval(feed_dict={x_: train_x, keep_prob: 1})
        acc = (np.round(y_samples.sum(axis=0) / samples_num) == train_y2).mean()
        #acc.append(temp)
        y_samples = y.sample(samples_num).eval(feed_dict={x_: test_x, keep_prob: 1})
        test_acc = (np.round(y_samples.sum(axis=0) / samples_num) == test_y2).mean()
        #accy_test.append(test_acc)
        
        if (epoch+1) % 10 == 0:
           tqdm.write('epoch:\t{}\taccuracy:\t{}\tvaridation accuracy:\t{}'.format(epoch+1, acc, test_acc))
    plt.hist(accy_test)
    plt.title("Histogram of prediction accuracies in the test data")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency") """




