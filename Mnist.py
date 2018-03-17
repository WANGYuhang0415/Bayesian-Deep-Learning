# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:12:21 2018

@author: lenovo-
"""  
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal
import edward as ed
import pandas as pd

# Use the TensorFlow method to download the data.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

N = 100   # batch size.
D = 784   # number of features.
K = 10    # number of classes.

# Create a placeholder 
x = tf.placeholder(tf.float32, [None, D])
# Normal(0,1) priors for the variables
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
# Categorical likelihood for classication.
y = Categorical(tf.matmul(x,w)+b)


qw = Normal(loc=tf.Variable(tf.random_normal([D, K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qb = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))


y_pre = tf.placeholder(tf.int32, [N])
#minimise the KL divergence between q and p.
inference = ed.KLqp({w: qw, b: qb}, data={y:y_pre})

inference.initialize()

sess = tf.InteractiveSession()
# Initialise all the vairables in the session.
tf.global_variables_initializer().run()

for _ in range(1000):
    X_batch, Y_batch = mnist.train.next_batch(N)   
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict = inference.update(feed_dict={x: X_batch, y_pre: Y_batch})
    inference.print_progress(info_dict)
    
# Load the test images.
X_test = mnist.test.images
# TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
Y_test = np.argmax(mnist.test.labels,axis=1)

