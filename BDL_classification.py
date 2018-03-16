# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 20:52:01 2018

@author: lenovo-
"""    

import numpy as np # linear algebra
import pandas as pd 
import os
#import sys
from tqdm import tqdm
import tensorflow as tf
import edward as ed
from edward.models import Normal, Categorical
import matplotlib.pyplot as plt

# get data
def get_files(file_dir):
    # file_dir: data path
    # return: image and lable

    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    # read label
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print("There are %d cats\nThere are %d dogs" % (len(cats), len(dogs)))

    # change order
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()     # transposition
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list

# batch
def get_batch(image, label, image_W, image_H, batch_size, capacity):
   
    # image_W, image_H: height and width of the image
    # batch_size
    # capacity: queue length
    # return: batch 

    # change type for tensorflow can understand
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # build queue
    input_queue = tf.train.slice_input_producer([image, label])

    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # change image size 
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)   # standard data
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,   # threads
                                              capacity=CAPACITY)

    image_batch = tf.reshape(image_batch, [batch_size, -1])  #

    label_batch = tf.reshape(label_batch, [batch_size, -1])

    return image_batch, label_batch
    
BATCH_SIZE = 5
CAPACITY = 256
IMG_W = 208
IMG_H = 208
train_dir = "E://ICM/semestre III/PRcode/MR/mini/train"
image_list, label_list = get_files(train_dir)
image_train_batch, label_train_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#label_train_batch.shape
D = image_train_batch.shape[1].value
K = label_train_batch.shape[1].value
#print(K)
# Create a placeholder to hold the data (in minibatches) in a TensorFlow graph.
x = tf.placeholder(tf.float32, [None, D])
# Normal(0,1) priors for the variables. 
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
# Categorical likelihood for classication.
y = Categorical(tf.matmul(x,w)+b)

# Contruct the q(w) and q(b). in this case we assume Normal distributions.
qw = Normal(loc=tf.Variable(tf.random_normal([D, K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qb = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

# use a placeholder for the labels in anticipation of the traning data.
y_ph = tf.placeholder(tf.int32, [BATCH_SIZE])

inference = ed.KLqp({w: qw, b: qb}, data={y: y_ph})
inference.initialize()

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(BATCH_SIZE):
        batch_x, batch_y = sess.run([image_train_batch, label_train_batch])
    batch_y = np.argmax(batch_y,axis=1)
    info_dict = inference.update(feed_dict={x: batch_x, y_ph: batch_y})
    inference.print_progress(info_dict)
    
    coord.request_stop()
    coord.join(threads)
    sess.close()

