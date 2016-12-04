#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# make runs deterministic
np.random.seed(0)
tf.set_random_seed(0)

image_size = 28
num_labels = 10

# configurable parameters
batch_size = 64
num_kernels1 = 4
num_kernels2 = 32
num_hidden = 512
regularization_factor = 1e-4
dropout_keep_probability = 0.5
learning_rate = 0.001
kernel1_size = 5
kernel2_size = 5

# pool twice: image size 28x28 down to 7x7
flat_size = (image_size/4)**2*num_kernels2

conv1_weights = tf.Variable(tf.truncated_normal([kernel1_size, kernel1_size, 1, num_kernels1], stddev=0.1))
conv1_biases  = tf.Variable(tf.constant(0.0, tf.float32, [num_kernels1]))
conv2_weights = tf.Variable(tf.truncated_normal([kernel2_size, kernel2_size, num_kernels1, num_kernels2], stddev=0.1))
conv2_biases  = tf.Variable(tf.constant(0.1, tf.float32, [num_kernels2]))
fc1_weights   = tf.Variable(tf.truncated_normal([flat_size, num_hidden], stddev=0.1))
fc1_biases    = tf.Variable(tf.constant(0.1, tf.float32, [num_hidden]))
fc2_weights   = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
fc2_biases    = tf.Variable(tf.constant(0.1, tf.float32, [num_labels]))

data      = tf.placeholder(tf.float32, [None, image_size**2])
results   = tf.placeholder(tf.float32, [None, num_labels])
keep_prob = tf.placeholder(tf.float32)

X = data
# reshape data to image shape
X = tf.reshape(X, [-1, image_size, image_size, 1])
# conv relu pool
X = tf.nn.conv2d(X, conv1_weights, strides=[1,1,1,1], padding='SAME')
X = tf.nn.relu(tf.nn.bias_add(X, conv1_biases))
X = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#  conv relu pool
X = tf.nn.conv2d(X, conv2_weights, strides=[1,1,1,1], padding='SAME')
X = tf.nn.relu(tf.nn.bias_add(X, conv2_biases))
X = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# flatten data to row shape
X = tf.reshape(X, [-1, flat_size])
# relu(XW + b)
X = tf.nn.relu(tf.matmul(X, fc1_weights) + fc1_biases)
# dropout
X = tf.nn.dropout(X, keep_prob)
# XW + b
X = tf.matmul(X, fc2_weights) + fc2_biases
Y = X

# regularization to keep weights small
regularization = (
	tf.nn.l2_loss(fc1_weights) +
	tf.nn.l2_loss(fc2_weights) +
	tf.nn.l2_loss(fc1_biases ) +
	tf.nn.l2_loss(fc2_biases ))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, results))
loss += regularization_factor*regularization
correct_prediction = tf.equal(tf.argmax(results, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

mnist = read_data_sets("../mnist", one_hot=True)

for batch in range(10000+1):
	batch_data, batch_results = mnist.train.next_batch(batch_size)
	feed_dict = {data:batch_data, results:batch_results, keep_prob:dropout_keep_probability}
	sess.run([train], feed_dict=feed_dict)
	
	if batch % 100 == 0:
		feed_dict = {data:mnist.test.images, results:mnist.test.labels, keep_prob:1.0}
		acc, = sess.run([accuracy], feed_dict=feed_dict)
		print("batch %5d: %f accuracy"%(batch, acc))
