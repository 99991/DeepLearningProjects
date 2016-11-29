#!/usr/bin/env python

# based on
# https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html

# 97% accuracy withc batch_size = 10
# 98% accuracy withc batch_size = 100

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

image_size = 28*28
n_labels = 10
batch_size = 100

mnist = read_data_sets("../mnist", one_hot=True)

def make_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1.0/np.sqrt(shape[0])))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# so many weights
W_conv1 = make_weights([5, 5, 1, 8])
b_conv1 = make_weights([8])
W_conv2 = make_weights([5, 5, 8, 64])
b_conv2 = make_weights([64])
W_fc1 = make_weights([7 * 7 * 64, 1024])
b_fc1 = make_weights([1024])
W_fc2 = make_weights([1024, 10])
b_fc2 = make_weights([10])

# placeholders
x = tf.placeholder(tf.float32, [None, image_size])
y = tf.placeholder(tf.float32, [None, n_labels])
# this is for dropout
keep_prob = tf.placeholder(tf.float32)

# we get vector data, but conv layer needs image-shaped data
x_image = tf.reshape(x, [-1, 28, 28, 1])

# pool(relu(conv(X, W) + b))
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# another one
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# flatten the images back to vectors
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# relu(XW + b) layer
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropout layer
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# XW + b layer
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# softmax cross entropy loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y))

# accuracy testing network
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# adam optimizer is best optimizer (maybe)
train = tf.train.AdamOptimizer().minimize(loss)

# tensorflow magic
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

for epoch in range(1000+1):
    print("epoch %d"%epoch)
    # train
    batch_train_x, batch_train_y = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict={x: batch_train_x, y: batch_train_y, keep_prob: 0.5})
    
    if epoch % 100 == 0:
        # calculate accuracy
        feed_dict = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}
        acc = sess.run(accuracy, feed_dict=feed_dict)
        print("%f accuracy"%acc)
        
